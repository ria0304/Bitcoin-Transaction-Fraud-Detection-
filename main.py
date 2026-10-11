"""
main.py
=======
Entry-point for the Bitcoin Fraud Detection pipeline 

Pipeline
--------
1.  Load & engineer features (src/data.py)
2.  Build PyG graph (src/data.py)
3.  [Optional] Self-supervised pre-training of ElliGAT
4.  Fine-tune all GNN models (ElliGAT, EvolveGCN, BaselineGNN, ROLAND, TGN) — 5 seeds
5.  Train tabular baselines (MLP, XGBoost, LightGBM, RF) — 5 seeds
6.  Fit MetaEnsemble stacker on validation predictions
7.  Aggregate results, run Wilcoxon tests
8.  Generate & save results figure
9.  Save best model weights + metrics JSON
10. Pseudo-Label Propagation (PLP — Novel Contribution #3)
11. GNNExplainer motif attribution (Novel Contribution #4)

Novel Contributions (Journal)
-----------------------------
NC1: FAGA — Fraud-Aware Graph Augmentation (src/augmentation.py)
NC2: AUGT — Adaptive Uncertainty-Gated Threshold + AURC (src/uncertainty.py)
NC3: PLP  — Pseudo-Label Propagation on unlabelled nodes (src/pseudo_label.py)
NC4: GNNExplainer fraud motif attribution (src/explain.py)

Run
---
    python main.py

Environment variables (set in .env or shell)
---------------------------------------------
    BITCOIN_DATA_PATH  – local path to Elliptic dataset folder
    KAGGLE_USERNAME    – Kaggle username (only needed for auto-download)
    KAGGLE_KEY         – Kaggle API key  (only needed for auto-download)
    OUTPUT_DIR         – where to save outputs (default: ./outputs)
    EPOCHS             – override training epochs
    HIDDEN_DIM         – override hidden dimension
    LEARNING_RATE      – override learning rate
    SEEDS              – comma-separated list of seeds, e.g. 42,123,2024
    METHODS            – which methods to run (default: all). Comma list of
                         ElliGAT,EvolveGCN,BaselineGNN,ROLAND,TGN,Tabular,Meta
                         ElliGAT also runs MC-dropout, PLP, GNNExplainer, heterophily.
                         Meta needs ElliGAT in the same run OR ELLIGAT_PROBS_DIR
                         pointing at a folder with elligat_probs_seed<N>.npz.
"""

import os
import gc
import json
import random
import warnings
warnings.filterwarnings("ignore")

# ─── Load .env file before anything else ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can be set directly in the shell

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

from configs.config import (
    SEEDS, HIDDEN_DIM, NUM_GAT_LAYERS, NUM_HEADS, DROPOUT, EDGE_DIM,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE, LR_WARMUP_STEPS,
    MC_SAMPLES, FOCAL_ALPHA, FOCAL_GAMMA,
    PRETRAIN_EPOCHS, MASK_RATIO,
    CONTRASTIVE_WEIGHT,
    LOSS_NAME, ASL_GAMMA_POS, ASL_GAMMA_NEG, ASL_CLIP,
    LOG_FILE, LOG_LEVEL,
    PLP_EPOCHS, EXPLAIN_NODES, EXPLAIN_EPOCHS,
    DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH,
    OUTPUT_DIR, BEST_MODEL_PATH, RESULTS_PLOT_PATH, METRICS_JSON_PATH,
)
from src.data import (
    find_dataset_path, load_elliptic,
    add_velocity_features, chronological_split,
    build_graph, VELOCITY_COLS,
)
from src.models import ElliGAT, EvolveGCN, BaselineGNN
from src.baselines import ROLAND, TGNMemory, TabularBaselines, MetaEnsemble
from src.trainer import (
    pretrain, train_gnn, evaluate_gnn,
    mc_uncertainty, compute_metrics,
)
from src.pseudo_label import run_plp
from src.explain import explain_fraud_nodes, top_fraud_features
from src.visualize import save_results_figure

os.makedirs(OUTPUT_DIR, exist_ok=True)
from src.logger import get_logger
logger = get_logger("elligat", log_file=LOG_FILE, level=LOG_LEVEL)
logger.info(f"Novelty: ElliGAT-ASL loss={LOSS_NAME} gamma_pos={ASL_GAMMA_POS} gamma_neg={ASL_GAMMA_NEG}")

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Bitcoin Fraud Detection — ElliGAT")
print(f"  Device : {device}")
print(f"  Output : {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ─── 1. Load dataset ─────────────────────────────────────────────────────────
print("Step 1: Loading Elliptic Bitcoin dataset …")
dataset_path = find_dataset_path(DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH)
if dataset_path is None:
    raise FileNotFoundError(
        "Elliptic dataset not found.\n"
        "  Option A: set BITCOIN_DATA_PATH in your .env file.\n"
        "  Option B: set KAGGLE_USERNAME + KAGGLE_KEY in .env for auto-download.\n"
        "  See .env.example for details."
    )
print(f"  Path: {dataset_path}")

df, X_all, edge_index, edge_attr, txid_to_idx, all_nodes = load_elliptic(dataset_path)

# ─── 2. Feature engineering ──────────────────────────────────────────────────
print("\nStep 2: Engineering velocity features …")
df = add_velocity_features(df)

feat_cols = [c for c in df.columns if c.startswith("f")]   # 166 raw Elliptic feats

# ─── 3. Chronological split ──────────────────────────────────────────────────
print("\nStep 3: Chronological train/val/test split (70/15/15) …")
train_df, val_df, test_df, train_size, val_size = chronological_split(df)
print(
    f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}"
)

# ─── 4. Tabular feature matrices ─────────────────────────────────────────────
tabular_cols = VELOCITY_COLS + feat_cols
print(f"  Tabular features: {len(tabular_cols)} (7 velocity + {len(feat_cols)} raw)")

X_train_raw = df.iloc[:train_size][tabular_cols].values
X_val_raw   = df.iloc[train_size:train_size + val_size][tabular_cols].values
X_test_raw  = df.iloc[train_size + val_size:][tabular_cols].values
y_train_np  = df.iloc[:train_size]["isFraud"].values.astype(np.float32)
y_val_np    = df.iloc[train_size:train_size + val_size]["isFraud"].values.astype(np.float32)
y_test_np   = df.iloc[train_size + val_size:]["isFraud"].values.astype(np.float32)

scaler_tab  = StandardScaler().fit(X_train_raw)
X_train_s   = scaler_tab.transform(X_train_raw)
X_val_s     = scaler_tab.transform(X_val_raw)
X_test_s    = scaler_tab.transform(X_test_raw)

n_pos = float(y_train_np.sum())
n_neg = float(len(y_train_np) - n_pos)
print(f"  Class balance — Fraud: {n_pos:.0f} ({n_pos/len(y_train_np):.3%})")

# ─── 5. Build PyG graph ──────────────────────────────────────────────────────
print("\nStep 4: Building Bitcoin transaction graph …")
graph_data, _, _ = build_graph(
    df, X_all, edge_index, edge_attr, txid_to_idx,
    train_size, val_size, feat_cols,
)
graph_data = graph_data.to(device)
in_dim = graph_data.num_node_features
print(
    f"  Nodes: {graph_data.num_nodes:,}  "
    f"Edges: {graph_data.num_edges:,}  "
    f"Features/node: {in_dim}"
)


# ─── Method selection (METHODS env var; default = everything) ────────────────
_ALL_METHODS = ["ElliGAT", "EvolveGCN", "BaselineGNN", "ROLAND", "TGN", "Tabular", "Meta"]
_raw_methods = os.environ.get("METHODS", "all").strip()
METHODS = (
    list(_ALL_METHODS) if _raw_methods.lower() in ("", "all")
    else [m.strip() for m in _raw_methods.split(",") if m.strip()]
)
_unknown = [m for m in METHODS if m not in _ALL_METHODS]
if _unknown:
    raise ValueError(f"Unknown METHODS {_unknown}; choose from {_ALL_METHODS}")


def want(name: str) -> bool:
    return name in METHODS


ELLIGAT_PROBS_DIR = os.environ.get("ELLIGAT_PROBS_DIR", OUTPUT_DIR)
print(f"  Methods to run: {METHODS}")

# ─── 6. Multi-seed experiments ───────────────────────────────────────────────
print(f"\nStep 5: Multi-seed experiments ({len(SEEDS)} seeds) …\n")

GNN_MODELS  = ["ElliGAT", "EvolveGCN", "BaselineGNN", "ROLAND", "TGN"]
TAB_MODELS  = ["MLP", "XGBoost", "LightGBM", "RandomForest"]
ALL_MODELS  = GNN_MODELS + TAB_MODELS + ["MetaEnsemble"]

results_per_seed: dict[str, list] = {m: [] for m in ALL_MODELS}

GNN_BASELINES = {
    "EvolveGCN":   ("EvolveGCN — temporal GNN baseline",
                    lambda: EvolveGCN(in_dim, HIDDEN_DIM, dropout=DROPOUT)),
    "BaselineGNN": ("BaselineGNN — GraphSAGE ablation",
                    lambda: BaselineGNN(in_dim, hidden_dim=128, dropout=DROPOUT)),
    "ROLAND":      ("ROLAND — Relational temporal GNN, Wang et al. 2022",
                    lambda: ROLAND(in_dim, hidden_dim=128, dropout=DROPOUT)),
    "TGN":         ("TGN — Temporal Graph Network, Rossi et al. 2020",
                    lambda: TGNMemory(in_dim, hidden_dim=128, dropout=DROPOUT)),
}

last_ellgat_model  = None
last_graph_data    = None
last_results_final = {}

for seed in SEEDS:
    print(f"\n{'─'*60}")
    print(f"  SEED {seed}")
    print(f"{'─'*60}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    gnn_val_probs = gnn_test_probs = None

    # ── ElliGAT ─────────────────────────────────────────────────────────────
    if want("ElliGAT"):
        print("\n  [ElliGAT — 4-layer GATv2 + temporal edges + heterophily]")
        ellgat = ElliGAT(
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_GAT_LAYERS,
            heads=NUM_HEADS,
            dropout=DROPOUT,
            edge_dim=EDGE_DIM,
        ).to(device)

        print("    Phase 1: Self-supervised pre-training …")
        pretrain(ellgat, graph_data, epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO, verbose=False)

        print("    Phase 2: Fine-tuning with combined loss …")
        train_gnn(
            ellgat, graph_data, pw,
            epochs=EPOCHS, lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, patience=PATIENCE,
            warmup_steps=LR_WARMUP_STEPS,
            lambda_c=CONTRASTIVE_WEIGHT, lambda_p=0.10,
            verbose=True, loss_name=LOSS_NAME, logger=logger,
        )
        ea = evaluate_gnn(ellgat, graph_data)
        results_per_seed["ElliGAT"].append(ea)
        print(f"    → AUC={ea['ROC-AUC']:.4f}  F1={ea['F1']:.4f}  MCC={ea['MCC']:.4f}")
        logger.info(f"[ElliGAT seed {seed}] AUC={ea['ROC-AUC']:.4f} F1={ea['F1']:.4f} MCC={ea['MCC']:.4f}")

        # Pre-PLP probabilities: used by MetaEnsemble (in this run or a later job)
        ellgat.eval()
        with torch.no_grad():
            _logits = ellgat(graph_data)
            gnn_val_probs  = torch.sigmoid(_logits[graph_data.val_mask]).cpu().numpy()
            gnn_test_probs = torch.sigmoid(_logits[graph_data.test_mask]).cpu().numpy()
        np.savez(
            os.path.join(OUTPUT_DIR, f"elligat_probs_seed{seed}.npz"),
            val=gnn_val_probs, test=gnn_test_probs,
        )
        last_ellgat_model = ellgat
        last_graph_data   = graph_data

    # ── GNN baselines (each one is independent) ─────────────────────────────
    for key, (title, build) in GNN_BASELINES.items():
        if not want(key):
            continue
        print(f"\n  [{title}]")
        torch.manual_seed(seed)
        model = build().to(device)
        train_gnn(
            model, graph_data, pw,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, warmup_steps=LR_WARMUP_STEPS,
            lambda_c=0, lambda_p=0, verbose=False,
        )
        res = evaluate_gnn(model, graph_data)
        results_per_seed[key].append(res)
        print(f"    → AUC={res['ROC-AUC']:.4f}  F1={res['F1']:.4f}  MCC={res['MCC']:.4f}")
        del model
        gc.collect()

    # ── Tabular baselines ────────────────────────────────────────────────────
    tab = None
    if want("Tabular") or want("Meta"):
        print("\n  [Tabular baselines]")
        tab = TabularBaselines(random_state=seed)
        tab.fit(X_train_s, y_train_np, X_val_s, y_val_np, n_pos=n_pos, n_neg=n_neg)
        tab_results = tab.evaluate(X_test_s, y_test_np)
        if want("Tabular"):
            for name, res in tab_results.items():
                results_per_seed[name].append(res)

    # ── MetaEnsemble ────────────────────────────────────────────────────────
    if want("Meta"):
        print("\n  [MetaEnsemble — stacked GNN + tabular]")
        if gnn_val_probs is None:
            _p = os.path.join(ELLIGAT_PROBS_DIR, f"elligat_probs_seed{seed}.npz")
            if not os.path.exists(_p):
                raise FileNotFoundError(
                    f"Meta needs ElliGAT probabilities but {_p} was not found. "
                    "Run the ElliGAT method first (same seed) and point "
                    "ELLIGAT_PROBS_DIR at its output folder."
                )
            _z = np.load(_p)
            gnn_val_probs, gnn_test_probs = _z["val"], _z["test"]
            print(f"    Loaded ElliGAT probabilities from {_p}")

        tab_val_probs  = tab.predict_proba(X_val_s)
        tab_test_probs = tab.predict_proba(X_test_s)

        meta = MetaEnsemble()
        meta.fit(gnn_val_probs, tab_val_probs, y_val_np)
        me_res = meta.evaluate(gnn_test_probs, tab_test_probs, y_test_np)
        results_per_seed["MetaEnsemble"].append(me_res)
        print(f"    → AUC={me_res['ROC-AUC']:.4f}  F1={me_res['F1']:.4f}  MCC={me_res['MCC']:.4f}")

    gc.collect()

# ─── 7. Aggregate results ────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"AGGREGATED RESULTS (mean ± std, {len(SEEDS)} seed(s))")
print("="*60)

METRICS = ["ROC-AUC", "F1", "MCC", "Balanced Acc", "Precision", "Recall"]

aggs: dict[str, dict] = {}
for model_name, res_list in results_per_seed.items():
    if not res_list:
        continue
    aggs[model_name] = {}
    print(f"\n  {model_name}")
    for k in METRICS:
        vals = [r[k] for r in res_list]
        aggs[model_name][k] = (np.mean(vals), np.std(vals), vals)
        print(f"    {k:<16}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# ─── 8. Wilcoxon significance tests ──────────────────────────────────────────
if "ElliGAT" in aggs:
    print(f"\n\n{'─'*60}")
    print("Wilcoxon Signed-Rank Tests (ElliGAT vs competitors)")
    print("─"*60)
    gnn_aucs = aggs["ElliGAT"]["ROC-AUC"][2]
    for comp in ["MetaEnsemble", "EvolveGCN", "ROLAND", "TGN",
                 "XGBoost", "BaselineGNN", "MLP"]:
        if comp not in aggs or comp == "ElliGAT":
            continue
        comp_aucs = aggs[comp]["ROC-AUC"][2]
        diffs = np.array(gnn_aucs) - np.array(comp_aucs)
        if len(set(diffs.tolist())) > 1:
            _, p = wilcoxon(gnn_aucs, comp_aucs)
            sig = "✅ p<0.05" if p < 0.05 else "⚠  ns"
            print(f"  ElliGAT vs {comp:<15}: p={p:.4f}  {sig}")

# ─── 9. Ablation table (Table 1) ─────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"TABLE 1 — Full Comparison (mean ± std, n={len(SEEDS)} seed(s))")
print("="*70)
print(f"{'Model':<18}{'AUC':>16}{'F1':>16}{'MCC':>16}{'AURC':>12}")
print("─" * 80)
for name in ["MLP", "RandomForest", "XGBoost", "LightGBM",
             "BaselineGNN", "EvolveGCN", "ROLAND", "TGN",
             "ElliGAT", "MetaEnsemble"]:
    if name not in aggs:
        continue
    a    = aggs[name]
    auc  = f"{a['ROC-AUC'][0]:.4f}±{a['ROC-AUC'][1]:.4f}"
    f1   = f"{a['F1'][0]:.4f}±{a['F1'][1]:.4f}"
    mcc  = f"{a['MCC'][0]:.4f}±{a['MCC'][1]:.4f}"
    if "AURC" in a:
        aurc_str = f"{a['AURC'][0]:.4f}±{a['AURC'][1]:.4f}"
    else:
        aurc_str = "    n/a"
    tag = " ★ proposed" if name == "ElliGAT" else (
          " ★ final"    if name == "MetaEnsemble" else "")
    print(f"{name:<18}{auc:>16}{f1:>16}{mcc:>16}{aurc_str:>12}{tag}")

# ─── 10-14. Post-hoc stages (need the trained ElliGAT from THIS run) ─────────
ea_plp = None
expl_results: dict = {}

if last_ellgat_model is not None:
    # ── MC Uncertainty ──────────────────────────────────────────────────────
    print(f"\n\nStep 6: MC Dropout uncertainty estimation ({MC_SAMPLES} samples) …")
    unc_mean, unc_std = mc_uncertainty(last_ellgat_model, last_graph_data, n_samples=MC_SAMPLES)
    unc_y = last_graph_data.y[last_graph_data.test_mask].cpu().numpy()
    print(f"  Mean uncertainty (std): {unc_std.mean():.5f}")
    gc.collect()

    # ── Pseudo-Label Propagation ────────────────────────────────────────────
    print("\n\nStep 6b: Pseudo-Label Propagation (PLP — Novel Contribution #3) …")
    pseudo_mask, pseudo_y = run_plp(
        last_ellgat_model,
        last_graph_data,
        n_mc_samples=MC_SAMPLES,
        confidence_threshold=0.10,
        lambda_pl=0.20,
        epochs=PLP_EPOCHS,
        lr=1e-4,
    )
    print("  Re-evaluating ElliGAT after PLP …")
    ea_plp = evaluate_gnn(last_ellgat_model, last_graph_data)
    print(
        f"  ElliGAT+PLP → AUC={ea_plp['ROC-AUC']:.4f}  "
        f"F1={ea_plp['F1']:.4f}  MCC={ea_plp['MCC']:.4f}"
    )
    if "AURC" in ea_plp:
        print(f"                AURC={ea_plp['AURC']:.4f}  "
              f"AUGT/Coverage={ea_plp.get('AUGT/Coverage', 0.0):.3f}")

    # Save metrics NOW so a crash in the (slow, memory-hungry) explainer
    # cannot throw away finished training results.
    _SKIP_KEYS = {"probs", "true", "aurc_coverages", "aurc_risks"}

    def _write_metrics(expl=None):
        out = {
            "_config": {
                "loss": LOSS_NAME, "seeds": SEEDS, "epochs": EPOCHS,
                "methods": METHODS,
                "novel": ["FAGA", "AUGT", "PLP", "GNNExplainer"],
            },
            **{
                name: {
                    k: {"mean": float(v[0]), "std": float(v[1])}
                    for k, v in aggs[name].items()
                    if k not in _SKIP_KEYS
                }
                for name in aggs
            },
        }
        if ea_plp is not None:
            out["ElliGAT+PLP"] = {
                k: {"mean": float(v), "std": 0.0}
                for k, v in ea_plp.items()
                if k not in _SKIP_KEYS and not isinstance(v, np.ndarray)
            }
        if expl and expl.get("available") and expl.get("n_explained", 0) > 0:
            out["_explainability"] = {
                "n_explained":         expl["n_explained"],
                "fidelity_plus_mean":  expl["fidelity_plus_mean"],
                "fidelity_plus_std":   expl["fidelity_plus_std"],
                "fidelity_minus_mean": expl["fidelity_minus_mean"],
                "fidelity_minus_std":  expl["fidelity_minus_std"],
            }
        with open(METRICS_JSON_PATH, "w") as f:
            json.dump(out, f, indent=2)
        return out

    _write_metrics()
    torch.save(last_ellgat_model.state_dict(), BEST_MODEL_PATH)
    print(f"  Saved interim metrics + model → {METRICS_JSON_PATH}")

    # ── GNNExplainer ────────────────────────────────────────────────────────
    print("\n\nStep 6c: GNNExplainer fraud subgraph attribution …")
    gc.collect()
    try:
        expl_results = explain_fraud_nodes(
            last_ellgat_model, last_graph_data,
            n_nodes=EXPLAIN_NODES,
            top_k_edges=10,
            epochs=EXPLAIN_EPOCHS,
        )
        if expl_results.get("available") and expl_results.get("n_explained", 0) > 0:
            top_feats = top_fraud_features(
                expl_results["node_results"],
                feature_names=[f"f{i}" for i in range(last_graph_data.num_node_features)],
                top_k=10,
            )
            print("\n  Top-10 fraud-predictive features (GNNExplainer attribution):")
            for fname, score in top_feats:
                print(f"    {fname:<8}: {score:.4f}")
    except Exception as exc:  # explainer is optional; never lose the main results
        print(f"  ⚠ GNNExplainer failed ({type(exc).__name__}: {exc}); continuing.")
        expl_results = {}

    # ── Heterophily ─────────────────────────────────────────────────────────
    print("\nStep 7: Computing heterophily scores …")
    hetero_scores = last_ellgat_model.heterophily_score(last_graph_data)
    test_hetero   = hetero_scores[last_graph_data.test_mask]
    test_labels   = last_graph_data.y[last_graph_data.test_mask]
    h_fraud = test_hetero[test_labels == 1].mean().item()
    h_licit = test_hetero[test_labels == 0].mean().item()
    print(f"  Heterophily score — Fraud: {h_fraud:.3f}  Licit: {h_licit:.3f}  "
          f"(ratio: {h_fraud / max(h_licit, 1e-8):.2f}×)")

    # ── Figure ──────────────────────────────────────────────────────────────
    print("\nStep 8: Generating result plots (with journal panels) …")
    last_results_final = {
        name: results_per_seed[name][-1]
        for name in results_per_seed
        if results_per_seed[name]
    }
    aurc_data = None
    if "aurc_coverages" in ea_plp and "aurc_risks" in ea_plp:
        aurc_data = {
            "coverages": ea_plp["aurc_coverages"],
            "risks":     ea_plp["aurc_risks"],
            "aurc":      ea_plp.get("AURC", 0.0),
        }
    pseudo_data_for_plot = {
        "mask":             pseudo_mask,
        "y":                pseudo_y,
        "total_unlabelled": int((last_graph_data.y == -1).sum()),
    }
    hetero_data = {"scores": test_hetero, "labels": test_labels}
    try:
        save_results_figure(
            last_results_final, aggs, unc_std, unc_y, RESULTS_PLOT_PATH,
            aurc_data=aurc_data,
            pseudo_data=pseudo_data_for_plot,
            expl_results=expl_results,
            hetero_data=hetero_data,
        )
    except Exception as exc:
        print(f"  ⚠ Plot skipped ({type(exc).__name__}: {exc}). "
              "Normal when only a subset of methods was run.")

    # ── Final metrics (now including explainability) ────────────────────────
    _write_metrics(expl_results)
    print(f"  Saved model   → {BEST_MODEL_PATH}")
    print(f"  Saved metrics → {METRICS_JSON_PATH}")
    logger.info(f"Saved metrics → {METRICS_JSON_PATH} | loss={LOSS_NAME}")
else:
    print("\n(ElliGAT not in METHODS → skipping MC-dropout / PLP / explainer / plots)")
    _SKIP_KEYS = {"probs", "true", "aurc_coverages", "aurc_risks"}
    metrics_out = {
        "_config": {"loss": LOSS_NAME, "seeds": SEEDS, "epochs": EPOCHS, "methods": METHODS},
        **{
            name: {
                k: {"mean": float(v[0]), "std": float(v[1])}
                for k, v in aggs[name].items()
                if k not in _SKIP_KEYS
            }
            for name in aggs
        },
    }
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics → {METRICS_JSON_PATH}")

# ─── Human-readable summary ──────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")
with open(summary_path, "w") as f:
    f.write(
        f"ElliGAT-J (journal) novelty run\n"
        f"loss={LOSS_NAME}  seeds={SEEDS}  methods={METHODS}\n"
        f"{'='*60}\n\n"
    )
    for name in aggs:
        f.write(f"\n{name}\n")
        for k, v in aggs[name].items():
            if k in _SKIP_KEYS or not isinstance(v, tuple) or len(v) < 2:
                continue
            f.write(f"  {k}: {v[0]:.4f} +/- {v[1]:.4f}\n")
    if ea_plp is not None:
        f.write("\nElliGAT+PLP\n")
        for k, v in ea_plp.items():
            if k in _SKIP_KEYS or isinstance(v, np.ndarray):
                continue
            f.write(f"  {k}: {v:.4f}\n")
print(f"  Saved summary → {summary_path}")
logger.info(f"Saved summary → {summary_path}")

print(f"\n✅  Done. Methods run: {METHODS}\n")
