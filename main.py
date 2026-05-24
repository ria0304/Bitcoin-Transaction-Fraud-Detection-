"""
main.py
=======
Entry-point for the Bitcoin Fraud Detection pipeline.

Pipeline
--------
1.  Load & engineer features (src/data.py)
2.  Build PyG graph (src/data.py)
3.  [Optional] Self-supervised pre-training of ElliGAT
4.  Fine-tune all GNN models (ElliGAT, EvolveGCN, BaselineGNN) — 5 seeds
5.  Train tabular baselines (MLP, XGBoost, LightGBM, RF) — 5 seeds
6.  Fit MetaEnsemble stacker on validation predictions
7.  Aggregate results, run Wilcoxon tests
8.  Generate & save 8-panel results figure
9.  Save best model weights + metrics JSON

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
"""

import os
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
    DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH,
    OUTPUT_DIR, BEST_MODEL_PATH, RESULTS_PLOT_PATH, METRICS_JSON_PATH,
)
from src.data import (
    find_dataset_path, load_elliptic,
    add_velocity_features, chronological_split,
    build_graph, VELOCITY_COLS,
)
from src.models import ElliGAT, EvolveGCN, BaselineGNN
from src.trainer import (
    pretrain, train_gnn, evaluate_gnn,
    mc_uncertainty, compute_metrics,
)
from src.baselines import TabularBaselines, MetaEnsemble
from src.visualize import save_results_figure

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ─── 6. Multi-seed experiments ───────────────────────────────────────────────
print(f"\nStep 5: Multi-seed experiments ({len(SEEDS)} seeds) …\n")

GNN_MODELS  = ["ElliGAT", "EvolveGCN", "BaselineGNN"]
TAB_MODELS  = ["MLP", "XGBoost", "LightGBM", "RandomForest"]
ALL_MODELS  = GNN_MODELS + TAB_MODELS + ["MetaEnsemble"]

results_per_seed: dict[str, list] = {m: [] for m in ALL_MODELS}

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

    # ── ElliGAT ─────────────────────────────────────────────────────────────
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
        verbose=True,
    )
    ea = evaluate_gnn(ellgat, graph_data)
    results_per_seed["ElliGAT"].append(ea)
    print(f"    → AUC={ea['ROC-AUC']:.4f}  F1={ea['F1']:.4f}  MCC={ea['MCC']:.4f}")

    # ── EvolveGCN ───────────────────────────────────────────────────────────
    print("\n  [EvolveGCN — temporal GNN baseline]")
    torch.manual_seed(seed)
    evolve = EvolveGCN(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    train_gnn(
        evolve, graph_data, pw,
        epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        patience=PATIENCE, warmup_steps=LR_WARMUP_STEPS,
        lambda_c=0, lambda_p=0, verbose=False,
    )
    ev = evaluate_gnn(evolve, graph_data)
    results_per_seed["EvolveGCN"].append(ev)
    print(f"    → AUC={ev['ROC-AUC']:.4f}  F1={ev['F1']:.4f}  MCC={ev['MCC']:.4f}")

    # ── BaselineGNN ─────────────────────────────────────────────────────────
    print("\n  [BaselineGNN — GraphSAGE ablation]")
    torch.manual_seed(seed)
    base = BaselineGNN(in_dim, hidden_dim=128, dropout=DROPOUT).to(device)
    train_gnn(
        base, graph_data, pw,
        epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        patience=PATIENCE, warmup_steps=LR_WARMUP_STEPS,
        lambda_c=0, lambda_p=0, verbose=False,
    )
    ba = evaluate_gnn(base, graph_data)
    results_per_seed["BaselineGNN"].append(ba)
    print(f"    → AUC={ba['ROC-AUC']:.4f}  F1={ba['F1']:.4f}  MCC={ba['MCC']:.4f}")

    # ── Tabular baselines ────────────────────────────────────────────────────
    print("\n  [Tabular baselines]")
    tab = TabularBaselines(random_state=seed)
    tab.fit(X_train_s, y_train_np, X_val_s, y_val_np, n_pos=n_pos, n_neg=n_neg)
    tab_results = tab.evaluate(X_test_s, y_test_np)
    for name, res in tab_results.items():
        results_per_seed[name].append(res)

    # ── MetaEnsemble ────────────────────────────────────────────────────────
    print("\n  [MetaEnsemble — stacked GNN + tabular]")
    ellgat.eval()
    with torch.no_grad():
        gnn_val_logits  = ellgat(graph_data)
        gnn_val_probs   = torch.sigmoid(
            gnn_val_logits[graph_data.val_mask]
        ).cpu().numpy()
        gnn_test_logits = ellgat(graph_data)
        gnn_test_probs  = torch.sigmoid(
            gnn_test_logits[graph_data.test_mask]
        ).cpu().numpy()

    tab_val_probs  = tab.predict_proba(X_val_s)
    tab_test_probs = tab.predict_proba(X_test_s)

    meta = MetaEnsemble()
    meta.fit(gnn_val_probs, tab_val_probs, y_val_np)
    me_res = meta.evaluate(gnn_test_probs, tab_test_probs, y_test_np)
    results_per_seed["MetaEnsemble"].append(me_res)
    print(f"    → AUC={me_res['ROC-AUC']:.4f}  F1={me_res['F1']:.4f}  MCC={me_res['MCC']:.4f}")

    last_ellgat_model = ellgat
    last_graph_data   = graph_data

# ─── 7. Aggregate results ────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("AGGREGATED RESULTS (mean ± std, 5 seeds)")
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
print(f"\n\n{'─'*60}")
print("Wilcoxon Signed-Rank Tests (ElliGAT vs competitors)")
print("─"*60)
gnn_aucs = aggs["ElliGAT"]["ROC-AUC"][2]
for comp in ["MetaEnsemble", "EvolveGCN", "XGBoost", "BaselineGNN", "MLP"]:
    if comp not in aggs or comp == "ElliGAT":
        continue
    comp_aucs = aggs[comp]["ROC-AUC"][2]
    diffs = np.array(gnn_aucs) - np.array(comp_aucs)
    if len(set(diffs.tolist())) > 1:
        _, p = wilcoxon(gnn_aucs, comp_aucs)
        sig = "✅ p<0.05" if p < 0.05 else "⚠  ns"
        print(f"  ElliGAT vs {comp:<15}: p={p:.4f}  {sig}")

# ─── 9. Ablation table ───────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("TABLE 1 — Ablation Study (mean ± std, n=5 seeds)")
print("="*60)
print(f"{'Model':<18}{'AUC':>16}{'F1':>16}{'MCC':>16}")
print("─" * 66)
for name in ["MLP", "RandomForest", "XGBoost", "LightGBM",
             "BaselineGNN", "EvolveGCN", "ElliGAT", "MetaEnsemble"]:
    if name not in aggs:
        continue
    a    = aggs[name]
    auc  = f"{a['ROC-AUC'][0]:.4f}±{a['ROC-AUC'][1]:.4f}"
    f1   = f"{a['F1'][0]:.4f}±{a['F1'][1]:.4f}"
    mcc  = f"{a['MCC'][0]:.4f}±{a['MCC'][1]:.4f}"
    tag  = " ★ proposed" if name == "ElliGAT" else (
           " ★ final"    if name == "MetaEnsemble" else "")
    print(f"{name:<18}{auc:>16}{f1:>16}{mcc:>16}{tag}")

# ─── 10. MC Uncertainty ──────────────────────────────────────────────────────
print(f"\n\nStep 6: MC Dropout uncertainty estimation ({MC_SAMPLES} samples) …")
unc_mean, unc_std = mc_uncertainty(last_ellgat_model, last_graph_data, n_samples=MC_SAMPLES)
unc_y = last_graph_data.y[last_graph_data.test_mask].cpu().numpy()
print(f"  Mean uncertainty (std): {unc_std.mean():.5f}")

# ─── 11. Save figure ─────────────────────────────────────────────────────────
print("\nStep 7: Generating result plots …")
last_results_final = {
    name: results_per_seed[name][-1]
    for name in results_per_seed
    if results_per_seed[name]
}
save_results_figure(last_results_final, aggs, unc_std, unc_y, RESULTS_PLOT_PATH)

# ─── 12. Save model + metrics ────────────────────────────────────────────────
torch.save(last_ellgat_model.state_dict(), BEST_MODEL_PATH)
print(f"  Saved model   → {BEST_MODEL_PATH}")

metrics_out = {
    name: {
        k: {"mean": float(v[0]), "std": float(v[1])}
        for k, v in aggs[name].items()
        if k not in ("probs", "true")
    }
    for name in aggs
}
with open(METRICS_JSON_PATH, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"  Saved metrics → {METRICS_JSON_PATH}")

print("\n✅  Pipeline complete.\n")
