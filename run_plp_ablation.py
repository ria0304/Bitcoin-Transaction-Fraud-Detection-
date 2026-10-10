#!/usr/bin/env python3
"""
PLP Ablation Experiments
========================
Tests the impact of Pseudo-Label Propagation components.

Ablations:
1. No PLP (baseline ElliGAT)
2. PLP with different confidence thresholds
3. PLP with different lambda_pl weights
4. PLP with different MC sample counts
5. PLP with different number of epochs
6. PLP with different probability bounds (prob_low, prob_high)
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    SEEDS, HIDDEN_DIM, NUM_GAT_LAYERS, NUM_HEADS, DROPOUT, EDGE_DIM,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE, LR_WARMUP_STEPS,
    PRETRAIN_EPOCHS, MASK_RATIO, CONTRASTIVE_WEIGHT, MC_SAMPLES,
    LOSS_NAME, ASL_GAMMA_POS, ASL_GAMMA_NEG, ASL_CLIP,
    LOG_FILE, LOG_LEVEL,
    DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH,
    OUTPUT_DIR, BEST_MODEL_PATH,
)
from src.data import find_dataset_path, load_elliptic, add_velocity_features, chronological_split, build_graph, VELOCITY_COLS
from src.models import ElliGAT
from src.trainer import pretrain, train_gnn, evaluate_gnn
from src.pseudo_label import run_plp, generate_pseudo_labels, pseudo_label_finetune
from src.logger import get_logger
from src.losses import CombinedASLLoss, CombinedLoss

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = get_logger("plp_ablation", log_file=os.path.join(OUTPUT_DIR, "plp_ablation.log"), level=LOG_LEVEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── Load dataset ────────────────────────────────────────────────────────────
print("Loading Elliptic dataset...")
dataset_path = find_dataset_path(DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH)
if dataset_path is None:
    raise FileNotFoundError("Elliptic dataset not found. Set BITCOIN_DATA_PATH in .env")

df, X_all, edge_index, edge_attr, txid_to_idx, all_nodes = load_elliptic(dataset_path)
df = add_velocity_features(df)

feat_cols = [c for c in df.columns if c.startswith("f")]
train_df, val_df, test_df, train_size, val_size = chronological_split(df)

tabular_cols = VELOCITY_COLS + feat_cols
X_train_raw = df.iloc[:train_size][tabular_cols].values
X_val_raw   = df.iloc[train_size:train_size + val_size][tabular_cols].values
X_test_raw  = df.iloc[train_size + val_size:][tabular_cols].values
y_train_np  = df.iloc[:train_size]["isFraud"].values.astype(np.float32)
y_val_np    = df.iloc[train_size:train_size + val_size]["isFraud"].values.astype(np.float32)
y_test_np   = df.iloc[train_size + val_size:]["isFraud"].values.astype(np.float32)

from sklearn.preprocessing import StandardScaler
scaler_tab  = StandardScaler().fit(X_train_raw)
X_train_s   = scaler_tab.transform(X_train_raw)
X_val_s     = scaler_tab.transform(X_val_raw)
X_test_s    = scaler_tab.transform(X_test_raw)

print("Building graph...")
graph_data, _, _ = build_graph(df, X_all, edge_index, edge_attr, txid_to_idx, train_size, val_size, feat_cols)
graph_data = graph_data.to(device)
in_dim = graph_data.num_node_features
print(f"  Nodes: {graph_data.num_nodes:,}  Edges: {graph_data.num_edges:,}  Features: {in_dim}")

n_pos = float(y_train_np.sum())
n_neg = float(len(y_train_np) - n_pos)
pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)


# ─── PLP Ablation Configurations ─────────────────────────────────────────────

PLP_ABLATION_CONFIGS = {
    "no_plp": {
        "name": "No PLP (Baseline ElliGAT)",
        "run_plp": False,
    },
    "plp_default": {
        "name": "PLP Default",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_strict_conf": {
        "name": "PLP Strict Confidence (0.05)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.05,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_loose_conf": {
        "name": "PLP Loose Confidence (0.20)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.20,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_high_lambda": {
        "name": "PLP High Lambda (0.50)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.50,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_low_lambda": {
        "name": "PLP Low Lambda (0.10)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.10,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_more_epochs": {
        "name": "PLP More Epochs (100)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 100,
        "lr": 1e-4,
    },
    "plp_fewer_mc": {
        "name": "PLP Fewer MC Samples (10)",
        "run_plp": True,
        "n_mc_samples": 10,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_more_mc": {
        "name": "PLP More MC Samples (50)",
        "run_plp": True,
        "n_mc_samples": 50,
        "confidence_threshold": 0.10,
        "prob_low": 0.20,
        "prob_high": 0.80,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_tight_bounds": {
        "name": "PLP Tight Prob Bounds (0.15/0.85)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.15,
        "prob_high": 0.85,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
    "plp_wide_bounds": {
        "name": "PLP Wide Prob Bounds (0.30/0.70)",
        "run_plp": True,
        "n_mc_samples": 30,
        "confidence_threshold": 0.10,
        "prob_low": 0.30,
        "prob_high": 0.70,
        "lambda_pl": 0.20,
        "epochs": 50,
        "lr": 1e-4,
    },
}


def train_base_model(seed=42):
    """Train base ElliGAT model (phases 1-2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ElliGAT(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GAT_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
    ).to(device)

    # Phase 1: Pre-training
    print("    Phase 1: Pre-training...")
    pretrain(model, graph_data, epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO, verbose=False)

    # Phase 2: Fine-tuning
    print("    Phase 2: Fine-tuning...")
    if LOSS_NAME == "asymmetric":
        criterion = CombinedASLLoss(
            gamma_pos=ASL_GAMMA_POS,
            gamma_neg=ASL_GAMMA_NEG,
            clip=ASL_CLIP,
            lambda_c=CONTRASTIVE_WEIGHT,
            lambda_p=0.10,
        )
    else:
        criterion = CombinedLoss(
            alpha=0.80, gamma=2.5,
            lambda_c=CONTRASTIVE_WEIGHT,
            lambda_p=0.10,
            pos_weight=pw,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    from src.trainer import _warmup_cosine
    scheduler = _warmup_cosine(optimizer, LR_WARMUP_STEPS, EPOCHS)

    best_score, no_improve, best_state = 0.0, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(graph_data)
        tr_logits = logits[graph_data.train_mask]
        tr_labels = graph_data.y[graph_data.train_mask]

        l_c = model.contrastive_loss(graph_data)
        l_p = model.pretrain_forward(graph_data, mask_ratio=0.10)

        loss = criterion(tr_logits, tr_labels, l_contrast=l_c, l_pretrain=l_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(graph_data)
            val_probs = torch.sigmoid(val_logits[graph_data.val_mask]).cpu().numpy()
            val_y = graph_data.y[graph_data.val_mask].cpu().numpy()

        if len(np.unique(val_y)) >= 2:
            from sklearn.metrics import roc_auc_score, f1_score
            val_auc = roc_auc_score(val_y, val_probs)
            val_f1 = f1_score(val_y, (val_probs > 0.5).astype(int), zero_division=0)
            score = 2 * val_auc * val_f1 / max(val_auc + val_f1, 1e-8)

            if score > best_score:
                best_score = score
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model


def run_plp_ablation(config_name, config, seed=42):
    """Run a single PLP ablation configuration."""
    print(f"\n{'='*60}")
    print(f"  PLP Ablation: {config['name']}")
    print(f"{'='*60}")

    # Train base model
    print("  Training base ElliGAT...")
    model = train_base_model(seed)

    # Evaluate before PLP
    print("  Evaluating before PLP...")
    result_before = evaluate_gnn(model, graph_data, n_mc=30)
    print(f"    Before PLP: AUC={result_before['ROC-AUC']:.4f}  F1={result_before['F1']:.4f}  MCC={result_before['MCC']:.4f}  AURC={result_before.get('AURC', 0):.4f}")

    if not config.get("run_plp", True):
        return {"before": result_before, "after": result_before, "pseudo_mask_count": 0}

    # Run PLP
    print("  Running PLP...")
    plp_params = {k: v for k, v in config.items() if k != "run_plp" and k != "name"}
    pseudo_mask, pseudo_y = run_plp(
        model, graph_data,
        n_mc_samples=plp_params["n_mc_samples"],
        confidence_threshold=plp_params["confidence_threshold"],
        lambda_pl=plp_params["lambda_pl"],
        epochs=plp_params["epochs"],
        lr=plp_params["lr"],
    )

    n_pseudo = int(pseudo_mask.sum().item())
    print(f"    Pseudo-labels assigned: {n_pseudo}")

    # Evaluate after PLP
    print("  Evaluating after PLP...")
    result_after = evaluate_gnn(model, graph_data, n_mc=30)
    print(f"    After PLP:  AUC={result_after['ROC-AUC']:.4f}  F1={result_after['F1']:.4f}  MCC={result_after['MCC']:.4f}  AURC={result_after.get('AURC', 0):.4f}")

    return {
        "before": result_before,
        "after": result_after,
        "pseudo_mask_count": n_pseudo,
        "plp_params": plp_params,
    }


# ─── Run All Ablations ───────────────────────────────────────────────────────

results = {}

for config_name, config in PLP_ABLATION_CONFIGS.items():
    try:
        result = run_plp_ablation(config_name, config, seed=42)
        results[config_name] = result
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[config_name] = {"error": str(e)}

# Save results
output_path = os.path.join(OUTPUT_DIR, "plp_ablation_results.json")
with open(output_path, "w") as f:
    serializable = {}
    for k, v in results.items():
        if "error" in v:
            serializable[k] = v
        else:
            serializable[k] = {
                "before": {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v["before"].items()},
                "after": {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v["after"].items()},
                "pseudo_mask_count": v.get("pseudo_mask_count", 0),
                "plp_params": v.get("plp_params", {}),
            }
    json.dump(serializable, f, indent=2)

print(f"\n\n{'='*60}")
print("PLP ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<35} {'AUC Before':>10} {'AUC After':>10} {'ΔAUC':>8} {'F1 Before':>10} {'F1 After':>10} {'ΔF1':>8} {'Pseudo':>8}")
print("-" * 105)
for config_name, config in PLP_ABLATION_CONFIGS.items():
    if config_name in results and "error" not in results[config_name]:
        r = results[config_name]
        before = r["before"]
        after = r["after"]
        delta_auc = after['ROC-AUC'] - before['ROC-AUC']
        delta_f1 = after['F1'] - before['F1']
        pseudo = r.get("pseudo_mask_count", 0)
        print(f"{config['name']:<35} {before['ROC-AUC']:>10.4f} {after['ROC-AUC']:>10.4f} {delta_auc:>+8.4f} {before['F1']:>10.4f} {after['F1']:>10.4f} {delta_f1:>+8.4f} {pseudo:>8}")
    else:
        print(f"{config['name']:<35} {'ERROR':>10}")

print(f"\nResults saved to: {output_path}")