#!/usr/bin/env python3
"""
FAGA Ablation Experiments
=========================
Tests the impact of each FAGA component on model performance.

Ablations:
1. No FAGA (random augmentations)
2. Structure-aware edge drop only
3. Feature-aware mask only
4. Full FAGA (both)
5. Different drop/mask probability settings
"""

import os
import sys
import json
import numpy as np
import torch
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    SEEDS, HIDDEN_DIM, NUM_GAT_LAYERS, NUM_HEADS, DROPOUT, EDGE_DIM,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE, LR_WARMUP_STEPS,
    PRETRAIN_EPOCHS, MASK_RATIO, CONTRASTIVE_WEIGHT,
    LOSS_NAME, ASL_GAMMA_POS, ASL_GAMMA_NEG, ASL_CLIP,
    LOG_FILE, LOG_LEVEL,
    DATASET, PROJECT_DATA_PATH, ELLIPTIC_CACHE_PATH,
    OUTPUT_DIR, BEST_MODEL_PATH,
)
from src.data import find_dataset_path, load_elliptic, add_velocity_features, chronological_split, build_graph, VELOCITY_COLS
from src.models import ElliGAT
from src.trainer import pretrain, train_gnn, evaluate_gnn
from src.augmentation import augment_pair, structure_aware_edge_drop, feature_aware_mask
from src.logger import get_logger

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = get_logger("faga_ablation", log_file=os.path.join(OUTPUT_DIR, "faga_ablation.log"), level=LOG_LEVEL)

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


# ─── FAGA Ablation Configurations ────────────────────────────────────────────

ABLATION_CONFIGS = {
    "no_faga": {
        "name": "No FAGA (Random Augmentations)",
        "use_faga": False,
        "use_structure_aware_edge_drop": False,
        "use_feature_aware_mask": False,
        "lambda_c": 0.30,
    },
    "structure_only": {
        "name": "Structure-Aware Edge Drop Only",
        "use_faga": True,
        "use_structure_aware_edge_drop": True,
        "use_feature_aware_mask": False,
        "lambda_c": 0.30,
    },
    "feature_only": {
        "name": "Feature-Aware Mask Only",
        "use_faga": True,
        "use_structure_aware_edge_drop": False,
        "use_feature_aware_mask": True,
        "lambda_c": 0.30,
    },
    "full_faga": {
        "name": "Full FAGA (Both)",
        "use_faga": True,
        "use_structure_aware_edge_drop": True,
        "use_feature_aware_mask": True,
        "lambda_c": 0.30,
    },
    "full_faga_stronger": {
        "name": "Full FAGA (Stronger Aug)",
        "use_faga": True,
        "use_structure_aware_edge_drop": True,
        "use_feature_aware_mask": True,
        "lambda_c": 0.30,
        "drop_prob_high": 0.40,
        "drop_prob_low": 0.02,
        "feat_mask_high": 0.40,
        "feat_mask_low": 0.02,
    },
    "full_faga_weaker": {
        "name": "Full FAGA (Weaker Aug)",
        "use_faga": True,
        "use_structure_aware_edge_drop": True,
        "use_feature_aware_mask": True,
        "lambda_c": 0.30,
        "drop_prob_high": 0.20,
        "drop_prob_low": 0.05,
        "feat_mask_high": 0.20,
        "feat_mask_low": 0.10,
    },
}


# ─── Custom ElliGAT with configurable FAGA ──────────────────────────────────

class AblationElliGAT(ElliGAT):
    """ElliGAT with configurable FAGA ablation settings."""

    def __init__(self, *args, ablation_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_config = ablation_config or {}

    def contrastive_loss(self, data, temperature: float = 0.07) -> torch.Tensor:
        """Override contrastive loss with configurable FAGA."""
        from src.augmentation import augment_pair, structure_aware_edge_drop, feature_aware_mask
        import torch.nn.functional as F

        self.train()
        mask = data.train_mask

        config = self.ablation_config

        if config.get("use_faga", True):
            # Use FAGA augmentations
            if config.get("use_structure_aware_edge_drop", True) and config.get("use_feature_aware_mask", True):
                # Full FAGA
                view1, view2 = augment_pair(
                    data,
                    drop_prob_high=config.get("drop_prob_high", 0.30),
                    drop_prob_low=config.get("drop_prob_low", 0.03),
                    feat_mask_high=config.get("feat_mask_high", 0.30),
                    feat_mask_low=config.get("feat_mask_low", 0.05),
                )
            elif config.get("use_structure_aware_edge_drop", True):
                # Structure only - second view is just feature mask with default
                view1 = structure_aware_edge_drop(
                    data,
                    drop_prob_high=config.get("drop_prob_high", 0.30),
                    drop_prob_low=config.get("drop_prob_low", 0.03),
                )
                view2 = feature_aware_mask(
                    data,
                    feat_mask_high=0.30,
                    feat_mask_low=0.05,
                )
            elif config.get("use_feature_aware_mask", True):
                # Feature only - first view is just edge drop with default
                view1 = structure_aware_edge_drop(data, drop_prob_high=0.30, drop_prob_low=0.03)
                view2 = feature_aware_mask(
                    data,
                    feat_mask_high=config.get("feat_mask_high", 0.30),
                    feat_mask_low=config.get("feat_mask_low", 0.05),
                )
        else:
            # No FAGA - use random augmentations (original broken approach)
            # View 1: random edge drop
            ei = data.edge_index
            rand = torch.rand(ei.size(1), device=ei.device)
            keep_mask = rand >= 0.2  # 20% random drop
            view1 = data.clone()
            view1.edge_index = ei[:, keep_mask]
            view1.edge_attr = data.edge_attr[keep_mask] if data.edge_attr is not None else None

            # View 2: random feature mask
            x = data.x.clone()
            N, D = x.shape
            mask_rand = torch.rand(N, D, device=x.device) < 0.2
            x[mask_rand] = 0.0
            view2 = data.clone()
            view2.x = x

        z1 = F.normalize(
            self.contrast_head(
                self.encode(view1.x, view1.edge_index, view1.edge_attr)[mask]
            ), dim=-1
        )
        z2 = F.normalize(
            self.contrast_head(
                self.encode(view2.x, view2.edge_index, view2.edge_attr)[mask]
            ), dim=-1
        )

        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / temperature

        labels = torch.arange(B, device=z.device)
        loss = (
            F.cross_entropy(sim[:B, B:], labels) +
            F.cross_entropy(sim[B:, :B], labels)
        ) / 2
        return loss


def run_ablation(config_name, config, seed=42):
    """Run a single ablation configuration."""
    print(f"\n{'='*60}")
    print(f"  Ablation: {config['name']}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AblationElliGAT(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GAT_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
        ablation_config=config,
    ).to(device)

    # Pre-training
    print("  Phase 1: Pre-training...")
    pretrain(model, graph_data, epochs=PRETRAIN_EPOCHS, mask_ratio=MASK_RATIO, verbose=False)

    # Fine-tuning
    print("  Phase 2: Fine-tuning...")
    if LOSS_NAME == "asymmetric":
        from src.losses import CombinedASLLoss
        criterion = CombinedASLLoss(
            gamma_pos=ASL_GAMMA_POS,
            gamma_neg=ASL_GAMMA_NEG,
            clip=ASL_CLIP,
            lambda_c=config.get("lambda_c", CONTRASTIVE_WEIGHT),
            lambda_p=0.10,
        )
    else:
        from src.losses import CombinedLoss
        criterion = CombinedLoss(
            alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
            lambda_c=config.get("lambda_c", CONTRASTIVE_WEIGHT),
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
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0:
            print(f"    ep {epoch:03d} | loss={loss.item():.4f} | AUC={val_auc:.4f} | F1={val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluation
    print("  Evaluating...")
    result = evaluate_gnn(model, graph_data, n_mc=30)

    return result


# ─── Run All Ablations ───────────────────────────────────────────────────────

results = {}

for config_name, config in ABLATION_CONFIGS.items():
    try:
        result = run_ablation(config_name, config, seed=42)
        results[config_name] = result
        print(f"\n  Result: AUC={result['ROC-AUC']:.4f}  F1={result['F1']:.4f}  MCC={result['MCC']:.4f}  AURC={result.get('AURC', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[config_name] = {"error": str(e)}

# Save results
output_path = os.path.join(OUTPUT_DIR, "faga_ablation_results.json")
with open(output_path, "w") as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for k, v in results.items():
        if "error" in v:
            serializable[k] = v
        else:
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
    json.dump(serializable, f, indent=2)

print(f"\n\n{'='*60}")
print("FAGA ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Config':<30} {'AUC':>8} {'F1':>8} {'MCC':>8} {'AURC':>8}")
print("-" * 65)
for config_name, config in ABLATION_CONFIGS.items():
    if config_name in results and "error" not in results[config_name]:
        r = results[config_name]
        print(f"{config['name']:<30} {r['ROC-AUC']:>8.4f} {r['F1']:>8.4f} {r['MCC']:>8.4f} {r.get('AURC', 0):>8.4f}")
    else:
        print(f"{config['name']:<30} {'ERROR':>8}")

print(f"\nResults saved to: {output_path}")