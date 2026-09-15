"""
src/baselines.py
================
Tabular baseline models and meta-ensemble stacking.

Models
------
TabularBaselines – trains MLP, XGBoost, LightGBM, and RandomForest
ROLAND           – Relational temporal GNN baseline (2022)
TGN              – Temporal Graph Network baseline (Rossi et al. 2020)
MetaEnsemble     – stacks GNN + tabular model predictions with a logistic meta-learner

The MetaEnsemble is the final submission model: it combines the graph-aware
ElliGAT predictions with tabular signals into a robust, calibrated classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, TransformerConv
from torch_geometric.utils import degree

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [warn] XGBoost not found – pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [warn] LightGBM not found – pip install lightgbm")

from src.trainer import compute_metrics


# ─── ROLAND (Wang et al. 2022) ───────────────────────────────────────────────

class ROLAND(nn.Module):
    """
    ROLAND: Graph Learning with Evolving Relations (WWW 2022).

    Key idea: node embeddings are updated across timesteps using a GRU
    that takes the new GNN embedding as input and the previous embedding
    as hidden state. This gives each node a persistent memory across time.

    Architecture:
        • 2-layer GraphSAGE encoder (shared across timesteps)
        • GRU cell: h_t = GRU(GNN(G_t, h_{t-1}), h_{t-1})
        • MLP classifier on final h_t

    forward(data) processes the full graph in one pass (static mode),
    matching the ElliGAT/EvolveGCN interface used by train_gnn().
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        # GNN encoder
        self.proj  = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # GRU cell for temporal node-state evolution
        self.gru = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
        )
        self.norm_gru = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Persistent node hidden state (reset between forward calls)
        self._h: torch.Tensor | None = None

    def reset_state(self):
        self._h = None

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj(x))
        h = self.norm1(F.gelu(self.conv1(h, edge_index)) + h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm2(F.gelu(self.conv2(h, edge_index)) + h)
        return h

    def forward(self, data) -> torch.Tensor:
        """
        Full-graph forward pass.
        If called sequentially per timestep, node states accumulate.
        train_gnn() calls this once per graph — which covers all timesteps
        simultaneously (static approximation of ROLAND).
        """
        N = data.x.size(0)

        # Initialise hidden state on first call or size mismatch
        if self._h is None or self._h.size(0) != N:
            self._h = torch.zeros(N, self.hidden_dim, device=data.x.device)

        gnn_out    = self._encode(data.x, data.edge_index)
        h_new      = self.gru(gnn_out, self._h)          # (N, hidden)
        h_new      = self.norm_gru(h_new)
        self._h    = h_new.detach()                       # persist for next timestep

        return self.classifier(h_new).squeeze(-1)


# ─── TGN — Temporal Graph Network (Rossi et al. NeurIPS 2020) ────────────────

class TGNMemory(nn.Module):
    """
    Lightweight TGN memory module.

    Full TGN requires streaming edge events; we adapt it to the Elliptic
    static-snapshot setting by treating each node's neighbourhood as its
    "interaction history" and using a GRU to update a persistent memory.

    This is a faithful architectural approximation of TGN for batch
    node-classification on a static graph snapshot, matching the interface
    expected by train_gnn() and evaluate_gnn().

    Reference: Rossi et al. "Temporal Graph Networks for Deep Learning on
    Dynamic Graphs", NeurIPS 2020 Workshop.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout
        self.mem_dim    = hidden_dim

        # Message function: aggregate neighbour features → message
        self.msg_proj = nn.Linear(in_dim, hidden_dim)

        # Memory updater: GRU cell
        self.mem_gru  = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=self.mem_dim,
        )

        # Embedding module: TransformerConv attends over memory + raw features
        self.emb_conv = TransformerConv(
            in_channels=in_dim + self.mem_dim,
            out_channels=hidden_dim // 4,
            heads=4,
            dropout=dropout,
            concat=True,
        )
        self.norm_emb = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._memory: torch.Tensor | None = None

    def reset_state(self):
        self._memory = None

    def _compute_messages(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbour raw features as messages for each node."""
        src, dst    = edge_index
        msg         = F.gelu(self.msg_proj(x[src]))       # (E, hidden)
        agg         = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        deg         = degree(dst, num_nodes=x.size(0), dtype=torch.float).clamp(min=1)
        return agg / deg.unsqueeze(1)                      # mean-aggregated msg

    def forward(self, data) -> torch.Tensor:
        N  = data.x.size(0)
        x  = data.x
        ei = data.edge_index

        # Initialise memory
        if self._memory is None or self._memory.size(0) != N:
            self._memory = torch.zeros(N, self.mem_dim, device=x.device)

        # 1. Compute messages from neighbourhood
        msgs = self._compute_messages(x, ei)               # (N, hidden)

        # 2. Update memory with GRU
        mem_new      = self.mem_gru(msgs, self._memory)    # (N, mem_dim)
        self._memory = mem_new.detach()

        # 3. Embedding: concat raw features + memory → TransformerConv
        x_aug = torch.cat([x, mem_new], dim=-1)            # (N, in+mem)
        h     = F.gelu(self.emb_conv(x_aug, ei))
        h     = self.norm_emb(h)
        h     = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)


# ─── Tabular Baselines ───────────────────────────────────────────────────────

class TabularBaselines:
    """Train and evaluate MLP, XGBoost, LightGBM, and RF on tabular features."""

    def __init__(self, random_state: int = 42):
        self.rs = random_state
        self.models: dict = {}

    def fit(
        self,
        X_train, y_train,
        X_val,   y_val,
        n_pos: float,
        n_neg: float,
    ):
        scale_pw = n_neg / max(n_pos, 1)

        # ── MLP ─────────────────────────────────────────────────────────────
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            max_iter=500,
            random_state=self.rs,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=1e-4,
        )
        mlp.fit(X_train, y_train)
        self.models["MLP"] = mlp

        # ── XGBoost ─────────────────────────────────────────────────────────
        if HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=600,
                max_depth=7,
                learning_rate=0.03,
                scale_pos_weight=scale_pw,
                subsample=0.80,
                colsample_bytree=0.70,
                min_child_weight=5,
                reg_lambda=1.0,
                reg_alpha=0.1,
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=self.rs,
                verbosity=0,
                early_stopping_rounds=30,
            )
            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models["XGBoost"] = xgb

        # ── LightGBM ────────────────────────────────────────────────────────
        if HAS_LGB:
            lgbm = lgb.LGBMClassifier(
                n_estimators=600,
                max_depth=7,
                learning_rate=0.03,
                scale_pos_weight=scale_pw,
                subsample=0.80,
                colsample_bytree=0.70,
                reg_lambda=1.0,
                reg_alpha=0.1,
                random_state=self.rs,
                verbosity=-1,
            )
            lgbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            self.models["LightGBM"] = lgbm

        # ── Random Forest ────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            class_weight={0: 1, 1: int(scale_pw)},
            random_state=self.rs,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return {model_name: fraud_proba} for all fitted models."""
        return {
            name: m.predict_proba(X)[:, 1]
            for name, m in self.models.items()
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict]:
        results = {}
        for name, probs in self.predict_proba(X_test).items():
            results[name] = compute_metrics(y_test, probs)
            print(
                f"  {name:<15} AUC={results[name]['ROC-AUC']:.4f} "
                f"F1={results[name]['F1']:.4f} "
                f"MCC={results[name]['MCC']:.4f}"
            )
        return results


# ─── Meta-Ensemble ───────────────────────────────────────────────────────────

class MetaEnsemble:
    """
    Stacked generalisation: GNN + tabular models → logistic meta-learner.

    Stack features: [gnn_prob, mlp_prob, xgb_prob, lgbm_prob, rf_prob]
    Meta-learner: Platt-scaled logistic regression (isotonic calibration).
    """

    def __init__(self):
        self.meta = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=5,
            method="isotonic",
        )
        self._fitted = False

    def _stack(
        self,
        gnn_probs: np.ndarray,
        tab_probs: dict[str, np.ndarray],
    ) -> np.ndarray:
        cols = [gnn_probs] + list(tab_probs.values())
        return np.column_stack(cols)

    def fit(
        self,
        gnn_val_probs: np.ndarray,
        tab_val_probs: dict[str, np.ndarray],
        y_val: np.ndarray,
    ):
        X_meta = self._stack(gnn_val_probs, tab_val_probs)
        self.meta.fit(X_meta, y_val)
        self._fitted = True

    def predict_proba(
        self,
        gnn_test_probs: np.ndarray,
        tab_test_probs: dict[str, np.ndarray],
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MetaEnsemble.fit() must be called before predict_proba()")
        X_meta = self._stack(gnn_test_probs, tab_test_probs)
        return self.meta.predict_proba(X_meta)[:, 1]

    def evaluate(
        self,
        gnn_test_probs: np.ndarray,
        tab_test_probs: dict[str, np.ndarray],
        y_test: np.ndarray,
    ) -> dict:
        probs  = self.predict_proba(gnn_test_probs, tab_test_probs)
        result = compute_metrics(y_test, probs)
        print(
            f"  MetaEnsemble     AUC={result['ROC-AUC']:.4f} "
            f"F1={result['F1']:.4f} "
            f"MCC={result['MCC']:.4f}"
        )
        return result
