"""
src/baselines.py
================
Tabular baseline models and meta-ensemble stacking.

Models
------
TabularBaselines – trains MLP, XGBoost, LightGBM, and RandomForest
MetaEnsemble     – stacks GNN + tabular model predictions with a logistic meta-learner

The MetaEnsemble is the final submission model: it combines the graph-aware
ElliGAT predictions with tabular signals into a robust, calibrated classifier.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

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
                eval_metric="aucpr",      # PR-AUC better than ROC-AUC here
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
            params = dict(
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
            lgbm = lgb.LGBMClassifier(**params)
            lgbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)],
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

    This is the final model submitted for SOTA comparison.
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
        probs = self.predict_proba(gnn_test_probs, tab_test_probs)
        result = compute_metrics(y_test, probs)
        print(
            f"  MetaEnsemble     AUC={result['ROC-AUC']:.4f} "
            f"F1={result['F1']:.4f} "
            f"MCC={result['MCC']:.4f}"
        )
        return result
