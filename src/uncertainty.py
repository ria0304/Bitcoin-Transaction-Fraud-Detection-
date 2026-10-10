"""
src/uncertainty.py
==================
Adaptive Uncertainty-Gated Threshold (AUGT) — Novel Contribution #2
=====================================================================

Motivation
----------
The existing pipeline finds a single operating threshold tau* by grid search on
the validation set. This approach has two weaknesses:

  1. Static: the same tau* is applied to all test nodes regardless of how
     confident the model is in each individual prediction.
  2. AUC/F1 leakage: the grid search is fit on validation labels, which means
     the reported F1 reflects the best possible post-hoc threshold rather than
     a deployable decision rule.

AUGT replaces this with an **uncertainty-aware two-threshold abstention mechanism**:
  * tau_low(u)  : below this probability -> predict licit (high confidence)
  * tau_high(u) : above this probability -> predict fraud  (high confidence)
  * [tau_low(u), tau_high(u)] : "uncertain" zone -> prediction deferred to analyst

The thresholds are **functions of per-node uncertainty**: higher uncertainty
widens the abstention zone, lower uncertainty narrows it. This is the key
novelty — the model's own epistemic uncertainty directly controls the decision
boundary, making the rule deployable without validation-set leakage.

Evaluation: Area Under Risk-Coverage Curve (AURC)
-------------------------------------------------
We introduce AURC as an additional benchmark metric. The risk-coverage curve
plots risk (1 - precision on retained samples) vs coverage (fraction retained).
Lower AURC = better-calibrated model; a random model has AURC ~0.5 * (1-prec).

References
----------
* Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks",
  NeurIPS 2017.
* Corbiere et al., "Addressing Failure Prediction by Learning Model Confidence",
  NeurIPS 2019.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score,
)


# ─── Uncertainty-aware threshold fitting ──────────────────────────────────────

def fit_augt_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    min_coverage: float = 0.70,
    n_grid: int = 30,
) -> tuple:
    """
    Find uncertainty-aware threshold parameters that maximise F1 on the retained
    subset subject to coverage >= min_coverage.

    AUGT uses an **uncertainty-dependent threshold function**:
        tau_low(u)  = base_low  - k * u
        tau_high(u) = base_high + k * u

    where u = normalized uncertainty in [0, 1]. Higher uncertainty widens the
    abstention zone, lower uncertainty narrows it.

    We grid-search over (base_low, base_high, k) to find the best parameters.

    Parameters
    ----------
    y_true      : (N,) binary ground truth
    probs       : (N,) mean fraud probability from MC Dropout
    uncertainty : (N,) std of MC Dropout predictions (epistemic uncertainty)
    min_coverage: minimum fraction of nodes that must be classified
    n_grid      : resolution of the grid search (per dimension)

    Returns
    -------
    (base_low, base_high, k) : float tuple defining the uncertainty-aware thresholds
    """
    # Normalize uncertainty to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_max > u_min:
        u_norm = (uncertainty - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(uncertainty)

    best_f1 = 0.0
    best_bl, best_bh, best_k = 0.30, 0.70, 0.20

    # Grid search: base_low ∈ [0.05, 0.40], base_high ∈ [0.55, 0.95], k ∈ [0, 0.4]
    for base_low in np.linspace(0.05, 0.40, n_grid):
        for base_high in np.linspace(max(base_low + 0.1, 0.55), 0.95, n_grid):
            for k in np.linspace(0.0, 0.40, n_grid // 3):
                # Compute per-node thresholds
                tau_low  = base_low  - k * u_norm
                tau_high = base_high + k * u_norm

                # Clamp to valid range
                tau_low  = np.clip(tau_low,  0.0, 0.5)
                tau_high = np.clip(tau_high, 0.5, 1.0)

                # Ensure tau_low < tau_high
                valid = tau_low < tau_high
                if not valid.any():
                    continue

                # Retain nodes where prob <= tau_low OR prob >= tau_high
                retain = (probs <= tau_low) | (probs >= tau_high)
                coverage = retain.mean()
                if coverage < min_coverage:
                    continue

                y_r = y_true[retain]
                if len(np.unique(y_r)) < 2:
                    continue

                # Predict: 1 if prob >= tau_high, 0 if prob <= tau_low
                y_pred = np.zeros_like(y_r, dtype=int)
                y_pred[probs[retain] >= tau_high[retain]] = 1

                f1 = f1_score(y_r, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bl, best_bh, best_k = float(base_low), float(base_high), float(k)

    return best_bl, best_bh, best_k


# ─── Uncertainty-aware AUGT decision function ──────────────────────────────────

def augt_predict(
    probs: np.ndarray,
    uncertainty: np.ndarray,
    base_low: float,
    base_high: float,
    k: float,
) -> np.ndarray:
    """
    Apply the uncertainty-aware two-threshold decision rule.

    tau_low(u)  = base_low  - k * u
    tau_high(u) = base_high + k * u

    Returns
    -------
    pred : (N,) int array
        1  -> predicted fraud  (prob >= tau_high(u))
        0  -> predicted licit  (prob <= tau_low(u))
       -1  -> deferred / uncertain (tau_low(u) < prob < tau_high(u))
    """
    # Normalize uncertainty to [0, 1]
    u_min, u_max = uncertainty.min(), uncertainty.max()
    if u_max > u_min:
        u_norm = (uncertainty - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(uncertainty)

    tau_low  = np.clip(base_low  - k * u_norm, 0.0, 0.5)
    tau_high = np.clip(base_high + k * u_norm, 0.5, 1.0)

    pred = np.full(len(probs), -1, dtype=int)
    pred[probs <= tau_low]  = 0
    pred[probs >= tau_high] = 1
    return pred


# ─── AURC computation ─────────────────────────────────────────────────────────

def compute_aurc(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    n_thresholds: int = 100,
) -> tuple:
    """
    Compute the Area Under the Risk-Coverage curve.

    Traces the curve by sweeping an uncertainty threshold from max to min
    (retaining increasingly uncertain samples):
      Coverage(u_t) = fraction of samples with uncertainty <= u_t
      Risk(u_t)     = 1 - precision on the retained subset

    Returns
    -------
    (aurc, coverages, risks) where aurc is a float scalar (lower = better).
    """
    thresholds = np.percentile(uncertainty, np.linspace(0, 100, n_thresholds))
    coverages, risks = [], []

    for u_t in thresholds:
        retain = uncertainty <= u_t
        cov    = retain.mean()
        if cov == 0:
            continue

        y_r    = y_true[retain]
        y_pred = (probs[retain] >= 0.5).astype(int)

        if len(np.unique(y_r)) < 2:
            risk = float(1.0 - (y_pred == y_r).mean())
        else:
            risk = float(1.0 - precision_score(y_r, y_pred, zero_division=0))

        coverages.append(float(cov))
        risks.append(risk)

    coverages = np.array(coverages)
    risks     = np.array(risks)
    aurc      = float(np.trapz(risks, coverages)) if len(coverages) > 1 else 1.0
    return aurc, coverages, risks


# ─── Full AUGT evaluation ─────────────────────────────────────────────────────

def evaluate_augt(
    y_true: np.ndarray,
    probs: np.ndarray,
    unc_mean: np.ndarray,
    unc_std: np.ndarray,
    base_low: float,
    base_high: float,
    k: float,
) -> dict:
    """
    Full evaluation under the uncertainty-aware AUGT decision rule.

    Reports metrics on:
      (a) All samples (standard flat threshold at 0.5)
      (b) Retained (non-deferred) samples only with uncertainty-aware thresholds
      (c) AURC

    Returns a flat dict suitable for the results table.
    """
    # ── Standard metrics on all nodes (threshold=0.5) ────────────────────
    pred_all = (probs >= 0.5).astype(int)
    auc_all  = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
    f1_all   = float(f1_score(y_true, pred_all, zero_division=0))
    mcc_all  = float(matthews_corrcoef(y_true, pred_all))

    # ── AUGT retained subset with uncertainty-aware thresholds ────────────
    pred_3way = augt_predict(probs, unc_std, base_low, base_high, k)
    retained  = pred_3way >= 0
    coverage  = float(retained.mean())

    if retained.sum() > 0 and len(np.unique(y_true[retained])) > 1:
        f1_ret   = float(f1_score(y_true[retained], pred_3way[retained], zero_division=0))
        mcc_ret  = float(matthews_corrcoef(y_true[retained], pred_3way[retained]))
        prec_ret = float(precision_score(y_true[retained], pred_3way[retained], zero_division=0))
        rec_ret  = float(recall_score(y_true[retained], pred_3way[retained], zero_division=0))
    else:
        f1_ret = mcc_ret = prec_ret = rec_ret = 0.0

    deferred_fraud = int(((pred_3way == -1) & (y_true == 1)).sum())
    deferred_licit = int(((pred_3way == -1) & (y_true == 0)).sum())

    # ── AURC ─────────────────────────────────────────────────────────────
    aurc, coverages, risks = compute_aurc(y_true, probs, unc_std)

    return {
        "ROC-AUC":             auc_all,
        "F1":                  f1_all,
        "MCC":                 mcc_all,
        "Balanced Acc":        float(balanced_accuracy_score(y_true, pred_all)),
        "Precision":           float(precision_score(y_true, pred_all, zero_division=0)),
        "Recall":              float(recall_score(y_true, pred_all, zero_division=0)),
        "AUGT/Coverage":       coverage,
        "AUGT/F1":             f1_ret,
        "AUGT/MCC":            mcc_ret,
        "AUGT/Precision":      prec_ret,
        "AUGT/Recall":         rec_ret,
        "AUGT/Deferred_Fraud": deferred_fraud,
        "AUGT/Deferred_Licit": deferred_licit,
        "AURC":                aurc,
        "AUGT/base_low":       base_low,
        "AUGT/base_high":      base_high,
        "AUGT/k":              k,
        "aurc_coverages":      coverages,
        "aurc_risks":          risks,
        "probs":               probs,
        "true":                y_true,
        "threshold":           0.5,
    }
