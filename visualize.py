"""
src/visualize.py
================
All plotting and visualisation routines.

Plots generated
---------------
1.  ROC curves – all models on one axis
2.  PR curves  – better metric for imbalanced classes
3.  Ablation bar chart (AUC / F1 / MCC, 5 seeds)
4.  Calibration curves (reliability diagram)
5.  MC Dropout uncertainty histogram
6.  Confusion matrix (best model)
7.  AUC distribution box-plot (5 seeds)
8.  Feature importance (top-20 from XGBoost)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


PALETTE = {
    "ElliGAT":      "#e74c3c",
    "MetaEnsemble": "#8e44ad",
    "EvolveGCN":    "#e67e22",
    "BaselineGNN":  "#3498db",
    "LightGBM":     "#1abc9c",
    "XGBoost":      "#f39c12",
    "MLP":          "#2ecc71",
    "RandomForest": "#95a5a6",
}
ORDER = list(PALETTE.keys())


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_roc(results: dict, ax):
    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        fpr, tpr, _ = roc_curve(r["true"], r["probs"])
        ax.plot(
            fpr, tpr,
            color=PALETTE[name],
            label=f"{name} AUC={r['ROC-AUC']:.3f}",
            linewidth=1.8,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.legend(fontsize=7, loc="lower right")
    _ax_style(ax, "ROC Curves", "FPR", "TPR")


def plot_pr(results: dict, ax):
    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        prec, rec, _ = precision_recall_curve(r["true"], r["probs"])
        auprc = float(np.trapz(prec[::-1], rec[::-1]))
        ax.plot(
            rec, prec,
            color=PALETTE[name],
            label=f"{name} AUPRC={auprc:.3f}",
            linewidth=1.8,
        )
    ax.legend(fontsize=7, loc="upper right")
    _ax_style(ax, "Precision-Recall Curves", "Recall", "Precision")


def plot_ablation(aggs: dict, ax):
    metrics = ["ROC-AUC", "F1", "MCC"]
    x = np.arange(len(metrics))
    n_models = sum(1 for m in ORDER if m in aggs)
    w = 0.8 / max(n_models, 1)

    for i, name in enumerate([m for m in ORDER if m in aggs]):
        means = [aggs[name][k][0] for k in metrics]
        stds  = [aggs[name][k][1] for k in metrics]
        ax.bar(
            x + i * w, means, w,
            label=name,
            color=PALETTE.get(name, "#999"),
            yerr=stds, capsize=3, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=2)
    _ax_style(ax, "Ablation Study (5 seeds)")


def plot_calibration(results: dict, ax):
    for name in ["ElliGAT", "MetaEnsemble", "XGBoost"]:
        if name not in results:
            continue
        r = results[name]
        try:
            prob_true, prob_pred = calibration_curve(r["true"], r["probs"], n_bins=10)
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
            ax.plot(prob_pred, prob_true, "o-", color=PALETTE[name],
                    label=f"{name} ECE={ece:.3f}", linewidth=1.5, markersize=4)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.legend(fontsize=8)
    _ax_style(ax, "Calibration Curves", "Mean Predicted Prob", "Actual Fraud Rate")


def plot_uncertainty(unc_std: np.ndarray, y_true: np.ndarray, ax):
    unc_illicit = unc_std[y_true == 1]
    unc_licit   = unc_std[y_true == 0]
    ax.hist(unc_licit,   bins=40, alpha=0.6, color="#2ecc71", label="Licit")
    ax.hist(unc_illicit, bins=40, alpha=0.7, color="#e74c3c", label="Illicit (fraud)")
    ax.legend(fontsize=8)
    _ax_style(ax, "MC Dropout Uncertainty", "Prediction Std Dev", "# Transactions")


def plot_confusion(result: dict, ax):
    thresh = result.get("threshold", 0.5)
    ConfusionMatrixDisplay.from_predictions(
        result["true"],
        (result["probs"] > thresh).astype(int),
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"ElliGAT Confusion Matrix (τ={thresh:.2f})", fontsize=10, fontweight="bold")


def plot_boxplot(aggs: dict, ax):
    names  = [m for m in ORDER if m in aggs]
    data   = [aggs[m]["ROC-AUC"][2] for m in names]
    colors = [PALETTE.get(n, "#999") for n in names]
    bp     = ax.boxplot(data, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ROC-AUC")
    _ax_style(ax, "AUC Distribution (5 seeds)")


def save_results_figure(
    last_results: dict,
    aggs: dict,
    unc_std: np.ndarray,
    unc_y: np.ndarray,
    output_path: str = "outputs/results.png",
):
    """Compose 8-panel figure and save to disk."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "Bitcoin Fraud Detection — ElliGAT vs SOTA\n(Elliptic Dataset)",
        fontsize=14, fontweight="bold",
    )

    plot_roc(last_results, axes[0, 0])
    plot_pr(last_results, axes[0, 1])
    plot_ablation(aggs, axes[0, 2])
    plot_calibration(last_results, axes[0, 3])
    plot_uncertainty(unc_std, unc_y, axes[1, 0])
    plot_confusion(last_results.get("ElliGAT", next(iter(last_results.values()))), axes[1, 1])
    plot_boxplot(aggs, axes[1, 2])

    # Placeholder: summary table in last panel
    ax = axes[1, 3]
    ax.axis("off")
    rows = []
    for name in ORDER:
        if name not in aggs:
            continue
        a = aggs[name]
        rows.append([
            name,
            f"{a['ROC-AUC'][0]:.4f}±{a['ROC-AUC'][1]:.4f}",
            f"{a['F1'][0]:.4f}±{a['F1'][1]:.4f}",
            f"{a['MCC'][0]:.4f}±{a['MCC'][1]:.4f}",
        ])
    if rows:
        tbl = ax.table(
            cellText=rows,
            colLabels=["Model", "AUC", "F1", "MCC"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.5)
        ax.set_title("Summary Table (mean ± std)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure → {output_path}")
    plt.close(fig)
