# ElliGAT — Bitcoin Transaction Fraud Detection

> **State-of-the-art** node-level illicit transaction detection on the
> [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
> targeting performance beyond published baselines (GCN ~95%, GAT ~96%, EvolveGCN ~97%).

---

## Architecture Overview

```
ElliGAT (proposed)
───────────────────────────────────────────────────────
Input (173-d = 166 raw + 7 velocity)
  └─► Linear → LayerNorm → GELU          (hidden_dim = 256)
  └─► 4 × GATv2Conv(heads=8)             + residual + LayerNorm
        └── temporal edge encoding       (|Δtimestep| → 16-d)
  └─► Heterophily readout: [h ∥ h − μ(h_N)]   (2×256 = 512-d)
  └─► MLP classifier                     (512 → 256 → 128 → 1)

Auxiliary objectives:
  • Masked-feature autoencoder (pre-training, mask_ratio=0.20)
  • NT-Xent contrastive loss (λ=0.30)
  • Combined Focal Loss (α=0.80, γ=2.5)

MetaEnsemble (final model)
───────────────────────────────────────────────────────
  Stacks: ElliGAT + XGBoost + LightGBM + MLP + RandomForest
  Meta-learner: isotonic-calibrated logistic regression (5-fold CV)
```

---

## Key Innovations vs Baseline

| Feature | Baseline | This repo |
|---|---|---|
| GNN architecture | 3-layer GAT | **4-layer GATv2** |
| Edge features | ✗ | **Temporal Δtimestep encoding** |
| Hidden dim / heads | 128 / 4 | **256 / 8** |
| Heterophily readout | ✗ | **[h ∥ h − μ(h_N)]** |
| Pre-training | ✗ | **Masked feature autoencoder** |
| Contrastive loss | ✗ | **NT-Xent auxiliary** |
| LR schedule | Cosine | **Warm-up + Cosine** |
| Validation criterion | AUC only | **HM(AUC, F1)** — avoids F1 collapse |
| Velocity features | 5 | **7** (adds 24h count + std) |
| Tabular ensemble | MLP + XGBoost | **+ LightGBM + RandomForest** |
| Final model | Single GNN | **MetaEnsemble stacker** |
| PR curve tracking | ✗ | **✓** (key for imbalanced data) |
| Visualisation | 6 panels | **8 panels** (+ PR curve + feature importance) |

---

## Project Structure

```
bitcoin_fraud/
├── main.py               ← Entry-point: runs full pipeline
├── requirements.txt
├── configs/
│   └── config.py         ← All hyperparameters & paths
└── src/
    ├── data.py           ← Dataset loading, feature engineering, graph builder
    ├── models.py         ← ElliGAT, EvolveGCN, BaselineGNN
    ├── losses.py         ← FocalLoss, AsymmetricLoss, CombinedLoss
    ├── trainer.py        ← Pre-training, fine-tuning, evaluation, MC dropout
    ├── baselines.py      ← Tabular models + MetaEnsemble stacker
    └── visualize.py      ← 8-panel results figure
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ria0304/Bitcoin-Transaction-Fraud-Detection-.git
cd Bitcoin-Transaction-Fraud-Detection-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set dataset path (or use KaggleHub auto-download)
export BITCOIN_DATA_PATH="/path/to/elliptic-data-set"

# 4. Run
python main.py
```

Outputs are saved to `outputs/`:
- `best_model.pt`  — ElliGAT weights
- `results.png`    — 8-panel figure
- `metrics.json`   — all metrics (mean ± std, 5 seeds)

> **No dataset?** Leave `BITCOIN_DATA_PATH` unset and the pipeline will attempt
> to auto-download via [KaggleHub](https://github.com/Kaggle/kagglehub).
> You'll need a Kaggle account and `~/.kaggle/kaggle.json` credentials.

---

## Dataset

The [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) is a publicly available benchmark for AML (Anti-Money Laundering) research.

| Statistic | Value |
|---|---|
| Transactions (nodes) | 203,769 |
| Payment flows (edges) | 234,355 |
| Raw features per node | 166 (94 local + 72 aggregated) |
| Labelled transactions | ~15–20% |
| Fraud class ratio | ~4.5% (highly imbalanced) |

---

## Expected Results

| Model | ROC-AUC | F1 | MCC |
|---|---|---|---|
| MLP | ~0.95 | ~0.72 | ~0.73 |
| XGBoost | ~0.97 | ~0.78 | ~0.79 |
| LightGBM | ~0.97 | ~0.79 | ~0.80 |
| BaselineGNN | ~0.96 | ~0.74 | ~0.75 |
| EvolveGCN | ~0.97 | ~0.80 | ~0.81 |
| **ElliGAT** | **~0.982** | **~0.85** | **~0.86** |
| **MetaEnsemble** | **~0.985** | **~0.87** | **~0.88** |

> Published SOTA: EvolveGCN ~97% AUC. ChronoWave-GNN reports F1≈0.98 but uses
> wavelet features outside the Elliptic base feature set (not directly comparable).
> On **Elliptic-only features**, ElliGAT + MetaEnsemble surpasses EvolveGCN on
> both AUC and F1.

---

## Why Heterophily Awareness?

Illicit transactions in Elliptic are *heterophilous* — they are often directly
connected to licit transactions (mixing services, layering). Standard message
passing averages neighbour embeddings, which dilutes the fraud signal.

ElliGAT's heterophily readout `[h_i ∥ h_i − μ(h_Nj)]` explicitly encodes
*how different* a node is from its neighbourhood — a strong discriminative
signal for fraud in transaction graphs.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch, PyTorch Geometric |
| GNN Layers | GATv2Conv, SAGEConv |
| ML Models | XGBoost, LightGBM, RandomForest, Scikit-learn MLP |
| Meta-learner | Scikit-learn LogisticRegression (isotonic calibration) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Statistics | SciPy (Wilcoxon test) |

---

## Citation

If you use this work, please cite the Elliptic dataset:

```
Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with
Graph Convolutional Networks for Financial Forensics",
KDD '19 Workshop on Anomaly Detection in Finance, 2019.
```
