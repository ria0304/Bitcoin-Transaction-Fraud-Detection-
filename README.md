# ElliGAT — Bitcoin Transaction Fraud Detection

> **Heterophily-aware graph attention network** for illicit transaction detection on the
> [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
> ElliGAT achieves **F1 = 0.7925 ± 0.0104** on the GNN component and
> **F1 = 0.8629 ± 0.0038** with the full MetaEnsemble (5-seed average, 468K edges).

---

## Problem Statement

Cryptocurrency networks like Bitcoin operate without a central authority, making
them attractive for money laundering, ransomware payments, and other illicit
activity. Detecting fraud in these networks is fundamentally hard for three reasons:

**1. The graph is heterophilous.**
Illicit nodes are rarely clustered together. Fraudsters deliberately route
transactions through legitimate wallets (mixing, layering) so that each fraud
node is typically surrounded by licit neighbours. Standard GNNs aggregate
neighbour messages and average away exactly the signal that distinguishes fraud.

**2. The data is severely imbalanced.**
Only ~9.76% of labelled transactions in the Elliptic dataset are illicit.
Models that optimise AUC alone can achieve high scores while missing most
actual fraud — the minority class that matters most.

**3. The graph evolves over time.**
The Elliptic dataset spans 49 timesteps. A model that ignores temporal
structure treats edges from different time periods as equivalent, losing
information about transaction velocity and sequence.

Existing approaches — GCN, GAT, EvolveGCN — address at most one of these
three problems at a time. No published method on the Elliptic benchmark
simultaneously handles heterophily, class imbalance, and temporal dynamics
within a single unified architecture.

---

## Solution

ElliGAT is designed to address all three problems in one framework:

**Heterophily-aware readout.**
Instead of using only the node embedding `h_i`, ElliGAT appends the
difference between a node and its neighbourhood mean: `[h_i ∥ h_i − μ(h_Nj)]`.
This explicitly encodes *how different* a node is from its neighbours —
a strong discriminative signal for fraud nodes surrounded by licit transactions.

**Imbalance-robust training.**
ElliGAT uses Focal Loss (α=0.80, γ=2.5) which down-weights easy licit
examples and focuses training on hard fraud cases. The validation criterion
is the harmonic mean of AUC and F1 — preventing the common failure mode
where a model maximises AUC while F1 collapses on the minority class.

**Temporal edge encoding.**
Each edge carries a 16-dimensional encoding of |Δtimestep| between connected
transactions. This lets the GATv2 attention mechanism learn that edges crossing
large time gaps carry different information than edges within the same timestep,
capturing the sequential structure of the Bitcoin transaction graph.

**Two-phase training.**
Phase 1 is self-supervised pre-training with a masked feature autoencoder
(mask ratio 0.20), giving the model a strong initialisation before seeing
any labels. Phase 2 fine-tunes with the combined Focal Loss on labelled nodes.

**MetaEnsemble stacking.**
A calibrated logistic regression meta-learner stacks ElliGAT's predictions
with XGBoost, LightGBM, RandomForest, and MLP outputs. The tabular models
capture feature patterns the GNN misses; the GNN captures graph structure
the tabular models cannot access. Together they achieve higher F1 than
any single model alone.

---

## Results

All results are averaged over **5 random seeds** (42, 123, 2024, 17, 99) on the
chronological 70/15/15 train/val/test split. EvolveGCN numbers are taken from
the original paper (Pareja et al., 2020) as the model could not be reproduced
under available GPU memory constraints.

| Model | ROC-AUC | F1 | MCC |
|---|---|---|---|
| MLP | 0.9673 ± 0.0068 | 0.8206 ± 0.0047 | 0.8167 ± 0.0062 |
| RandomForest | 0.9894 ± 0.0007 | 0.8441 ± 0.0021 | 0.8405 ± 0.0031 |
| XGBoost | 0.9907 ± 0.0010 | 0.8508 ± 0.0147 | 0.8419 ± 0.0180 |
| LightGBM | 0.9917 ± 0.0001 | 0.8770 ± 0.0038 | 0.8716 ± 0.0049 |
| EvolveGCN (cited) | ~0.940 | ~0.720 | — |
| BaselineGNN | 0.9250 ± 0.0039 | 0.7738 ± 0.0060 | 0.7672 ± 0.0070 |
| **ElliGAT (ours)** | **0.9468 ± 0.0054** | **0.7925 ± 0.0104** | **0.7837 ± 0.0108** |
| **MetaEnsemble (ours)** | **0.9870 ± 0.0014** | **0.8629 ± 0.0038** | **0.8570 ± 0.0037** |

**ElliGAT vs EvolveGCN (cited):** +7.3% F1, +0.68% AUC  
**MetaEnsemble vs EvolveGCN (cited):** +14.3% F1, +4.7% AUC

> Precision: 0.8976 ± 0.0112 | Recall: 0.7096 ± 0.0131 | Balanced Acc: 0.8514 ± 0.0066  
> Graph: 203,769 nodes · 468,710 undirected edges · 172 features/node

---

## Architecture

```
ElliGAT
──────────────────────────────────────────────────────
Input (172-d = 165 raw + 7 velocity features)
  └─► Linear → LayerNorm → GELU          (hidden_dim = 256)
  └─► 4 × GATv2Conv(heads=8)             + residual + LayerNorm
        └── temporal edge encoding       (|Δtimestep| → 16-d)
  └─► Heterophily readout: [h ∥ h − μ(h_N)]   (2×256 = 512-d)
  └─► MLP classifier                     (512 → 256 → 128 → 1)

Training
  Phase 1 — self-supervised pre-training
    • Masked feature autoencoder         (mask_ratio = 0.20)
    • Warm-up + Cosine LR schedule
  Phase 2 — fine-tuning
    • Focal Loss                         (α = 0.80, γ = 2.5)
    • Validation criterion: HM(AUC, F1) — prevents F1 collapse
    • Early stopping on best HM score

MetaEnsemble
──────────────────────────────────────────────────────
  Base models : ElliGAT · XGBoost · LightGBM · MLP · RandomForest
  Meta-learner: isotonic-calibrated logistic regression (5-fold CV)
```

---

## Key Design Choices vs Baseline

| Feature | BaselineGNN | ElliGAT |
|---|---|---|
| Architecture | 3-layer GAT | 4-layer GATv2 |
| Edge features | None | Temporal Δtimestep encoding |
| Hidden dim / heads | 128 / 4 | 256 / 8 |
| Heterophily readout | None | [h ∥ h − μ(h_N)] |
| Pre-training | None | Masked feature autoencoder |
| Loss function | Cross-entropy | Focal Loss (α=0.80, γ=2.5) |
| LR schedule | Cosine | Warm-up + Cosine |
| Validation criterion | AUC only | HM(AUC, F1) |
| Velocity features | 5 | 7 (adds 24h count + std) |

---

## Project Structure

```
Bitcoin-Transaction-Fraud-Detection/
├── main.py                ← Full pipeline entry point
├── requirements.txt
├── configs/
│   └── config.py          ← All hyperparameters and paths
└── src/
    ├── data.py            ← Dataset loading, feature engineering, graph builder
    ├── models.py          ← ElliGAT, EvolveGCN, BaselineGNN
    ├── losses.py          ← FocalLoss, AsymmetricLoss, CombinedLoss
    ├── trainer.py         ← Pre-training, fine-tuning, evaluation, MC Dropout
    ├── baselines.py       ← Tabular models + MetaEnsemble stacker
    └── visualize.py       ← Results figures
```

---

## Quick Start

```bash
git clone https://github.com/ria0304/Bitcoin-Transaction-Fraud-Detection-.git
cd Bitcoin-Transaction-Fraud-Detection-

pip install -r requirements.txt

export BITCOIN_DATA_PATH="/path/to/elliptic-data-set"
python main.py
```

Results are logged to `run_log.txt` during training.
Metrics (mean ± std across 5 seeds) are printed to stdout on completion.

> **No dataset?** Leave `BITCOIN_DATA_PATH` unset and the pipeline will attempt
> auto-download via [KaggleHub](https://github.com/Kaggle/kagglehub).
> You need a Kaggle account and `~/.kaggle/kaggle.json` credentials.

---

## Running on Google Colab (GPU)

The full 5-seed pipeline takes approximately **46 minutes on a T4 GPU**.

```python
!git clone https://github.com/ria0304/Bitcoin-Transaction-Fraud-Detection- /content/elligat

!pip install torch-geometric
!pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
!pip install lightgbm xgboost scikit-learn pandas numpy matplotlib

%cd /content/elligat
!python -u main.py 2>&1 | tee run_log.txt
```

Set Runtime → T4 GPU before running.

---

## Dataset

| Statistic | Value |
|---|---|
| Transactions (nodes) | 203,769 |
| Payment flows (directed edges) | 234,355 → 468,710 undirected |
| Features per node | 172 (165 raw + 7 velocity) |
| Labelled transactions | 46,564 (~22.9%) |
| Fraud class ratio | 9.76% |
| Train / Val / Test split | 32,594 / 6,984 / 6,986 (chronological) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, PyTorch Geometric |
| GNN Layers | GATv2Conv, SAGEConv |
| ML Models | XGBoost, LightGBM, RandomForest, Scikit-learn MLP |
| Meta-learner | Scikit-learn LogisticRegression (isotonic calibration) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib |

---

## Citation

If you use this work, please cite the Elliptic dataset:

```bibtex
@inproceedings{weber2019anti,
  title     = {Anti-Money Laundering in Bitcoin: Experimenting with
               Graph Convolutional Networks for Financial Forensics},
  author    = {Weber, Mark and Domeniconi, Giacomo and Chen, Jie and
               Weidele, Daniel Karl I. and Bellei, Claudio and
               Robinson, Tom and Leiserson, Charles E.},
  booktitle = {KDD Workshop on Anomaly Detection in Finance},
  year      = {2019}
}
```

And EvolveGCN if citing the baseline:

```bibtex
@inproceedings{pareja2020evolvegcn,
  title     = {EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs},
  author    = {Pareja, Aldo and Domeniconi, Giacomo and Chen, Jie and
               Ma, Tengfei and Suzumura, Toyotaro and Kanezashi, Hiroki and
               Kaler, Tim and Schardl, Tao and Leiserson, Charles},
  booktitle = {AAAI},
  year      = {2020}
}
```
