# ElliGAT — Bitcoin Transaction Fraud Detection

> **Heterophily-aware graph attention network** for illicit transaction detection on the
> [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
> ElliGAT combines a 4-layer GATv2 encoder with temporal edge encoding, a heterophily
> readout, and a MetaEnsemble stacker — achieving **F1 = 0.7925 ± 0.0104** on the GNN
> component and **F1 = 0.8629 ± 0.0038** with the full ensemble (5-seed average).

---

## Results

All results are averaged over **5 random seeds** (42, 123, 2024, 17, 99) on the
chronological 70/15/15 train/val/test split of the Elliptic dataset.
EvolveGCN numbers are taken from the original paper (Pareja et al., 2020) as the
model could not be reproduced under available GPU memory constraints.

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

> Graph edges (468,710 undirected) were loaded for all experiments.
> Precision: 0.8976 ± 0.0112 | Recall: 0.7096 ± 0.0131 | Balanced Acc: 0.8514 ± 0.0066

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

Training objectives:
  • Masked-feature autoencoder  (pre-training, mask_ratio = 0.20)
  • Focal Loss                  (α = 0.80, γ = 2.5)
  • Validation criterion        HM(AUC, F1) — prevents F1 collapse
  • LR schedule                 Warm-up + Cosine annealing

MetaEnsemble
──────────────────────────────────────────────────────
  Base models : ElliGAT · XGBoost · LightGBM · MLP · RandomForest
  Meta-learner: isotonic-calibrated logistic regression (5-fold CV)
```

---

## Why Heterophily Awareness?

Illicit transactions in Elliptic are *heterophilous* — they are often directly
connected to licit transactions (mixing services, layering). Standard message
passing averages neighbour embeddings, which dilutes the fraud signal.

ElliGAT's heterophily readout `[h_i ∥ h_i − μ(h_Nj)]` explicitly encodes
*how different* a node is from its neighbourhood — a strong discriminative
signal for fraud detection in transaction graphs.

---

## Key Design Choices vs Baseline

| Feature | BaselineGNN | ElliGAT |
|---|---|---|
| Architecture | 3-layer GAT | 4-layer GATv2 |
| Edge features | None | Temporal Δtimestep encoding |
| Hidden dim / heads | 128 / 4 | 256 / 8 |
| Heterophily readout | None | [h ∥ h − μ(h_N)] |
| Pre-training | None | Masked feature autoencoder |
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

Outputs saved to `outputs/`:
- `best_model.pt`  — ElliGAT weights
- `results.png`    — results figure
- `metrics.json`   — all metrics (mean ± std, 5 seeds)

> **No dataset?** Leave `BITCOIN_DATA_PATH` unset and the pipeline will attempt
> auto-download via [KaggleHub](https://github.com/Kaggle/kagglehub).
> You need a Kaggle account and `~/.kaggle/kaggle.json` credentials.

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

## Running on Google Colab (GPU)

The full 5-seed pipeline takes approximately **46 minutes on a T4 GPU**.

```python
!git clone https://github.com/ria0304/Bitcoin-Transaction-Fraud-Detection- /content/elligat

!pip install torch-geometric
!pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
!pip install lightgbm xgboost scikit-learn pandas numpy matplotlib

%cd /content/elligat
!python main.py
```

Set Runtime → T4 GPU before running.

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
