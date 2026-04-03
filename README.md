Bitcoin Fraud Detection using Graph Neural Networks

 Overview
This project implements an advanced Bitcoin transaction fraud detection system using Graph Neural Networks (GNNs) and machine learning models. It leverages both transaction behavior and blockchain graph relationships to accurately detect illicit activities in cryptocurrency networks.

Bitcoin transactions form a natural graph structure where:

Nodes = Transactions

Edges = Payment flows between transactions

This project exploits this graph structure to identify fraudulent patterns that traditional methods miss.

✨ Key Features
🔗 Graph-based fraud detection using Graph Neural Networks (GNN)

⏱️ Velocity features (transaction frequency and spending patterns per wallet)

📊 Multiple model comparison:

BitcoinGNN – 3‑layer Graph Attention Network (GAT) – Proposed Model

BaselineGNN – GraphSAGE (GNN without attention) – Ablation

MLP Classifier – Tabular baseline

XGBoost – Gradient boosting baseline (optional)

📈 Evaluation metrics: ROC-AUC, F1‑score, Precision, Recall, MCC, Balanced Accuracy

🔁 Multi‑seed training (5 seeds) for robust, reproducible results

🎲 Uncertainty estimation using Monte Carlo (MC) Dropout

📐 Calibration analysis with Expected Calibration Error (ECE)

🧠 Models Used
BitcoinGNN (Proposed)
3‑layer Graph Attention Network (GAT)

Multi‑head attention (4 heads)

Residual connections + LayerNorm

Designed for node‑level fraud classification on Bitcoin transaction graph

BaselineGNN (Ablation)
2‑layer GraphSAGE

Simple mean aggregation (no attention)

Demonstrates value of attention mechanism

MLP Classifier
Fully connected neural network (256 → 128 → 64)

Uses only tabular features (no graph structure)

XGBoost
Gradient boosting with scale_pos_weight for imbalance

Strong tabular baseline for comparison

📂 Dataset
This project uses the Elliptic Bitcoin Dataset – a publicly available benchmark for Bitcoin transaction fraud detection (Anti‑Money Laundering).

Statistic	Value
Transactions (nodes)	203,769
Payment flows (edges)	234,355
Raw features per node	166 (94 local + 72 aggregated)
Labelled transactions	~15‑20%
Fraud class ratio	~4.5% (highly imbalanced)
Download dataset from Kaggle:
🔗 Elliptic Data Set

Note: Dataset is not included in this repository due to size constraints. The code will attempt to download it automatically via kagglehub.

🛠️ Tech Stack
Category	Technologies
Deep Learning	PyTorch, PyTorch Geometric
GNN Layers	GATConv, SAGEConv
ML Models	XGBoost, Scikit-learn MLP
Data Processing	Pandas, NumPy
Visualization	Matplotlib
Statistics	SciPy (Wilcoxon test)
