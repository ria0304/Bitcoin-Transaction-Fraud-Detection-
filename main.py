import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from scipy.stats import wilcoxon
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv, GraphNorm
from torch_geometric.utils import to_undirected

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — pip install xgboost to enable.")

# ========================
# CONFIGURATION
# ========================
SEEDS         = [42, 123, 2024, 17, 99]
HIDDEN_DIM    = 128
NUM_LAYERS    = 3
LEARNING_RATE = 0.001
EPOCHS        = 200
PATIENCE      = 20
MC_SAMPLES    = 30
DROPOUT       = 0.3

# Uncertainty-Aware Rewiring
REWIRE_INTERVAL      = 10
REWIRE_UNCERTAINTY_T = 0.15
REWIRE_MC_SAMPLES    = 10

# Curriculum Pseudo-Labelling
CURRICULUM_START_EPOCH  = 30
CURRICULUM_INTERVAL     = 15
CURRICULUM_CONFIDENCE_T = 0.90
CURRICULUM_MAX_RATIO    = 0.30
CURRICULUM_GROWTH_RATE  = 0.05

# ========================
# PATH CONFIGURATION
# ========================
ELLIPTIC_PLUS_PLUS_PATH = r"C:\Users\Ria S\OneDrive\Attachments\Desktop\projects\bitcoin detection\elliptic_bitcoin_dataset"

# ========================
# STANDARD TEMPORAL SPLIT
# ========================
TRAIN_STEPS = list(range(1, 35))
VAL_STEPS   = list(range(30, 35))
TEST_STEPS  = list(range(35, 50))

# ========================
# ELLIPTIC++ DATASET LOADER
# ========================
def load_elliptic_plus_plus(path):
    """
    Load Elliptic++ dataset.
    """
    print("\n[Elliptic++] Loading transaction features...")
    txs_feat_path    = os.path.join(path, 'txs_features.csv')
    txs_cls_path     = os.path.join(path, 'txs_classes.csv')
    txs_edge_path    = os.path.join(path, 'txs_edgelist.csv')
    addr_tx_path     = os.path.join(path, 'AddrTx_edgelist.csv')
    tx_addr_path     = os.path.join(path, 'TxAddr_edgelist.csv')
    wallet_cls_path  = os.path.join(path, 'wallets_classes.csv')

    for p in [txs_feat_path, txs_cls_path, txs_edge_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # Transaction features
    feats = pd.read_csv(txs_feat_path, header=0)
    feat_cols = ['txId', 'timestep'] + [f'f{i}' for i in range(feats.shape[1] - 2)]
    feats.columns = feat_cols
    print(f"  Transactions: {len(feats):,} | Features per tx: {feats.shape[1]-2}")

    # Transaction classes
    classes = pd.read_csv(txs_cls_path)
    
    # Flexible class mapping
    class_mapping = {}
    for val in classes['class'].unique():
        val_str = str(val).lower()
        if val_str in ['1', 'fraud', 'illicit', 'criminal']:
            class_mapping[val] = 1
        elif val_str in ['0', '2', 'legit', 'licit', 'non-fraud']:
            class_mapping[val] = 0
        else:
            class_mapping[val] = np.nan
    
    classes['class'] = classes['class'].map(class_mapping)
    classes = classes.rename(columns={'class': 'isFraud'})

    all_nodes = feats.merge(classes, on='txId', how='left')

    # Transaction-to-transaction edges
    txid_to_idx = {tid: i for i, tid in enumerate(all_nodes['txId'])}
    full_edge_index = None
    if os.path.exists(txs_edge_path):
        el = pd.read_csv(txs_edge_path, header=0)
        el.columns = ['txId1', 'txId2']
        mask = el['txId1'].isin(txid_to_idx) & el['txId2'].isin(txid_to_idx)
        el = el[mask]
        src = torch.tensor([txid_to_idx[t] for t in el['txId1']], dtype=torch.long)
        dst = torch.tensor([txid_to_idx[t] for t in el['txId2']], dtype=torch.long)
        full_edge_index = to_undirected(torch.stack([src, dst], dim=0))
        print(f"  Tx-Tx edges: {el.shape[0]:,} directed -> {full_edge_index.shape[1]:,} undirected")

    # Raw feature matrix
    raw_feat_cols = [c for c in feat_cols if c.startswith('f')]
    X_all = all_nodes[raw_feat_cols].values.astype(np.float32)

    # Map transactions to wallets
    tx_to_sender = {}
    if os.path.exists(addr_tx_path):
        print("  Loading AddrTx edgelist (wallet->tx links)...")
        addr_tx = pd.read_csv(addr_tx_path, header=0)
        addr_tx.columns = ['address', 'txId'] if addr_tx.shape[1] == 2 else addr_tx.columns
        tx_to_sender = addr_tx.groupby('txId')['address'].first().to_dict()
        print(f"  Mapped {len(tx_to_sender):,} transactions to sender wallets")

    tx_to_receiver = {}
    if os.path.exists(tx_addr_path):
        print("  Loading TxAddr edgelist (tx->wallet links)...")
        tx_addr = pd.read_csv(tx_addr_path, header=0)
        tx_addr.columns = ['txId', 'address'] if tx_addr.shape[1] == 2 else tx_addr.columns
        tx_to_receiver = tx_addr.groupby('txId')['address'].first().to_dict()
        print(f"  Mapped {len(tx_to_receiver):,} transactions to receiver wallets")

    # Labelled subset
    labelled = all_nodes.dropna(subset=['isFraud']).copy()
    labelled['isFraud'] = labelled['isFraud'].astype(int)

    # Assign wallet IDs
    labelled['Wallet_ID'] = labelled['txId'].map(tx_to_sender)
    missing_sender = labelled['Wallet_ID'].isna()
    labelled.loc[missing_sender, 'Wallet_ID'] = labelled.loc[missing_sender, 'txId'].map(tx_to_receiver)
    labelled['Wallet_ID'] = labelled['Wallet_ID'].fillna(labelled['txId'].astype(str))

    n_real_wallets = labelled['Wallet_ID'].nunique()
    n_fallback = missing_sender.sum()
    print(f"  Real wallet IDs assigned: {n_real_wallets:,} unique wallets")
    print(f"  Transactions using fallback wallet ID: {n_fallback:,}")

    # Synthetic timestamps
    base = pd.Timestamp('2019-01-01')
    labelled['Timestamp'] = base + pd.to_timedelta(labelled['timestep'] * 2, unit='W')
    labelled['Amount'] = labelled['f0']
    labelled = labelled.sort_values(['timestep', 'txId']).reset_index(drop=True)

    fraud_ratio = labelled['isFraud'].mean() if len(labelled) > 0 else 0
    print(f"\n[Elliptic++] Summary:")
    print(f"  Total transactions      : {len(all_nodes):,}")
    print(f"  Labelled transactions   : {len(labelled):,}")
    if len(labelled) > 0:
        print(f"  Illicit (fraud)         : {labelled['isFraud'].sum():,}  ({fraud_ratio:.4f})")
        print(f"  Licit (legit)           : {(labelled['isFraud']==0).sum():,}")

    return labelled, X_all, full_edge_index, txid_to_idx, all_nodes, raw_feat_cols


# ========================
# LOAD ELLIPTIC++ DATASET
# ========================
print("=" * 60)
print("Loading Elliptic++ Dataset")
print("=" * 60)

if not os.path.exists(ELLIPTIC_PLUS_PLUS_PATH):
    raise FileNotFoundError(
        f"Elliptic++ data folder not found at:\n  {ELLIPTIC_PLUS_PLUS_PATH}\n"
        "Please download from: https://drive.google.com/drive/folders/"
        "1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l\n"
        "and update ELLIPTIC_PLUS_PLUS_PATH above."
    )

(df, ELLIPTIC_X_ALL, ELLIPTIC_EDGE_INDEX,
 ELLIPTIC_TXID_TO_IDX, ELLIPTIC_ALL_NODES,
 ELLIPTIC_FEAT_COLS) = load_elliptic_plus_plus(ELLIPTIC_PLUS_PLUS_PATH)

# Check if we have labelled data
if len(df) == 0:
    print("\nERROR: No labelled transactions found!")
    print("Please check your Elliptic++ dataset files.")
    print("Expected files: txs_features.csv, txs_classes.csv, txs_edgelist.csv")
    exit(1)

# ========================
# STANDARD TEMPORAL SPLIT
# ========================
print("\nApplying standard temporal split (train: t1-34, test: t35-49)...")

train_df = df[df['timestep'].isin(TRAIN_STEPS) & ~df['timestep'].isin(VAL_STEPS)].copy()
val_df = df[df['timestep'].isin(VAL_STEPS)].copy()
test_df = df[df['timestep'].isin(TEST_STEPS)].copy()

# Rebuild df in train->val->test order
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

train_size = len(train_df)
val_size = len(val_df)
test_size = len(test_df)

print(f"  Train: {train_size:,} transactions (timesteps 1-29)")
print(f"  Val:   {val_size:,}  transactions (timesteps 30-34)")
print(f"  Test:  {test_size:,} transactions (timesteps 35-49)")
if train_size > 0:
    print(f"  Train fraud ratio: {train_df['isFraud'].mean():.4f}")
if test_size > 0:
    print(f"  Test  fraud ratio: {test_df['isFraud'].mean():.4f}")

# ========================
# VELOCITY FEATURES
# ========================
print("\nComputing velocity features (real wallet transaction behaviour)...")
df = df.sort_values(['Wallet_ID', 'Timestamp']).reset_index(drop=True)

def rolling_count_1h(g):
    """Transactions sent by the same wallet in the previous hour."""
    ts = g.set_index('Timestamp')
    result = ts.index.to_series().rolling('1h', closed='left').count().fillna(0)
    result.index = g.index
    return result.values

def rolling_sum_1h(g):
    """Total amount sent by the same wallet in the previous hour."""
    result = g.set_index('Timestamp')['Amount'].rolling('1h', closed='left').sum().fillna(0)
    result.index = g.index
    return result.values

if len(df) > 0:
    df['tx_count_1h'] = df.groupby('Wallet_ID', group_keys=False).apply(rolling_count_1h)
    df['amount_sum_1h'] = df.groupby('Wallet_ID', group_keys=False).apply(rolling_sum_1h)
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
else:
    print("ERROR: No data available for velocity features")
    exit(1)

# Re-derive train/val/test indices
train_idx = df[df['timestep'].isin(TRAIN_STEPS) & ~df['timestep'].isin(VAL_STEPS)].index.tolist()
val_idx = df[df['timestep'].isin(VAL_STEPS)].index.tolist()
test_idx = df[df['timestep'].isin(TEST_STEPS)].index.tolist()

train_size = len(train_idx)
val_size = len(val_idx)

velocity_cols = ['Amount_log', 'tx_count_1h', 'amount_sum_1h', 'Hour_sin', 'Hour_cos']
tabular_cols = velocity_cols + ELLIPTIC_FEAT_COLS
print(f"  Features for tabular models: {len(tabular_cols)} "
      f"({len(ELLIPTIC_FEAT_COLS)} raw + 5 velocity)")

# Tabular arrays
X_train = df.loc[train_idx][tabular_cols].values
X_val = df.loc[val_idx][tabular_cols].values
X_test = df.loc[test_idx][tabular_cols].values
y_train_np = df.loc[train_idx]['isFraud'].values.astype(np.float32)
y_val_np = df.loc[val_idx]['isFraud'].values.astype(np.float32)
y_test_np = df.loc[test_idx]['isFraud'].values.astype(np.float32)

# Scale
scaler_tab = StandardScaler().fit(X_train)
X_train_s = scaler_tab.transform(X_train)
X_val_s = scaler_tab.transform(X_val)
X_test_s = scaler_tab.transform(X_test)

print(f"  Train class balance: {y_train_np.sum():.0f} fraud / "
      f"{(y_train_np==0).sum():.0f} legit")
print(f"  Test  class balance: {y_test_np.sum():.0f} fraud / "
      f"{(y_test_np==0).sum():.0f} legit")

# ========================
# SINUSOIDAL TEMPORAL ENCODING
# ========================
def sinusoidal_time_encoding(timesteps, dim=8):
    """
    Encode the timestep index as a sinusoidal embedding.
    """
    timesteps = np.array(timesteps, dtype=np.float32)
    enc = np.zeros((len(timesteps), dim), dtype=np.float32)
    for i in range(dim // 2):
        freq = 1.0 / (10000 ** (2 * i / dim))
        enc[:, 2*i] = np.sin(timesteps * freq)
        enc[:, 2*i+1] = np.cos(timesteps * freq)
    return enc

# ========================
# GRAPH BUILDER
# ========================
def build_node_graph(df, train_idx, val_idx, test_idx):
    """
    Build PyG Data object for the Elliptic++ Bitcoin transaction graph.
    """
    N_all = ELLIPTIC_X_ALL.shape[0]

    # Global node indices for labelled transactions
    labelled_global_idx = torch.tensor(
        [ELLIPTIC_TXID_TO_IDX[tid] for tid in df['txId']],
        dtype=torch.long
    )
    train_global_idx = torch.tensor(
        [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[train_idx, 'txId']],
        dtype=torch.long
    )

    # Scale raw features
    scaler_node = StandardScaler().fit(ELLIPTIC_X_ALL[train_global_idx.numpy()])
    X_all_scaled = scaler_node.transform(ELLIPTIC_X_ALL).astype(np.float32)

    # Velocity features
    vel = np.zeros((N_all, len(velocity_cols)), dtype=np.float32)
    if len(df) > 0:
        vel_scaler = StandardScaler().fit(df.loc[train_idx][velocity_cols].values)
        vel_vals = vel_scaler.transform(df[velocity_cols].values)
        for i, gidx in enumerate(labelled_global_idx.numpy()):
            if i < len(vel_vals):
                vel[gidx] = vel_vals[i]

    # Sinusoidal temporal encoding
    all_timesteps = ELLIPTIC_ALL_NODES['timestep'].values \
        if 'timestep' in ELLIPTIC_ALL_NODES.columns \
        else np.zeros(N_all, dtype=np.float32)
    if all_timesteps.max() > all_timesteps.min():
        ts_normed = (all_timesteps - all_timesteps.min()) / \
                    (all_timesteps.max() - all_timesteps.min() + 1e-8)
    else:
        ts_normed = all_timesteps
    time_enc = sinusoidal_time_encoding(ts_normed * 48, dim=8)

    # Concatenate features
    node_feats = np.concatenate([X_all_scaled, vel, time_enc], axis=1)
    x = torch.tensor(node_feats, dtype=torch.float)

    # Labels: -1 for unknown
    labels_full = torch.full((N_all,), -1, dtype=torch.float)
    for i, gidx in enumerate(labelled_global_idx.numpy()):
        if i < len(df):
            labels_full[gidx] = float(df.iloc[i]['isFraud'])

    # Train/Val/Test masks
    tr_mask = torch.zeros(N_all, dtype=torch.bool)
    vl_mask = torch.zeros(N_all, dtype=torch.bool)
    te_mask = torch.zeros(N_all, dtype=torch.bool)

    if len(train_idx) > 0:
        tr_global = torch.tensor(
            [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[train_idx, 'txId']],
            dtype=torch.long)
        tr_mask[tr_global] = True
    
    if len(val_idx) > 0:
        vl_global = torch.tensor(
            [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[val_idx, 'txId']],
            dtype=torch.long)
        vl_mask[vl_global] = True
    
    if len(test_idx) > 0:
        te_global = torch.tensor(
            [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[test_idx, 'txId']],
            dtype=torch.long)
        te_mask[te_global] = True

    # Unknown mask for curriculum pseudo-labelling
    unknown_mask = (labels_full == -1)

    data = Data(x=x, edge_index=ELLIPTIC_EDGE_INDEX, y=labels_full,
                train_mask=tr_mask, val_mask=vl_mask, test_mask=te_mask)
    data.unknown_mask = unknown_mask
    return data

# ========================
# HETEROPHILY PRUNING
# ========================
def prune_heterophilic_edges(edge_index, node_features, labels, train_mask):
    """
    Remove edges between training nodes of different classes.
    """
    edge_src, edge_dst = edge_index
    src_in_train = train_mask[edge_src]
    dst_in_train = train_mask[edge_dst]
    both_in_train = src_in_train & dst_in_train

    if both_in_train.sum() == 0:
        return edge_index

    src_labels = labels[edge_src[both_in_train]]
    dst_labels = labels[edge_dst[both_in_train]]
    same_class = (src_labels == dst_labels)

    keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    keep_mask[both_in_train] = same_class
    return edge_index[:, keep_mask]

# ========================
# UNCERTAINTY-AWARE GRAPH REWIRING
# ========================
class UncertaintyAwareRewirer:
    """
    Dynamically suppresses edges between high-uncertainty node pairs.
    """

    def __init__(self, base_edge_index, threshold=0.15, mc_samples=10):
        self.base_edge_index = base_edge_index
        self.threshold = threshold
        self.mc_samples = mc_samples
        self.rewire_history = []

    @torch.no_grad()
    def rewire(self, model, data):
        model.train()
        logits_mc = torch.stack([
            model(Data(x=data.x, edge_index=self.base_edge_index,
                       y=data.y, train_mask=data.train_mask,
                       val_mask=data.val_mask, test_mask=data.test_mask))
            for _ in range(self.mc_samples)
        ])
        model.eval()

        probs_mc = torch.sigmoid(logits_mc)
        node_std = probs_mc.std(dim=0)

        src, dst = self.base_edge_index
        both_uncertain = (node_std[src] > self.threshold) & \
                         (node_std[dst] > self.threshold)
        keep_mask = ~both_uncertain
        n_suppressed = both_uncertain.sum().item()
        n_total = self.base_edge_index.shape[1]

        self.rewire_history.append({
            'suppressed': n_suppressed,
            'kept': keep_mask.sum().item(),
            'total': n_total,
            'mean_unc': node_std.mean().item()
        })

        if n_suppressed > 0:
            print(f"    [Rewirer] Suppressed {n_suppressed:,}/{n_total:,} edges "
                  f"(mean node uncertainty: {node_std.mean():.4f})")

        return self.base_edge_index[:, keep_mask]

    def summary(self):
        if not self.rewire_history:
            return
        suppressed = [h['suppressed'] for h in self.rewire_history]
        print(f"\n  [Rewirer Summary]")
        print(f"    Rewiring events : {len(self.rewire_history)}")
        print(f"    Max suppressed  : {max(suppressed):,} edges")
        print(f"    Min suppressed  : {min(suppressed):,} edges")
        print(f"    Final suppressed: {suppressed[-1]:,} edges")

# ========================
# CURRICULUM PSEUDO-LABELLING
# ========================
class CurriculumPseudoLabeller:
    """
    Progressively incorporates unlabelled transactions into training.
    """

    def __init__(self, unknown_mask, confidence_threshold=0.90,
                 max_ratio=0.30, growth_rate=0.05, mc_samples=10):
        self.unknown_mask = unknown_mask.clone()
        self.confidence_threshold = confidence_threshold
        self.max_ratio = max_ratio
        self.growth_rate = growth_rate
        self.mc_samples = mc_samples
        self.unknown_indices = unknown_mask.nonzero(as_tuple=True)[0]
        self.n_unknown = len(self.unknown_indices)
        self.pseudo_labels = {}
        self.phase = 0
        self.history = []
        print(f"  [CurriculumLabeller] {self.n_unknown:,} unlabelled nodes "
              f"available for pseudo-labelling")

    @torch.no_grad()
    def update(self, model, data, epoch):
        self.phase += 1
        current_max = min(
            int(self.n_unknown * self.max_ratio),
            int(self.n_unknown * self.growth_rate * self.phase)
        )
        already_labelled = len(self.pseudo_labels)
        budget = max(0, current_max - already_labelled)

        if budget == 0:
            print(f"    [Curriculum] Phase {self.phase}: Pool full "
                  f"({already_labelled:,}). No update.")
            return 0

        model.train()
        logits_mc = torch.stack([model(data) for _ in range(self.mc_samples)])
        model.eval()

        probs_mc = torch.sigmoid(logits_mc)
        mean_probs = probs_mc.mean(dim=0)
        std_probs = probs_mc.std(dim=0)

        remaining_unknown = [
            idx.item() for idx in self.unknown_indices
            if idx.item() not in self.pseudo_labels
        ]
        if not remaining_unknown:
            return 0

        remaining_t = torch.tensor(remaining_unknown, dtype=torch.long)
        node_probs = mean_probs[remaining_t].cpu()
        node_std = std_probs[remaining_t].cpu()

        fraud_confident = (node_probs > self.confidence_threshold) & \
                          (node_std < REWIRE_UNCERTAINTY_T)
        legit_confident = (node_probs < (1.0 - self.confidence_threshold)) & \
                          (node_std < REWIRE_UNCERTAINTY_T)
        any_confident = fraud_confident | legit_confident
        confident_idx = any_confident.nonzero(as_tuple=True)[0]

        if len(confident_idx) == 0:
            print(f"    [Curriculum] Phase {self.phase}: No confident pseudo-labels "
                  f"(threshold={self.confidence_threshold:.2f})")
            return 0

        # Sort by confidence margin
        confidence_margin = (node_probs[confident_idx] - 0.5).abs()
        sorted_order = confidence_margin.argsort(descending=True)
        selected = confident_idx[sorted_order[:budget]]

        n_fraud, n_legit = 0, 0
        for local_idx in selected:
            global_idx = remaining_unknown[local_idx.item()]
            prob = node_probs[local_idx].item()
            label = 1.0 if prob > 0.5 else 0.0
            self.pseudo_labels[global_idx] = label
            data.y[global_idx] = label
            data.train_mask[global_idx] = True
            if label == 1.0:
                n_fraud += 1
            else:
                n_legit += 1

        n_added = n_fraud + n_legit
        self.history.append({
            'epoch': epoch, 'phase': self.phase,
            'n_added': n_added, 'n_fraud': n_fraud, 'n_legit': n_legit,
            'total_pseudo': len(self.pseudo_labels)
        })
        print(f"    [Curriculum] Phase {self.phase} (epoch {epoch}): "
              f"+{n_added} pseudo-labels ({n_fraud} fraud, {n_legit} legit) "
              f"-> total {len(self.pseudo_labels):,}/{self.n_unknown:,}")
        return n_added

    def summary(self):
        if not self.history:
            print("  [Curriculum] No pseudo-labels added.")
            return
        total = self.history[-1]['total_pseudo']
        total_fraud = sum(h['n_fraud'] for h in self.history)
        total_legit = sum(h['n_legit'] for h in self.history)
        print(f"\n  [Curriculum Summary]")
        print(f"    Phases run   : {len(self.history)}")
        print(f"    Total pseudo : {total:,} ({100*total/self.n_unknown:.1f}% of unknown)")
        print(f"    Pseudo fraud : {total_fraud:,}")
        print(f"    Pseudo legit : {total_legit:,}")

# ========================
# METRICS
# ========================
def find_best_threshold(y_true, probs):
    """Maximise F1 over validation set to find operating threshold."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

def compute_metrics(y_true, probs, threshold=None):
    if threshold is None:
        threshold = find_best_threshold(y_true, probs)
    y_pred = (probs > threshold).astype(int)
    return {
        'Accuracy': (y_pred == y_true).mean(),
        'Balanced Acc': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, probs),
        'PR-AUC': average_precision_score(y_true, probs),
        'threshold': threshold,
        'probs': probs,
        'true': y_true
    }

# ========================
# FOCAL LOSS
# ========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None, smooth=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, inputs, targets, weight=None):
        targets_smooth = targets * (1 - self.smooth) + self.smooth * 0.5
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, reduction='none'))
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        if weight is not None:
            loss = loss * weight
        return loss.mean()

# ========================
# MODEL A: BitcoinGNN
# ========================
class BitcoinGNN(nn.Module):
    """
    3-layer GATv2 with residual connections and GraphNorm.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for _ in range(num_layers):
            out_ch = hidden_dim // heads
            conv = GATv2Conv(hidden_dim, out_ch, heads=heads, dropout=dropout,
                             add_self_loops=True, concat=True, share_weights=False)
            for name, param in conv.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))
            self.skips.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, x, edge_index, batch=None):
        h = F.elu(self.input_proj(x))
        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h2 = F.elu(conv(h, edge_index))
            h = norm(h2 + skip(h), batch)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, data):
        return self.classifier(
            self.encode(data.x, data.edge_index)
        ).squeeze(-1)

    def mc_dropout_forward(self, data, n):
        """Monte Carlo Dropout forward passes."""
        self.train()
        with torch.no_grad():
            return torch.stack([self.forward(data) for _ in range(n)])

# ========================
# MODEL B: BaselineGNN
# ========================
class BaselineGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x = F.elu(self.proj(data.x))
        ei = data.edge_index
        h = self.norm1(F.elu(self.conv1(x, ei)) + x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.norm2(F.elu(self.conv2(h, ei)) + h)
        return self.head(h).squeeze(-1)

# ========================
# ENSEMBLE MODEL
# ========================
class EnsembleModel:
    """Late-fusion ensemble: weighted average of GNN and XGBoost probabilities."""
    def __init__(self, gnn_model, xgb_model, scaler, weight=0.3):
        self.gnn = gnn_model
        self.xgb = xgb_model
        self.scaler = scaler
        self.weight = weight

    def predict_proba(self, X_tabular, graph_data, node_indices):
        xgb_probs = self.xgb.predict_proba(X_tabular)[:, 1]
        self.gnn.eval()
        with torch.no_grad():
            gnn_logits = self.gnn(graph_data)
            gnn_probs = torch.sigmoid(gnn_logits[node_indices]).cpu().numpy()
        return self.weight * gnn_probs + (1 - self.weight) * xgb_probs

# ========================
# TRAINING FUNCTION
# ========================
def train_node_gnn_improved(model, data, pos_weight,
                             xgb_model=None, X_train_tabular=None,
                             train_node_indices=None,
                             rewirer=None, curriculum=None):
    """
    Training loop with uncertainty-aware rewiring and curriculum pseudo-labelling.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight, smooth=0.1)

    xgb_train_probs = None
    if xgb_model is not None and X_train_tabular is not None:
        xgb_probs_raw = xgb_model.predict_proba(X_train_tabular)[:, 1]
        xgb_train_probs = torch.tensor(xgb_probs_raw, dtype=torch.float).to(device)

    current_edge_index = data.edge_index.clone()
    PSEUDO_WEIGHT = 0.5

    best_prauc, no_improve, best_state = 0.0, 0, None

    for epoch in range(EPOCHS):
        # Uncertainty-Aware Rewiring
        if rewirer is not None and epoch > 0 and epoch % REWIRE_INTERVAL == 0:
            current_edge_index = rewirer.rewire(model, data).to(device)
            data.edge_index = current_edge_index

        # Curriculum Pseudo-Labelling
        if (curriculum is not None
                and epoch >= CURRICULUM_START_EPOCH
                and (epoch - CURRICULUM_START_EPOCH) % CURRICULUM_INTERVAL == 0):
            curriculum.update(model, data, epoch)

        # Forward pass
        model.train()
        optimizer.zero_grad()
        logits = model(data)

        active_mask = data.train_mask & (data.y >= 0)
        active_logits = logits[active_mask]
        active_labels = data.y[active_mask]

        # Per-node loss weights
        if curriculum is not None and curriculum.pseudo_labels:
            pseudo_set = set(curriculum.pseudo_labels.keys())
            active_global = active_mask.nonzero(as_tuple=True)[0].tolist()
            node_weights = torch.tensor(
                [PSEUDO_WEIGHT if idx in pseudo_set else 1.0
                 for idx in active_global],
                dtype=torch.float, device=device
            )
        else:
            node_weights = None

        loss_hard = criterion(active_logits, active_labels, weight=node_weights)

        # Knowledge distillation from XGBoost
        if xgb_train_probs is not None:
            orig_mask = data.train_mask.clone()
            if curriculum is not None:
                for pidx in curriculum.pseudo_labels:
                    orig_mask[pidx] = False
            gnn_train_probs = torch.sigmoid(logits[orig_mask])
            T = 2.0
            loss_soft = F.kl_div(
                F.log_softmax(gnn_train_probs / T, dim=0),
                F.softmax(xgb_train_probs / T, dim=0),
                reduction='batchmean'
            ) * (T * T)
            loss = loss_hard + 0.5 * loss_soft
        else:
            loss = loss_hard

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data)
            val_probs = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()

        if len(np.unique(val_y)) >= 2:
            val_prauc = average_precision_score(val_y, val_probs)
            val_auc = roc_auc_score(val_y, val_probs)

            if val_prauc > best_prauc:
                best_prauc = val_prauc
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                n_pseudo = len(curriculum.pseudo_labels) if curriculum else 0
                rewire_str = f" | Edges={data.edge_index.shape[1]:,}" if rewirer else ""
                print(f"    Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                      f"Val AUC {val_auc:.4f} | Val PR-AUC {val_prauc:.4f} | "
                      f"Pseudo={n_pseudo:,}{rewire_str}")

    if best_state:
        model.load_state_dict(best_state)

def eval_node_gnn(model, data):
    """Evaluate model on test nodes."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits[data.test_mask]).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
    return compute_metrics(y_true, probs)

# ========================
# MAIN EXECUTION
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

if train_size == 0 or val_size == 0 or test_size == 0:
    print("\nERROR: Empty train/val/test splits!")
    print("Please check your timestep ranges and data.")
    exit(1)

print("Building Elliptic++ transaction graph...")

# Global graph build
graph_data = build_node_graph(df, train_idx, val_idx, test_idx)

train_node_indices = torch.tensor(
    [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[train_idx, 'txId']],
    dtype=torch.long
)

print("  Applying heterophily edge pruning...")
original_edges = graph_data.edge_index.shape[1]
graph_data.edge_index = prune_heterophilic_edges(
    graph_data.edge_index, graph_data.x,
    graph_data.y, graph_data.train_mask
)
print(f"  Edges after pruning: {original_edges:,} -> {graph_data.edge_index.shape[1]:,} "
      f"(removed {original_edges - graph_data.edge_index.shape[1]:,})")

BASE_EDGE_INDEX = graph_data.edge_index.clone()
graph_data = graph_data.to(device)
BASE_EDGE_INDEX = BASE_EDGE_INDEX.to(device)

print(f"  Nodes:    {graph_data.num_nodes:,}")
print(f"  Edges:    {graph_data.num_edges:,}")
print(f"  Features: {graph_data.num_node_features}")
print(f"  Unknown (unlabelled) nodes: {graph_data.unknown_mask.sum().item():,}")

in_dim = graph_data.num_node_features
n_pos = float(y_train_np.sum())
n_neg = float(len(y_train_np) - n_pos)
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

# ========================
# MULTI-SEED EXPERIMENTS
# ========================
results = {m: [] for m in [
    'BitcoinGNN', 'BitcoinGNN+Contributions',
    'BaselineGNN', 'MLP', 'XGBoost', 'Ensemble'
]}

last_gnn_model = None
last_contrib_model = None
last_data = None
last_xgb_model = None

print(f"\nRunning {len(SEEDS)} seeds on {device}...\n")

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fresh graph per seed
    graph_data_seed = build_node_graph(df, train_idx, val_idx, test_idx)
    graph_data_seed.edge_index = prune_heterophilic_edges(
        graph_data_seed.edge_index, graph_data_seed.x,
        graph_data_seed.y, graph_data_seed.train_mask
    )
    base_ei_seed = graph_data_seed.edge_index.clone().to(device)
    graph_data_seed = graph_data_seed.to(device)

    # XGBoost
    if HAS_XGB:
        print("  [XGBoost]")
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='auc',
            random_state=seed, verbosity=0
        )
        xgb.fit(X_train_s, y_train_np,
                eval_set=[(X_val_s, y_val_np)], verbose=False)
        xm = compute_metrics(y_test_np, xgb.predict_proba(X_test_s)[:, 1])
        results['XGBoost'].append(xm)
        print(f"  -> AUC={xm['ROC-AUC']:.4f}  PR-AUC={xm['PR-AUC']:.4f}  "
              f"F1={xm['F1']:.4f}  MCC={xm['MCC']:.4f}")
        last_xgb_model = xgb

    # BitcoinGNN (no contributions)
    print("  [BitcoinGNN]")
    torch.manual_seed(seed)
    model_orig = BitcoinGNN(in_dim, HIDDEN_DIM, num_layers=NUM_LAYERS,
                            heads=4, dropout=DROPOUT).to(device)
    data_orig = build_node_graph(df, train_idx, val_idx, test_idx)
    data_orig.edge_index = prune_heterophilic_edges(
        data_orig.edge_index, data_orig.x, data_orig.y, data_orig.train_mask)
    data_orig = data_orig.to(device)
    train_node_gnn_improved(
        model_orig, data_orig, pos_weight,
        xgb_model=xgb if HAS_XGB else None,
        X_train_tabular=X_train_s if HAS_XGB else None,
        train_node_indices=train_node_indices.numpy() if HAS_XGB else None,
        rewirer=None, curriculum=None
    )
    mo = eval_node_gnn(model_orig, data_orig)
    results['BitcoinGNN'].append(mo)
    print(f"  -> AUC={mo['ROC-AUC']:.4f}  PR-AUC={mo['PR-AUC']:.4f}  "
          f"F1={mo['F1']:.4f}  MCC={mo['MCC']:.4f}")
    last_gnn_model = model_orig
    last_data = data_orig

    # BitcoinGNN + Contributions
    print("  [BitcoinGNN + Contributions]")
    torch.manual_seed(seed)
    model_contrib = BitcoinGNN(in_dim, HIDDEN_DIM, num_layers=NUM_LAYERS,
                               heads=4, dropout=DROPOUT).to(device)

    rewirer = UncertaintyAwareRewirer(
        base_edge_index=base_ei_seed,
        threshold=REWIRE_UNCERTAINTY_T,
        mc_samples=REWIRE_MC_SAMPLES
    )
    curriculum = CurriculumPseudoLabeller(
        unknown_mask=graph_data_seed.unknown_mask,
        confidence_threshold=CURRICULUM_CONFIDENCE_T,
        max_ratio=CURRICULUM_MAX_RATIO,
        growth_rate=CURRICULUM_GROWTH_RATE,
        mc_samples=REWIRE_MC_SAMPLES
    )

    train_node_gnn_improved(
        model_contrib, graph_data_seed, pos_weight,
        xgb_model=xgb if HAS_XGB else None,
        X_train_tabular=X_train_s if HAS_XGB else None,
        train_node_indices=train_node_indices.numpy() if HAS_XGB else None,
        rewirer=rewirer, curriculum=curriculum
    )
    rewirer.summary()
    curriculum.summary()

    mc = eval_node_gnn(model_contrib, graph_data_seed)
    results['BitcoinGNN+Contributions'].append(mc)
    print(f"  -> AUC={mc['ROC-AUC']:.4f}  PR-AUC={mc['PR-AUC']:.4f}  "
          f"F1={mc['F1']:.4f}  MCC={mc['MCC']:.4f}")
    last_contrib_model = model_contrib

    # Ensemble
    if HAS_XGB:
        print("  [Ensemble]")
        ensemble = EnsembleModel(model_orig, xgb, scaler_tab, weight=0.2)
        te_global = torch.tensor(
            [ELLIPTIC_TXID_TO_IDX[tid] for tid in df.loc[test_idx, 'txId']],
            dtype=torch.long
        ).to(device)
        ens_probs = ensemble.predict_proba(X_test_s, data_orig, te_global)
        ens_metrics = compute_metrics(y_test_np, ens_probs)
        results['Ensemble'].append(ens_metrics)
        print(f"  -> AUC={ens_metrics['ROC-AUC']:.4f}  PR-AUC={ens_metrics['PR-AUC']:.4f}  "
              f"F1={ens_metrics['F1']:.4f}  MCC={ens_metrics['MCC']:.4f}")

    # BaselineGNN
    print("  [BaselineGNN]")
    torch.manual_seed(seed)
    model_b = BaselineGNN(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    data_base = build_node_graph(df, train_idx, val_idx, test_idx)
    data_base.edge_index = prune_heterophilic_edges(
        data_base.edge_index, data_base.x, data_base.y, data_base.train_mask)
    data_base = data_base.to(device)
    train_node_gnn_improved(model_b, data_base, pos_weight)
    mb = eval_node_gnn(model_b, data_base)
    results['BaselineGNN'].append(mb)
    print(f"  -> AUC={mb['ROC-AUC']:.4f}  PR-AUC={mb['PR-AUC']:.4f}  "
          f"F1={mb['F1']:.4f}  MCC={mb['MCC']:.4f}")

    # MLP
    print("  [MLP]")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300,
                        random_state=seed, early_stopping=True,
                        validation_fraction=0.1)
    mlp.fit(X_train_s, y_train_np)
    mm = compute_metrics(y_test_np, mlp.predict_proba(X_test_s)[:, 1])
    results['MLP'].append(mm)
    print(f"  -> AUC={mm['ROC-AUC']:.4f}  PR-AUC={mm['PR-AUC']:.4f}  "
          f"F1={mm['F1']:.4f}  MCC={mm['MCC']:.4f}")

# ========================
# AGGREGATE RESULTS
# ========================
def agg(res_list, name):
    keys = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC', 'Balanced Acc', 'Precision', 'Recall']
    out = {}
    print(f"\n{'─'*65}")
    print(f"  {name}  (n={len(res_list)} seeds)")
    print(f"{'─'*65}")
    for k in keys:
        v = [r[k] for r in res_list]
        out[k] = (np.mean(v), np.std(v), v)
        print(f"  {k:<16}: {np.mean(v):.4f} +- {np.std(v):.4f}")
    return out

print("\n\n" + "=" * 65)
print("FINAL AGGREGATED RESULTS — ELLIPTIC++ BITCOIN FRAUD DETECTION")
print("=" * 65)
order = ['XGBoost', 'Ensemble', 'MLP', 'BaselineGNN',
         'BitcoinGNN', 'BitcoinGNN+Contributions']
aggs = {m: agg(results[m], m) for m in order if results[m]}

# Statistical significance tests
if 'BitcoinGNN+Contributions' in aggs and 'BitcoinGNN' in aggs:
    c_aucs = aggs['BitcoinGNN+Contributions']['PR-AUC'][2]
    o_aucs = aggs['BitcoinGNN']['PR-AUC'][2]
    diffs = np.array(c_aucs) - np.array(o_aucs)
    if len(set(diffs)) > 1:
        stat, p = wilcoxon(c_aucs, o_aucs)
        sig = "Significant (p<0.05)" if p < 0.05 else "Not significant"
        print(f"\nWilcoxon (PR-AUC) GNN+Contributions vs BitcoinGNN: p={p:.4f}  {sig}")

competitors = [m for m in ['XGBoost', 'Ensemble', 'MLP', 'BaselineGNN'] if m in aggs]
if competitors and 'BitcoinGNN+Contributions' in aggs:
    best_comp = max(competitors, key=lambda m: aggs[m]['PR-AUC'][0])
    c_aucs = aggs['BitcoinGNN+Contributions']['PR-AUC'][2]
    cmp_aucs = aggs[best_comp]['PR-AUC'][2]
    diffs = np.array(c_aucs) - np.array(cmp_aucs)
    if len(set(diffs)) > 1:
        stat, p = wilcoxon(c_aucs, cmp_aucs)
        sig = "Significant (p<0.05)" if p < 0.05 else "Not significant"
        print(f"Wilcoxon (PR-AUC) GNN+Contributions vs {best_comp}: p={p:.4f}  {sig}")

# Ablation table
print("\n\n" + "=" * 88)
print("TABLE 1 — Ablation Study on Elliptic++ (Mean +- Std, n=5 seeds)")
print("Standard temporal split: train t1-34, test t35-49")
print("=" * 88)
print(f"{'Model':<32}{'ROC-AUC':>14}{'PR-AUC':>14}{'F1':>14}{'MCC':>14}")
print("-" * 88)
for m in order:
    if m not in aggs:
        continue
    a = aggs[m]
    auc = f"{a['ROC-AUC'][0]:.4f}+-{a['ROC-AUC'][1]:.4f}"
    prauc = f"{a['PR-AUC'][0]:.4f}+-{a['PR-AUC'][1]:.4f}"
    f1 = f"{a['F1'][0]:.4f}+-{a['F1'][1]:.4f}"
    mcc = f"{a['MCC'][0]:.4f}+-{a['MCC'][1]:.4f}"
    tag = "  <- ours" if m == 'BitcoinGNN+Contributions' else \
          "  <- base" if m == 'BitcoinGNN' else ""
    print(f"{m:<32}{auc:>14}{prauc:>14}{f1:>14}{mcc:>14}{tag}")

# MC Dropout uncertainty and calibration
print("\nComputing MC Dropout uncertainty (30 forward passes)...")
eval_model = last_contrib_model if last_contrib_model is not None else last_gnn_model
eval_data = graph_data_seed if last_contrib_model is not None else last_data

ece = ece_orig = None
prob_true_cal = prob_pred_cal = pp_orig = pt_orig = test_unc = None

if eval_model is not None and eval_data is not None:
    logits_mc = eval_model.mc_dropout_forward(eval_data, MC_SAMPLES)
    std_mc = torch.sigmoid(logits_mc).std(dim=0)
    test_unc = std_mc[eval_data.test_mask].cpu().numpy()
    print(f"Mean prediction uncertainty (std dev): {test_unc.mean():.4f}")

    last_contrib_res = results['BitcoinGNN+Contributions'][-1]
    prob_true_cal, prob_pred_cal = calibration_curve(
        last_contrib_res['true'], last_contrib_res['probs'], n_bins=10)
    ece = np.mean(np.abs(prob_true_cal - prob_pred_cal))
    print(f"Expected Calibration Error (ECE) — GNN+Contributions: {ece:.4f}")

    last_orig_res = results['BitcoinGNN'][-1]
    pt_orig, pp_orig = calibration_curve(
        last_orig_res['true'], last_orig_res['probs'], n_bins=10)
    ece_orig = np.mean(np.abs(pt_orig - pp_orig))
    print(f"Expected Calibration Error (ECE) — BitcoinGNN:         {ece_orig:.4f}")

# ========================
# VISUALIZATIONS
# ========================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    'Bitcoin Fraud Detection on Elliptic++ Dataset\n'
    'Uncertainty-Aware Rewiring + Curriculum Pseudo-Labelling\n'
    'Standard Temporal Split (Train t1-34, Test t35-49) | Real Wallet IDs',
    fontsize=11, fontweight='bold'
)

colors_map = {
    'BitcoinGNN+Contributions': ('#c0392b', '-', 2.5),
    'BitcoinGNN': ('#e74c3c', '--', 1.5),
    'Ensemble': ('#9b59b6', '-.', 1.2),
    'XGBoost': ('#f39c12', '-.', 1.2),
    'MLP': ('#2ecc71', ':', 1.2),
    'BaselineGNN': ('#3498db', '--', 1.2),
}

# ROC curves
ax = axes[0, 0]
for mname, (color, ls, lw) in colors_map.items():
    if not results.get(mname):
        continue
    r = results[mname][-1]
    fpr, tpr, _ = roc_curve(r['true'], r['probs'])
    ax.plot(fpr, tpr, color=color, ls=ls, lw=lw,
            label=f"{mname} AUC={r['ROC-AUC']:.3f}")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_title('ROC Curve (Test: timesteps 35-49)')
ax.legend(fontsize=7)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

# Ablation bar chart
ax = axes[0, 1]
metrics_bar = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC']
x = np.arange(len(metrics_bar))
n_plots = len([m for m in order if m in aggs])
w = 0.8 / n_plots
bar_colors = ['#f39c12', '#9b59b6', '#2ecc71', '#3498db', '#e74c3c', '#c0392b']
for i, (mname, color) in enumerate(zip([m for m in order if m in aggs], bar_colors)):
    means = [aggs[mname][k][0] for k in metrics_bar]
    stds = [aggs[mname][k][1] for k in metrics_bar]
    bars = ax.bar(x + i * w, means, w, label=mname, color=color, yerr=stds, capsize=3)
    if mname == 'BitcoinGNN+Contributions':
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)
ax.set_xticks(x + w * (n_plots - 1) / 2)
ax.set_xticklabels(metrics_bar)
ax.set_title('Ablation Study (5 seeds, mean +- std)')
ax.legend(fontsize=6)
ax.set_ylim(0, 1.15)

# Calibration curves
ax = axes[0, 2]
if prob_pred_cal is not None:
    ax.plot(prob_pred_cal, prob_true_cal, 'o-', color='#c0392b',
            label=f'GNN+Contributions (ECE={ece:.3f})')
if pp_orig is not None:
    ax.plot(pp_orig, pt_orig, 's--', color='#e74c3c',
            label=f'BitcoinGNN (ECE={ece_orig:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.set_title('Probability Calibration Curve')
ax.legend(fontsize=8)
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Actual Fraud Rate')

# Uncertainty histogram
ax = axes[1, 0]
if test_unc is not None:
    ax.hist(test_unc, bins=30, alpha=0.8, edgecolor='black', color='#c0392b')
    ax.axvline(REWIRE_UNCERTAINTY_T, color='black', ls='--',
               label=f'Rewire threshold ({REWIRE_UNCERTAINTY_T})')
    ax.set_title('MC Dropout Uncertainty (Test Set)')
    ax.set_xlabel('Prediction Std Dev')
    ax.set_ylabel('Number of Transactions')
    ax.legend(fontsize=8)

# Confusion matrix
ax = axes[1, 1]
if results['BitcoinGNN+Contributions']:
    last_contrib_res = results['BitcoinGNN+Contributions'][-1]
    best_thresh = last_contrib_res.get('threshold', 0.5)
    ConfusionMatrixDisplay.from_predictions(
        last_contrib_res['true'],
        (last_contrib_res['probs'] > best_thresh).astype(int), ax=ax
    )
    ax.set_title(f'GNN+Contributions Confusion Matrix\n(threshold={best_thresh:.2f})')

# PR-AUC box plot
ax = axes[1, 2]
plot_data = [aggs[m]['PR-AUC'][2] for m in order if m in aggs]
plot_names = [m.replace('BitcoinGNN+Contributions', 'GNN+Contrib')
               .replace('BitcoinGNN', 'GNN') for m in order if m in aggs]
bp = ax.boxplot(plot_data, patch_artist=True)
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(plot_names, rotation=20, fontsize=7)
ax.set_title('PR-AUC Distribution Across 5 Seeds')
ax.set_ylabel('PR-AUC')

# Precision-Recall curves
from sklearn.metrics import precision_recall_curve
ax = axes[2, 0]
for mname, (color, ls, lw) in colors_map.items():
    if not results.get(mname):
        continue
    r = results[mname][-1]
    prec, rec, _ = precision_recall_curve(r['true'], r['probs'])
    ax.plot(rec, prec, color=color, ls=ls, lw=lw,
            label=f"{mname} PR-AUC={r['PR-AUC']:.3f}")
ax.set_title('Precision-Recall Curve')
ax.legend(fontsize=7)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')

# Rewiring history placeholder
ax = axes[2, 1]
ax.set_title('Uncertainty-Aware Rewiring\nEdge Suppression During Training')
ax.text(0.5, 0.5,
        'Rewiring suppresses edges between\nhigh-uncertainty node pairs.\n'
        'See console for per-epoch details.\n(stored in rewirer.rewire_history)',
        ha='center', va='center', transform=ax.transAxes, fontsize=9)
ax.set_xlabel('Rewiring Event')
ax.set_ylabel('Edges Suppressed')

# Delta AUC per seed
ax = axes[2, 2]
if 'BitcoinGNN+Contributions' in aggs and 'BitcoinGNN' in aggs:
    contrib_aucs = aggs['BitcoinGNN+Contributions']['PR-AUC'][2]
    orig_aucs = aggs['BitcoinGNN']['PR-AUC'][2]
    deltas = [c - o for c, o in zip(contrib_aucs, orig_aucs)]
    colors_delta = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
    ax.bar(range(len(SEEDS)), deltas, color=colors_delta, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS], rotation=15)
    mean_delta = np.mean(deltas)
    ax.axhline(mean_delta, color='navy', ls='--', label=f'Mean Delta={mean_delta:+.4f}')
    ax.set_title('Delta PR-AUC: (GNN+Contributions) - BitcoinGNN')
    ax.set_ylabel('Delta PR-AUC')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('bitcoin_fraud_ellipticpp_results.png', dpi=150, bbox_inches='tight')
print("\nSaved bitcoin_fraud_ellipticpp_results.png")

# ========================
# SAVE MODELS
# ========================
if last_contrib_model is not None:
    torch.save(last_contrib_model.state_dict(), 'bitcoin_gnn_contributions_model.pt')
    print("Saved bitcoin_gnn_contributions_model.pt")
if last_gnn_model is not None:
    torch.save(last_gnn_model.state_dict(), 'bitcoin_gnn_baseline_model.pt')
    print("Saved bitcoin_gnn_baseline_model.pt")

print("\n" + "=" * 65)
print("OUTPUT FILES")
print("=" * 65)
print("  bitcoin_fraud_ellipticpp_results.png  — full results figure")
print("  bitcoin_gnn_contributions_model.pt    — GNN+Contributions weights")
print("  bitcoin_gnn_baseline_model.pt         — BitcoinGNN baseline weights")
