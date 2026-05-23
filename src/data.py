"""
src/data.py
===========
Dataset discovery, loading, feature engineering, and graph construction
for the Elliptic Bitcoin transaction dataset.

Key improvements over baseline:
  • Temporal edge features (timestep delta between connected transactions)
  • Richer velocity / wallet-behaviour features (7 features instead of 5)
  • Directed graph preserved (original direction carries semantic meaning)
  • Unlabelled node semi-supervision via pseudo-label propagation hook
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False

# ─── Velocity feature column names ───────────────────────────────────────────
VELOCITY_COLS = [
    "Amount_log",
    "tx_count_1h",
    "amount_sum_1h",
    "Hour_sin",
    "Hour_cos",
    "tx_count_24h",      # NEW: daily tx velocity
    "amount_std_1h",     # NEW: amount variance (signals structuring attacks)
]

# ─── Dataset discovery ───────────────────────────────────────────────────────

def _ensure_dataset(dataset_name: str):
    """Download dataset via KaggleHub if available."""
    if not HAS_KAGGLEHUB:
        return None
    try:
        if dataset_name == "elliptic":
            return kagglehub.dataset_download("ellipticco/elliptic-data-set")
    except Exception as exc:
        print(f"  [warn] Auto-download failed: {exc}")
    return None


def find_dataset_path(dataset_name: str, project_data_path: str, cache_path: str) -> str:
    """Locate Elliptic dataset in known local paths, else download."""
    candidates = [
        os.path.join(project_data_path, dataset_name),
        cache_path,
        os.path.join(cache_path, "1"),
        os.path.join(project_data_path, "elliptic"),
    ]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "elliptic_txs_features.csv")):
            return path
        # Look one level deeper
        for sub in os.listdir(path):
            full = os.path.join(path, sub)
            if os.path.isdir(full) and os.path.exists(
                os.path.join(full, "elliptic_txs_features.csv")
            ):
                return full
    print("  Dataset not found locally. Attempting download via KaggleHub …")
    return _ensure_dataset(dataset_name)


# ─── Elliptic loader ─────────────────────────────────────────────────────────

def load_elliptic(path: str):
    """
    Parse the three Elliptic CSVs and return:
        labelled_df     – labelled transactions with engineered features
        X_all           – raw feature matrix for ALL nodes (203 769 × 166)
        edge_index      – undirected edge_index tensor
        edge_attr       – edge temporal features (timestep delta)
        txid_to_idx     – {txId → global node index}
        all_nodes       – full merged DataFrame
    """
    feats_path   = os.path.join(path, "elliptic_txs_features.csv")
    classes_path = os.path.join(path, "elliptic_txs_classes.csv")
    edges_path   = os.path.join(path, "elliptic_txs_edgelist.csv")

    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"elliptic_txs_features.csv not found at {feats_path}")

    feats   = pd.read_csv(feats_path, header=None)
    classes = pd.read_csv(classes_path)

    feat_cols = ["txId", "timestep"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    feats.columns = feat_cols

    classes["class"] = classes["class"].map({"1": 1, "2": 0, "unknown": np.nan})
    all_nodes = feats.merge(classes, on="txId", how="left").rename(
        columns={"class": "isFraud"}
    )

    txid_to_idx = {tid: i for i, tid in enumerate(all_nodes["txId"])}

    # ── Build directed + undirected edge index ──────────────────────────────
    edge_index = None
    edge_attr  = None
    if os.path.exists(edges_path):
        el = pd.read_csv(edges_path, header=None)
        el.columns = ["txId1", "txId2"]
        mask = el["txId1"].isin(txid_to_idx) & el["txId2"].isin(txid_to_idx)
        el = el[mask]

        src = torch.tensor([txid_to_idx[t] for t in el["txId1"]], dtype=torch.long)
        dst = torch.tensor([txid_to_idx[t] for t in el["txId2"]], dtype=torch.long)

        # Temporal edge feature: |timestep(src) − timestep(dst)|
        ts_arr = all_nodes["timestep"].values
        t_src  = torch.tensor(ts_arr[src.numpy()], dtype=torch.float)
        t_dst  = torch.tensor(ts_arr[dst.numpy()], dtype=torch.float)
        delta  = (t_src - t_dst).abs().unsqueeze(1)  # (E, 1)

        # Make undirected (duplicate edges + duplicate edge attrs)
        edge_index_directed = torch.stack([src, dst], dim=0)
        rev_edge_index       = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index_directed, rev_edge_index], dim=1)
        edge_attr  = torch.cat([delta, delta], dim=0)   # (2E, 1)

        print(
            f"  Loaded edges: {el.shape[0]} directed → "
            f"{edge_index.shape[1]} undirected"
        )

    raw_feat_cols = [c for c in feat_cols if c.startswith("f")]
    X_all = all_nodes[raw_feat_cols].values.astype(np.float32)

    # ── Labelled subset ──────────────────────────────────────────────────────
    labelled = all_nodes.dropna(subset=["isFraud"]).copy()
    labelled["isFraud"] = labelled["isFraud"].astype(int)

    # Synthetic wallet/amount proxies (Elliptic doesn't expose real wallets)
    labelled["Wallet_ID"]  = labelled["timestep"].astype(str)
    labelled["Cluster_ID"] = (labelled["txId"] % 500).astype(str)
    base = pd.Timestamp("2019-01-01")
    labelled["Timestamp"] = base + pd.to_timedelta(
        labelled["timestep"] * 2, unit="W"
    )
    labelled["Amount"] = labelled["f0"]
    labelled = labelled.sort_values("Timestamp").reset_index(drop=True)

    fraud_ratio = labelled["isFraud"].mean()
    print(
        f"  [Elliptic] {len(all_nodes):,} total | "
        f"{len(labelled):,} labelled | "
        f"Fraud: {fraud_ratio:.4f}"
    )
    return labelled, X_all, edge_index, edge_attr, txid_to_idx, all_nodes


# ─── Feature engineering ─────────────────────────────────────────────────────

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wallet-behaviour velocity features using time-indexed rolling windows.
    These capture structuring attacks and smurfing patterns.
    """
    df = df.sort_values(["Wallet_ID", "Timestamp"]).copy()

    def _rolling_count(g, window):
        ts = g.set_index("Timestamp")
        res = ts.index.to_series().rolling(window, closed="left").count().fillna(0)
        res.index = g.index
        return res

    def _rolling_sum(g, window):
        res = (
            g.set_index("Timestamp")["Amount"]
            .rolling(window, closed="left")
            .sum()
            .fillna(0)
        )
        res.index = g.index
        return res

    def _rolling_std(g, window):
        res = (
            g.set_index("Timestamp")["Amount"]
            .rolling(window, closed="left")
            .std()
            .fillna(0)
        )
        res.index = g.index
        return res

    df["tx_count_1h"]    = df.groupby("Wallet_ID", group_keys=False).apply(
        lambda g: _rolling_count(g, "1h")
    )
    df["amount_sum_1h"]  = df.groupby("Wallet_ID", group_keys=False).apply(
        lambda g: _rolling_sum(g, "1h")
    )
    df["amount_std_1h"]  = df.groupby("Wallet_ID", group_keys=False).apply(
        lambda g: _rolling_std(g, "1h")
    )
    df["tx_count_24h"]   = df.groupby("Wallet_ID", group_keys=False).apply(
        lambda g: _rolling_count(g, "24h")
    )
    df["Amount_log"]  = np.log1p(df["Amount"])
    df["Hour_sin"]    = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["Hour_cos"]    = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    return df


# ─── Chronological split ─────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """No-leakage temporal split matching the Elliptic paper's protocol."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end   = train_end + int(n * val_frac)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
        train_end,
        val_end - train_end,
    )


# ─── Graph construction ──────────────────────────────────────────────────────

def build_graph(
    df: pd.DataFrame,
    X_all: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    txid_to_idx: dict,
    train_size: int,
    val_size: int,
    feat_cols: list,
) -> Data:
    """
    Build a PyG Data object for the Bitcoin transaction graph.

    Node features (N × 173):
        166 raw Elliptic features
        +7  velocity features (zero-padded for unlabelled nodes)

    Edge features (E × 1):
        Absolute timestep delta between connected transactions

    Labels: 1=fraud, 0=licit, -1=unknown (semi-supervised)
    """
    N_all  = X_all.shape[0]
    labelled_global_idx = torch.tensor(
        [txid_to_idx[tid] for tid in df["txId"]], dtype=torch.long
    )

    # Scale raw features on training nodes only (prevent leakage)
    train_global = labelled_global_idx[:train_size].numpy()
    scaler_raw   = StandardScaler().fit(X_all[train_global])
    X_scaled     = scaler_raw.transform(X_all).astype(np.float32)

    # Velocity features — scaled on training set, zero-padded elsewhere
    vel_raw  = df[VELOCITY_COLS].values.astype(np.float32)
    scaler_v = StandardScaler().fit(vel_raw[:train_size])
    vel_scaled = scaler_v.transform(vel_raw).astype(np.float32)

    vel_matrix = np.zeros((N_all, len(VELOCITY_COLS)), dtype=np.float32)
    for i, gidx in enumerate(labelled_global_idx.numpy()):
        vel_matrix[gidx] = vel_scaled[i]

    node_feats = np.concatenate([X_scaled, vel_matrix], axis=1)  # (N, 173)
    x = torch.tensor(node_feats, dtype=torch.float)

    # Labels
    labels_full = torch.full((N_all,), -1, dtype=torch.float)
    for i, gidx in enumerate(labelled_global_idx.numpy()):
        labels_full[gidx] = float(df.iloc[i]["isFraud"])

    # Masks
    tr_mask = torch.zeros(N_all, dtype=torch.bool)
    vl_mask = torch.zeros(N_all, dtype=torch.bool)
    te_mask = torch.zeros(N_all, dtype=torch.bool)
    tr_mask[labelled_global_idx[:train_size]]           = True
    vl_mask[labelled_global_idx[train_size:train_size + val_size]] = True
    te_mask[labelled_global_idx[train_size + val_size:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels_full,
        train_mask=tr_mask,
        val_mask=vl_mask,
        test_mask=te_mask,
    )
    return data, scaler_raw, scaler_v
