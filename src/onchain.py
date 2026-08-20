"""
src/onchain.py
===============
Live on-chain inference: fetch REAL Bitcoin transactions from a public
blockchain API and run the trained ElliGAT model on them.

This is the "actually touches the blockchain" counterpart to the static
Elliptic dataset used for training (see src/data.py). It is a genuinely
different data source: no CSVs, no pre-computed labels — transactions are
pulled live from a public Bitcoin block explorer.

── IMPORTANT LIMITATION (read this before trusting the output) ────────────
The Elliptic dataset's 166 raw node features are a proprietary, undocumented
transformation of on-chain data — their exact definitions were never
released. We cannot reconstruct them from a public API. So for live
transactions we:
  1. Zero-impute the 166 "raw Elliptic feature" slots (unknown / unavailable)
  2. Compute the 7 "velocity" features for real (Amount_log, tx_count_1h,
     amount_sum_1h, Hour_sin, Hour_cos, tx_count_24h, amount_std_1h) —
     these ARE fully defined in src/data.py::add_velocity_features and can
     be computed from public transaction data.
  3. Build real payment-flow edges between fetched transactions (input →
     spent-output linkage), same structure as the Elliptic edgelist.

Because 166/173 feature dimensions are missing, predictions from this
module are NOT comparable to the benchmark numbers in the README — treat
this as a feasibility demo for live deployment, not a validated fraud
score. See README.md § "Live On-Chain Inference".

Data source: Blockstream Esplora public API (no API key required).
https://github.com/Blockstream/esplora/blob/master/API.md
"""

import argparse
import sys
import numpy as np
import torch

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from torch_geometric.data import Data

ESPLORA_BASE = "https://blockstream.info/api"

# Must match src/data.py::VELOCITY_COLS order exactly
VELOCITY_COLS = [
    "Amount_log",
    "tx_count_1h",
    "amount_sum_1h",
    "Hour_sin",
    "Hour_cos",
    "tx_count_24h",
    "amount_std_1h",
]
NUM_RAW_ELLIPTIC_FEATS = 166   # unknown / zero-imputed for live data
IN_DIM = NUM_RAW_ELLIPTIC_FEATS + len(VELOCITY_COLS)  # 173, matches src/models.py ElliGAT(in_dim=...)


# ─── Fetching ─────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 15):
    if not HAS_REQUESTS:
        raise ImportError(
            "The 'requests' package is required for live on-chain inference. "
            "Install it with: pip install requests"
        )
    resp = requests.get(f"{ESPLORA_BASE}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_latest_block_txids(limit: int = 200) -> tuple[list, dict]:
    """
    Fetch txids from the most recently confirmed block (not mempool, so the
    graph structure — inputs spending prior outputs within the same batch —
    is meaningful, same as Elliptic's within-dataset edges).

    Returns (txids, block_meta).
    """
    tip_hash = _get("/blocks/tip/hash")
    block_meta = _get(f"/block/{tip_hash}")

    txids = []
    start = 0
    while len(txids) < limit:
        chunk = _get(f"/block/{tip_hash}/txids")
        txids.extend(chunk)
        break  # Esplora returns the full txid list in one call
    return txids[:limit], block_meta


def fetch_tx(txid: str) -> dict:
    """Full transaction details: inputs (with prevout values), outputs, fee, size."""
    return _get(f"/tx/{txid}")


# ─── Feature engineering (mirrors src/data.py::add_velocity_features) ────

def _extract_raw_fields(tx: dict) -> dict:
    vin = tx.get("vin", [])
    vout = tx.get("vout", [])
    input_value = sum(v.get("prevout", {}).get("value", 0) or 0 for v in vin)
    output_value = sum(v.get("value", 0) or 0 for v in vout)
    fee = tx.get("fee", max(input_value - output_value, 0))
    return {
        "txid": tx["txid"],
        "num_inputs": len(vin),
        "num_outputs": len(vout),
        "input_value_sat": input_value,
        "output_value_sat": output_value,
        "fee_sat": fee,
        "size": tx.get("size", 0),
        "weight": tx.get("weight", 0),
        "block_time": tx.get("status", {}).get("block_time"),
        # prevout txids this tx spends from — used to build payment-flow edges
        "spends_txids": [v.get("txid") for v in vin if v.get("txid")],
    }


def build_velocity_features(records: list) -> np.ndarray:
    """
    Compute the 7 VELOCITY_COLS for real, from fetched on-chain fields.

    Within a single block, all transactions share one confirmation
    timestamp, so tx_count_1h / amount_sum_1h / amount_std_1h / tx_count_24h
    collapse to block-level aggregates rather than true rolling windows
    (that requires multi-block history, which src/data.py's Elliptic
    version has via 49 timesteps and this single-block demo does not).
    This is a known simplification — see module docstring.
    """
    n = len(records)
    amounts = np.array([r["output_value_sat"] / 1e8 for r in records], dtype=np.float64)  # BTC
    block_time = records[0]["block_time"] if records and records[0]["block_time"] else 0
    hour = (block_time // 3600) % 24 if block_time else 0

    amount_log = np.log1p(amounts)
    tx_count_1h = np.full(n, n, dtype=np.float64)          # all txs in this one block/window
    amount_sum_1h = np.full(n, amounts.sum(), dtype=np.float64)
    hour_sin = np.full(n, np.sin(2 * np.pi * hour / 24))
    hour_cos = np.full(n, np.cos(2 * np.pi * hour / 24))
    tx_count_24h = tx_count_1h.copy()                       # same limitation as above
    amount_std_1h = np.full(n, amounts.std(), dtype=np.float64)

    vel = np.stack(
        [amount_log, tx_count_1h, amount_sum_1h, hour_sin, hour_cos, tx_count_24h, amount_std_1h],
        axis=1,
    ).astype(np.float32)
    return vel


def build_live_graph(records: list) -> tuple[Data, list]:
    """
    Build a PyG Data object from fetched live transactions, matching the
    (N, 173) node-feature / (E, 1) edge-attr schema src/models.py expects.
    """
    n = len(records)
    txid_to_idx = {r["txid"]: i for i, r in enumerate(records)}

    raw_feats = np.zeros((n, NUM_RAW_ELLIPTIC_FEATS), dtype=np.float32)  # unknown → zero
    vel_feats = build_velocity_features(records)
    x = torch.tensor(np.concatenate([raw_feats, vel_feats], axis=1), dtype=torch.float)

    # Payment-flow edges: tx A -> tx B if B spends an output of A, and A is
    # also in our fetched batch (mirrors elliptic_txs_edgelist.csv).
    src, dst = [], []
    for r in records:
        b_idx = txid_to_idx[r["txid"]]
        for prev_txid in r["spends_txids"]:
            if prev_txid in txid_to_idx:
                src.append(txid_to_idx[prev_txid])
                dst.append(b_idx)

    if src:
        edge_index_directed = torch.tensor([src, dst], dtype=torch.long)
        # No true timestep delta available within one block -> 0 (same confirmation time)
        delta = torch.zeros((len(src), 1), dtype=torch.float)
        rev_edge_index = torch.stack([edge_index_directed[1], edge_index_directed[0]], dim=0)
        edge_index = torch.cat([edge_index_directed, rev_edge_index], dim=1)
        edge_attr = torch.cat([delta, delta], dim=0)
    else:
        # No linked pairs found in this batch -> empty edge set (model still runs)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, [r["txid"] for r in records]


# ─── Inference ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained ElliGAT model. Falls back to random init (with a loud
    warning) if no checkpoint is found, so the pipeline stays runnable for
    a structural smoke-test even without a trained model on disk."""
    from src.models import ElliGAT
    from configs.config import HIDDEN_DIM, NUM_GAT_LAYERS, NUM_HEADS, DROPOUT, EDGE_DIM

    model = ElliGAT(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GAT_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
    ).to(device)

    import os
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded trained weights from {checkpoint_path}")
    else:
        print(
            f"  [warn] No checkpoint found at {checkpoint_path} — "
            "using randomly-initialised weights. Run main.py first to train "
            "and save a real model. Predictions below are NOT meaningful "
            "until you do."
        )
    model.eval()
    return model


def run_live_inference(num_tx: int = 200, top_k: int = 15, checkpoint_path: str = "outputs/best_model.pt"):
    print(f"Fetching latest confirmed block (up to {num_tx} transactions)...")
    txids, block_meta = fetch_latest_block_txids(limit=num_tx)
    print(f"  Block height {block_meta.get('height')} | {len(txids)} txids fetched")

    records = []
    for i, txid in enumerate(txids):
        try:
            tx = fetch_tx(txid)
            records.append(_extract_raw_fields(tx))
        except Exception as exc:
            print(f"  [warn] failed to fetch {txid[:12]}...: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(txids)} transactions...")

    if not records:
        print("No transactions fetched — aborting.")
        return

    print(f"Building graph from {len(records)} live transactions...")
    data, ordered_txids = build_live_graph(records)
    print(f"  Nodes: {data.num_nodes} | Edges: {data.num_edges}")

    model = load_model(checkpoint_path)
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).cpu().numpy()

    order = np.argsort(-probs)[:top_k]
    print(f"\nTop {top_k} highest predicted fraud-probability transactions:")
    print(f"{'txid':<66} {'P(fraud)':>10}")
    for idx in order:
        print(f"{ordered_txids[idx]:<66} {probs[idx]:>10.4f}")

    return probs, ordered_txids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ElliGAT on live Bitcoin transactions.")
    parser.add_argument("--num-tx", type=int, default=200, help="Number of transactions to fetch from the latest block")
    parser.add_argument("--top-k", type=int, default=15, help="How many top-risk transactions to print")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt", help="Path to trained ElliGAT weights")
    args = parser.parse_args()

    if not HAS_REQUESTS:
        print("Missing dependency: pip install requests", file=sys.stderr)
        sys.exit(1)

    run_live_inference(num_tx=args.num_tx, top_k=args.top_k, checkpoint_path=args.checkpoint)
