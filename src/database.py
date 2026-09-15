"""
src/database.py
================
Lightweight persistence layer for the Bitcoin Fraud Detection pipeline.

Why this exists
----------------
Previously `src/onchain.py` fetched live transactions from the Blockstream
API and discarded them the moment the script exited. That meant velocity
features like `tx_count_1h` / `tx_count_24h` could never be true rolling
windows — with no history to look back on, they collapsed to "count of
transactions in the one block I just fetched" (see the old docstring in
onchain.py). Every prediction was also thrown away after being printed.

This module adds a small SQLite database (`outputs/onchain_history.db` by
default — one file, zero setup, no server) that:

  1. Stores every fetched on-chain transaction, keyed by txid, so repeated
     runs of `onchain.py` accumulate real history over time.
  2. Lets velocity features be computed from genuine multi-block rolling
     windows (real elapsed wall-clock time between block confirmations),
     not a single-block snapshot.
  3. Logs every prediction (txid, fraud probability, checkpoint used,
     timestamp) so past inference runs are auditable instead of being
     print-and-forget.

SQLite (not Postgres/MySQL/Neo4j) is the right fit here: this is a local
research/training pipeline, not a multi-user backend service. SQLite needs
no server process, ships in the Python standard library, handles the
data volumes here (hundreds of thousands of rows) comfortably, and the
whole DB is a single portable file. If this project ever grows into a
deployed service with concurrent writers, swap `DB_PATH` for a Postgres
connection string — the query shapes below would translate directly.
"""

import os
import sqlite3
import time
from contextlib import contextmanager

DEFAULT_DB_PATH = os.path.join("outputs", "onchain_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    txid              TEXT PRIMARY KEY,
    block_height      INTEGER,
    block_time        INTEGER,      -- unix timestamp of block confirmation
    num_inputs        INTEGER,
    num_outputs       INTEGER,
    input_value_sat   INTEGER,
    output_value_sat  INTEGER,
    fee_sat           INTEGER,
    size              INTEGER,
    weight            INTEGER,
    fetched_at        REAL          -- unix timestamp when WE fetched it
);

CREATE INDEX IF NOT EXISTS idx_tx_block_time ON transactions(block_time);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    txid              TEXT,
    fraud_probability REAL,
    checkpoint_path   TEXT,
    predicted_at      REAL
);

CREATE INDEX IF NOT EXISTS idx_pred_txid ON predictions(txid);
"""


@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    """Open a SQLite connection, ensuring the schema exists and the
    parent directory is created. Always use as a context manager so the
    connection is closed/committed even if an exception is raised."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_transactions(records: list, db_path: str = DEFAULT_DB_PATH, block_height: int = None):
    """Insert (or refresh) fetched on-chain transactions into the history
    table. `records` are the dicts produced by
    `onchain._extract_raw_fields`. Safe to call repeatedly — re-fetching
    the same txid just updates `fetched_at`."""
    now = time.time()
    rows = [
        (
            r["txid"], block_height, r.get("block_time"),
            r["num_inputs"], r["num_outputs"],
            r["input_value_sat"], r["output_value_sat"],
            r["fee_sat"], r["size"], r["weight"], now,
        )
        for r in records
    ]
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO transactions
                (txid, block_height, block_time, num_inputs, num_outputs,
                 input_value_sat, output_value_sat, fee_sat, size, weight, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(txid) DO UPDATE SET fetched_at = excluded.fetched_at
            """,
            rows,
        )
    return len(rows)


def get_history(before_time: int, window_seconds: int, db_path: str = DEFAULT_DB_PATH) -> list:
    """Return all persisted transactions with block_time in
    (before_time - window_seconds, before_time], oldest first. Used to
    compute genuine rolling-window velocity features across multiple
    fetch runs rather than a single block snapshot."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM transactions
            WHERE block_time IS NOT NULL
              AND block_time > ? AND block_time <= ?
            ORDER BY block_time ASC
            """,
            (before_time - window_seconds, before_time),
        )
        return [dict(row) for row in cur.fetchall()]


def log_predictions(txids: list, probs, checkpoint_path: str, db_path: str = DEFAULT_DB_PATH):
    """Persist a batch of predictions so inference runs are auditable."""
    now = time.time()
    rows = [(txid, float(p), checkpoint_path, now) for txid, p in zip(txids, probs)]
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO predictions (txid, fraud_probability, checkpoint_path, predicted_at) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Quick summary of what's stored — useful for a sanity check after
    a few runs of `python -m src.onchain`."""
    with get_connection(db_path) as conn:
        n_tx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        n_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        span = conn.execute(
            "SELECT MIN(block_time), MAX(block_time) FROM transactions"
        ).fetchone()
    return {
        "num_transactions": n_tx,
        "num_predictions": n_pred,
        "earliest_block_time": span[0],
        "latest_block_time": span[1],
    }
