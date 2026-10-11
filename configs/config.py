"""
configs/config.py
=================
Centralised hyperparameter & path configuration for the
Bitcoin Fraud Detection project.
"""
import os


# ─── Env-override helpers (empty/unset env var -> default) ────────────────────
def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw.strip())


def _env_int_list(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


# ─── Reproducibility ──────────────────────────────────────────────────────────
# SEEDS env var: comma-separated, e.g. SEEDS=42,123,2024
SEEDS = _env_int_list("SEEDS", [42, 123, 2024, 17, 99])

# ─── Model Architecture ───────────────────────────────────────────────────────
HIDDEN_DIM      = _env("HIDDEN_DIM", 256, int)          # Wider hidden dim for better expressiveness
NUM_GAT_LAYERS  = _env("NUM_GAT_LAYERS", 4, int)            # Deeper than baseline (was 3)
NUM_HEADS       = _env("NUM_HEADS", 8, int)            # More attention heads
DROPOUT         = _env("DROPOUT", 0.3, float)
EDGE_DIM        = 16           # Edge feature projection dim (temporal edges)

# Temporal edge encoding on |Δtimestep|:
#   "sinusoidal" — fixed multi-scale sin/cos basis + learned projection (default)
#   "mlp"        — legacy Linear(1→d)→GELU→Linear(d→d); reproduces the verified
#                  5-seed results reported in the README
EDGE_ENCODING   = os.environ.get("EDGE_ENCODING", "sinusoidal")
# Largest wavelength in the sinusoidal basis. Elliptic deltas span [0, 48],
# so 100 keeps every channel informative; 10000 (the NLP default) would make
# the upper channels near-constant over this range.
EDGE_MAX_PERIOD = float(os.environ.get("EDGE_MAX_PERIOD", 100.0))

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = _env("LEARNING_RATE", 5e-4, float)
WEIGHT_DECAY    = _env("WEIGHT_DECAY", 1e-4, float)
EPOCHS          = _env("EPOCHS", 300, int)
PATIENCE        = _env("PATIENCE", 30, int)           # Larger patience for deeper model
LR_WARMUP_STEPS = 10
MC_SAMPLES      = _env("MC_SAMPLES", 30, int)           # Monte Carlo dropout passes

# ─── Focal Loss ───────────────────────────────────────────────────────────────
FOCAL_ALPHA     = 0.80         # Higher weight on minority (fraud) class
FOCAL_GAMMA     = 2.5

# ─── Self-supervised Pre-training ─────────────────────────────────────────────
PRETRAIN_EPOCHS = _env("PRETRAIN_EPOCHS", 50, int)
MASK_RATIO      = 0.20         # Fraction of nodes masked for feature reconstruction

# ─── Contrastive Learning ─────────────────────────────────────────────────────
CONTRASTIVE_TEMP    = 0.07
CONTRASTIVE_WEIGHT  = 0.30     # λ for contrastive loss in combined objective

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET         = "elliptic"

# Update these paths to match your local setup
PROJECT_DATA_PATH = os.environ.get(
    "BITCOIN_DATA_PATH",
    r"C:\Users\Ria S\OneDrive\Attachments\Desktop\projects\BITCOIN_FRAUD_DETECTION\data"
)
# Accept both ELLIPTIC_CACHE and ELLIPTIC_CACHE_PATH as the env var name —
# the two names have been used interchangeably in docs/instructions, so
# support both rather than silently ignoring one of them.
ELLIPTIC_CACHE_PATH = (
    os.environ.get("ELLIPTIC_CACHE_PATH")
    or os.environ.get("ELLIPTIC_CACHE")
    or r"C:\Users\Ria S\.cache\kagglehub\datasets\ellipticco\elliptic-data-set\versions\1"
)

# ─── Novelty: loss selection (focal | asymmetric) ────────────────────────────
LOSS_NAME = os.environ.get("LOSS_NAME", "asymmetric")
ASL_GAMMA_POS = float(os.environ.get("ASL_GAMMA_POS", 0.0))
ASL_GAMMA_NEG = float(os.environ.get("ASL_GAMMA_NEG", 4.0))
ASL_CLIP      = float(os.environ.get("ASL_CLIP", 0.05))

# ─── Post-hoc stages (PLP + GNNExplainer); env-overridable so CI smoke runs are fast ─
PLP_EPOCHS      = _env("PLP_EPOCHS", 50, int)
EXPLAIN_NODES   = _env("EXPLAIN_NODES", 30, int)
EXPLAIN_EPOCHS  = _env("EXPLAIN_EPOCHS", 150, int)

# ─── Outputs ──────────────────────────────────────────────────────────────────
OUTPUT_DIR          = os.environ.get("OUTPUT_DIR", "outputs")
BEST_MODEL_PATH     = os.path.join(OUTPUT_DIR, "best_model.pt")
RESULTS_PLOT_PATH   = os.path.join(OUTPUT_DIR, "results.png")
METRICS_JSON_PATH   = os.path.join(OUTPUT_DIR, "metrics.json")

# ─── Logging ────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(OUTPUT_DIR, "run_log.txt")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
