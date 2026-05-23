"""
configs/config.py
=================
Centralised hyperparameter & path configuration for the
Bitcoin Fraud Detection project.
"""
import os

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 2024, 17, 99]

# ─── Model Architecture ───────────────────────────────────────────────────────
HIDDEN_DIM      = 256          # Wider hidden dim for better expressiveness
NUM_GAT_LAYERS  = 4            # Deeper than baseline (was 3)
NUM_HEADS       = 8            # More attention heads
DROPOUT         = 0.3
EDGE_DIM        = 16           # Edge feature projection dim (temporal edges)

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-4
EPOCHS          = 300
PATIENCE        = 30           # Larger patience for deeper model
LR_WARMUP_STEPS = 10
MC_SAMPLES      = 30           # Monte Carlo dropout passes

# ─── Focal Loss ───────────────────────────────────────────────────────────────
FOCAL_ALPHA     = 0.80         # Higher weight on minority (fraud) class
FOCAL_GAMMA     = 2.5

# ─── Self-supervised Pre-training ─────────────────────────────────────────────
PRETRAIN_EPOCHS = 50
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
ELLIPTIC_CACHE_PATH = os.environ.get(
    "ELLIPTIC_CACHE",
    r"C:\Users\Ria S\.cache\kagglehub\datasets\ellipticco\elliptic-data-set\versions\1"
)

# ─── Outputs ──────────────────────────────────────────────────────────────────
OUTPUT_DIR          = "outputs"
BEST_MODEL_PATH     = os.path.join(OUTPUT_DIR, "best_model.pt")
RESULTS_PLOT_PATH   = os.path.join(OUTPUT_DIR, "results.png")
METRICS_JSON_PATH   = os.path.join(OUTPUT_DIR, "metrics.json")
