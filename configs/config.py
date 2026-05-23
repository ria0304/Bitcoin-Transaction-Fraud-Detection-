"""
Configuration file for Bitcoin fraud detection
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "elliptic_bitcoin_dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
MODEL_CONFIG = {
    'input_dim': 165,  # Elliptic dataset has 165 features
    'hidden_dims': [256, 128, 64],
    'output_dim': 2,
    'dropout': 0.3
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10,
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_seed': 42
}

# Fraud detection specific
FRAUD_CONFIG = {
    'fraud_weight': 5.0,  # Weight for fraud class to handle imbalance
    'detection_threshold': 0.5
}
