"""Configuration constants for tweet classification.

This module defines global constants used throughout the tweet_classifier package.
These values should NOT be redefined elsewhere - import them from here.
"""

from pathlib import Path

# ============================================================
# Target Configuration
# ============================================================
TARGET_COLUMN = "label_1d_3class"  # 1-day labels (less noisy than 1-hour)

# Label mapping (string to integer)
LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}
LABEL_MAP_INV = {0: "SELL", 1: "HOLD", 2: "BUY"}

NUM_CLASSES = 3

# ============================================================
# Feature Configuration
# ============================================================
# Numerical features (safe - computed BEFORE tweet timestamp)
NUMERICAL_FEATURES = [
    "volatility_7d",
    "relative_volume",
    "rsi_14",
    "distance_from_ma_20",
]

# Categorical features (will be embedded)
CATEGORICAL_FEATURES = ["author", "category"]

# Columns explicitly EXCLUDED from features (future-looking / targets / reference)
EXCLUDED_FROM_FEATURES = [
    "spy_return_1d",  # Uses day T close (future for intraday tweets!)
    "spy_return_1hr",  # Future SPY movement
    "return_1hr",  # Target
    "return_to_next_open",  # Target
    "price_1hr_after",  # Future price
    "price_next_open",  # Future price
    "label_3class",  # Target (backup)
    "label_1d_3class",  # Target (primary)
]

# Text column
TEXT_COLUMN = "text"

# ============================================================
# Data Paths
# ============================================================
# Path: src/tweet_classifier/config.py -> up 3 levels to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "output" / "15-dec-enrich7.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "finbert-tweet-classifier"

# ============================================================
# Model Configuration
# ============================================================
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_TEXT_LENGTH = 128

# Embedding dimensions
AUTHOR_EMBEDDING_DIM = 16
CATEGORY_EMBEDDING_DIM = 8
NUMERICAL_HIDDEN_DIM = 32

# ============================================================
# Training Defaults
# ============================================================
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 5
DEFAULT_DROPOUT = 0.3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01

# ============================================================
# Data Split Configuration
# ============================================================
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
RANDOM_SEED = 42

