"""Tweet Classifier module for FinBERT-based price prediction.

This module provides utilities for:
- Loading and filtering enriched tweet data
- Splitting data for training/validation/testing
- Computing class weights for imbalanced data
- PyTorch Dataset for multi-modal training
- Preprocessing artifact persistence (scaler, encodings)
- FinBERTMultiModal model for classification
"""

from tweet_classifier.config import (
    CATEGORICAL_FEATURES,
    LABEL_MAP,
    LABEL_MAP_INV,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.dataset import (
    TweetDataset,
    create_categorical_encodings,
    create_dataset_from_df,
    encode_categorical,
    load_categorical_encodings,
    load_preprocessing_artifacts,
    load_scaler,
    save_categorical_encodings,
    save_preprocessing_artifacts,
    save_scaler,
)
from tweet_classifier.model import FinBERTMultiModal

__all__ = [
    # Config
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "LABEL_MAP",
    "LABEL_MAP_INV",
    # Dataset
    "TweetDataset",
    "create_categorical_encodings",
    "create_dataset_from_df",
    "encode_categorical",
    # Model
    "FinBERTMultiModal",
    # Persistence
    "save_scaler",
    "load_scaler",
    "save_categorical_encodings",
    "load_categorical_encodings",
    "save_preprocessing_artifacts",
    "load_preprocessing_artifacts",
]

