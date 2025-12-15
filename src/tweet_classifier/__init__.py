"""Tweet Classifier module for FinBERT-based price prediction.

This module provides utilities for:
- Loading and filtering enriched tweet data
- Splitting data for training/validation/testing
- Computing class weights for imbalanced data
- PyTorch Dataset for multi-modal training
- Preprocessing artifact persistence (scaler, encodings)
- FinBERTMultiModal model for classification
- Training with class-weighted loss
- Comprehensive evaluation with trading metrics
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
from tweet_classifier.evaluate import (
    compute_baselines,
    compute_trading_metrics,
    evaluate,
    evaluate_on_test,
    generate_classification_report,
    plot_confusion_matrix,
    run_full_evaluation,
)
from tweet_classifier.model import FinBERTMultiModal
from tweet_classifier.train import create_training_args, train
from tweet_classifier.trainer import WeightedTrainer, compute_metrics

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
    # Training
    "WeightedTrainer",
    "compute_metrics",
    "create_training_args",
    "train",
    # Evaluation
    "evaluate",
    "evaluate_on_test",
    "compute_trading_metrics",
    "compute_baselines",
    "generate_classification_report",
    "plot_confusion_matrix",
    "run_full_evaluation",
    # Persistence
    "save_scaler",
    "load_scaler",
    "save_categorical_encodings",
    "load_categorical_encodings",
    "save_preprocessing_artifacts",
    "load_preprocessing_artifacts",
]

