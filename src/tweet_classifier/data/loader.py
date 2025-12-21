"""Data loading and filtering utilities for tweet classification."""

from pathlib import Path
from typing import Optional

import pandas as pd

from tweet_classifier.config import (
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_FEATURES,
    TEXT_COLUMN,
    DEFAULT_DATA_PATH,
)


def load_enriched_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load enriched tweet data from CSV.

    Args:
        path: Path to CSV file. If None, uses DEFAULT_DATA_PATH.

    Returns:
        DataFrame with enriched tweet data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        path = DEFAULT_DATA_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    required_cols = [TARGET_COLUMN, TEXT_COLUMN, "tweet_hash", "is_reliable_label"]
    required_cols.extend(NUMERICAL_FEATURES)
    required_cols.extend(CATEGORICAL_FEATURES)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def filter_reliable(df: pd.DataFrame, drop_missing_target: bool = True) -> pd.DataFrame:
    """Filter DataFrame to keep only reliable samples.

    Args:
        df: Input DataFrame with is_reliable_label column.
        drop_missing_target: If True, also drop rows with missing TARGET_COLUMN.

    Returns:
        Filtered DataFrame with only reliable samples.
    """
    # Filter to reliable labels
    df_filtered = df[df["is_reliable_label"] == True].copy()  # noqa: E712

    # Optionally drop missing targets
    if drop_missing_target:
        df_filtered = df_filtered.dropna(subset=[TARGET_COLUMN])

    return df_filtered


def prepare_features(df: pd.DataFrame) -> dict:
    """Extract and validate features from DataFrame.

    Args:
        df: Input DataFrame with all required columns.

    Returns:
        Dictionary with:
            - 'text': Series of tweet texts
            - 'numerical': DataFrame of numerical features
            - 'categorical': Dictionary of categorical columns
            - 'labels': Series of string labels
            - 'tweet_hash': Series of tweet hashes

    Raises:
        ValueError: If any forbidden columns are in NUMERICAL_FEATURES.
    """
    # Safety check: ensure no forbidden columns in features
    for col in EXCLUDED_FROM_FEATURES:
        if col in NUMERICAL_FEATURES:
            raise ValueError(
                f"LEAK: {col} found in NUMERICAL_FEATURES! This is future data."
            )

    # Extract features
    features = {
        "text": df[TEXT_COLUMN],
        "numerical": df[NUMERICAL_FEATURES].fillna(0),
        "categorical": {col: df[col] for col in CATEGORICAL_FEATURES},
        "labels": df[TARGET_COLUMN],
        "tweet_hash": df["tweet_hash"],
    }

    return features


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_samples": len(df),
        "reliable_samples": (
            df["is_reliable_label"].sum()
            if "is_reliable_label" in df.columns
            else len(df)
        ),
        "target_distribution": (
            df[TARGET_COLUMN].value_counts().to_dict()
            if TARGET_COLUMN in df.columns
            else {}
        ),
        "missing_target": (
            df[TARGET_COLUMN].isna().sum() if TARGET_COLUMN in df.columns else 0
        ),
        "unique_authors": df["author"].nunique() if "author" in df.columns else 0,
        "unique_tickers": df["ticker"].nunique() if "ticker" in df.columns else 0,
    }

    return summary
