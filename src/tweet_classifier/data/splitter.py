"""Data splitting utilities for tweet classification.

Splits data by tweet_hash to prevent text leakage across train/val/test sets.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from tweet_classifier.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, RANDOM_SEED


def split_by_hash(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by tweet_hash to prevent text leakage.

    This ensures that the same tweet text never appears in both training
    and validation/test sets, which would cause data leakage.

    Args:
        df: Input DataFrame with tweet_hash column.
        test_size: Fraction of data for test set (default: 0.15).
        val_size: Fraction of data for validation set (default: 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (df_train, df_val, df_test) DataFrames.

    Raises:
        ValueError: If tweet_hash column is missing.
    """
    if "tweet_hash" not in df.columns:
        raise ValueError("DataFrame must have 'tweet_hash' column for splitting")

    # Get unique tweet hashes
    unique_hashes = df["tweet_hash"].unique()

    # Calculate actual split sizes
    # test_size and val_size are fractions of total data
    # For sklearn, we need to split in two steps
    temp_size = test_size + val_size  # First split: train vs (val + test)
    val_of_temp = val_size / temp_size  # Second split: val vs test from temp

    # Split hashes (not rows!) to prevent text leakage
    train_hashes, temp_hashes = train_test_split(unique_hashes, test_size=temp_size, random_state=random_state)

    val_hashes, test_hashes = train_test_split(temp_hashes, test_size=(1 - val_of_temp), random_state=random_state)

    # Assign rows based on hash membership
    df_train = df[df["tweet_hash"].isin(train_hashes)].copy()
    df_val = df[df["tweet_hash"].isin(val_hashes)].copy()
    df_test = df[df["tweet_hash"].isin(test_hashes)].copy()

    return df_train, df_val, df_test


def get_split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Generate summary of data splits.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.

    Returns:
        Dictionary with split statistics.
    """
    total = len(df_train) + len(df_val) + len(df_test)

    summary = {
        "train": {
            "samples": len(df_train),
            "percentage": 100 * len(df_train) / total,
            "unique_hashes": df_train["tweet_hash"].nunique(),
        },
        "val": {
            "samples": len(df_val),
            "percentage": 100 * len(df_val) / total,
            "unique_hashes": df_val["tweet_hash"].nunique(),
        },
        "test": {
            "samples": len(df_test),
            "percentage": 100 * len(df_test) / total,
            "unique_hashes": df_test["tweet_hash"].nunique(),
        },
        "total": total,
    }

    return summary


def verify_no_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> bool:
    """Verify that no tweet hashes appear in multiple splits.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.

    Returns:
        True if no leakage detected, False otherwise.

    Raises:
        ValueError: If leakage is detected (optional - can be changed to just return False).
    """
    train_hashes = set(df_train["tweet_hash"].unique())
    val_hashes = set(df_val["tweet_hash"].unique())
    test_hashes = set(df_test["tweet_hash"].unique())

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    if train_val_overlap:
        raise ValueError(f"Leakage: {len(train_val_overlap)} hashes appear in both train and val")

    if train_test_overlap:
        raise ValueError(f"Leakage: {len(train_test_overlap)} hashes appear in both train and test")

    if val_test_overlap:
        raise ValueError(f"Leakage: {len(val_test_overlap)} hashes appear in both val and test")

    return True

