"""Data splitting utilities for tweet classification.

Supports two splitting strategies:
1. split_by_hash: Random split by tweet_hash (prevents text leakage)
2. split_by_time: Temporal split by timestamp (train on early, test on late)
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
    train_hashes, temp_hashes = train_test_split(
        unique_hashes, test_size=temp_size, random_state=random_state
    )

    val_hashes, test_hashes = train_test_split(
        temp_hashes, test_size=(1 - val_of_temp), random_state=random_state
    )

    # Assign rows based on hash membership
    df_train = df[df["tweet_hash"].isin(train_hashes)].copy()
    df_val = df[df["tweet_hash"].isin(val_hashes)].copy()
    df_test = df[df["tweet_hash"].isin(test_hashes)].copy()

    return df_train, df_val, df_test


def get_split_summary(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> dict:
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


def verify_no_leakage(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> bool:
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
        raise ValueError(
            f"Leakage: {len(train_val_overlap)} hashes appear in both train and val"
        )

    if train_test_overlap:
        raise ValueError(
            f"Leakage: {len(train_test_overlap)} hashes appear in both train and test"
        )

    if val_test_overlap:
        raise ValueError(
            f"Leakage: {len(val_test_overlap)} hashes appear in both val and test"
        )

    return True


def split_by_time(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by timestamp for temporal validation.

    This split ensures that training data is from earlier dates and test data
    is from later dates, simulating real-world trading scenarios where we only
    have past data to train and predict future returns.

    Args:
        df: Input DataFrame with timestamp column.
        test_size: Fraction of data for test set (latest dates). Default: 0.15.
        val_size: Fraction of data for validation set (middle dates). Default: 0.15.
        timestamp_col: Name of timestamp column. Default: "timestamp".

    Returns:
        Tuple of (df_train, df_val, df_test) DataFrames sorted by time.
        - df_train: Earliest dates (70% by default)
        - df_val: Middle dates (15% by default)
        - df_test: Latest dates (15% by default)

    Raises:
        ValueError: If timestamp column is missing or not datetime.
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"DataFrame must have '{timestamp_col}' column for temporal splitting"
        )

    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort by timestamp
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

    # Calculate split indices
    n_samples = len(df_sorted)
    train_end_idx = int(n_samples * (1 - test_size - val_size))
    val_end_idx = int(n_samples * (1 - test_size))

    # Split
    df_train = df_sorted.iloc[:train_end_idx].copy()
    df_val = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    df_test = df_sorted.iloc[val_end_idx:].copy()

    return df_train, df_val, df_test


def get_temporal_split_summary(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> dict:
    """Generate summary of temporal data splits.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        timestamp_col: Name of timestamp column.

    Returns:
        Dictionary with split statistics including date ranges.
    """
    total = len(df_train) + len(df_val) + len(df_test)

    def get_date_range(df: pd.DataFrame) -> dict:
        if len(df) == 0:
            return {"start": None, "end": None}
        return {
            "start": str(df[timestamp_col].min()),
            "end": str(df[timestamp_col].max()),
        }

    summary = {
        "train": {
            "samples": len(df_train),
            "percentage": 100 * len(df_train) / total,
            "date_range": get_date_range(df_train),
        },
        "val": {
            "samples": len(df_val),
            "percentage": 100 * len(df_val) / total,
            "date_range": get_date_range(df_val),
        },
        "test": {
            "samples": len(df_test),
            "percentage": 100 * len(df_test) / total,
            "date_range": get_date_range(df_test),
        },
        "total": total,
    }

    return summary
