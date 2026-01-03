"""Upload dataset_2025_full to HuggingFace Hub.

This script loads the financial tweets dataset, creates temporal train/val/test
splits, and uploads to HuggingFace with proper versioning and documentation.
"""

import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value


# Configuration
REPO_ID = "michael-zus/fintweet-sentiment-2025"
VERSION = "1.0.0"
CSV_PATH = "output/dataset_2025_full.csv"

# Columns to include in the public dataset (exclude internal/leakage columns)
COLUMNS_TO_KEEP = [
    # Metadata
    "timestamp",
    "ticker",
    "tweet_url",
    # Target
    "label_1d_3class",
    # Categorical features
    "author",
    "category",
    "session",
    "market_regime",
    "sector",
    "market_cap_bucket",
    # Numerical features
    "volatility_7d",
    "relative_volume",
    "rsi_14",
    "distance_from_ma_20",
    "return_5d",
    "return_20d",
    "above_ma_20",
    "slope_ma_20",
    "gap_open",
    "intraday_range",
    # Text feature
    "text",
]


def get_features() -> Features:
    """Define the HuggingFace Features schema with proper types."""
    return Features(
        {
            "timestamp": Value("string"),
            "ticker": Value("string"),
            "tweet_url": Value("string"),
            "label_1d_3class": ClassLabel(names=["BUY", "HOLD", "SELL"]),
            "author": Value("string"),
            "category": Value("string"),
            "session": Value("string"),
            "market_regime": Value("string"),
            "sector": Value("string"),
            "market_cap_bucket": Value("string"),
            "volatility_7d": Value("float32"),
            "relative_volume": Value("float32"),
            "rsi_14": Value("float32"),
            "distance_from_ma_20": Value("float32"),
            "return_5d": Value("float32"),
            "return_20d": Value("float32"),
            "above_ma_20": Value("float32"),
            "slope_ma_20": Value("float32"),
            "gap_open": Value("float32"),
            "intraday_range": Value("float32"),
            "text": Value("string"),
        }
    )


def load_and_split_dataset(csv_path: str) -> DatasetDict:
    """Load CSV and create temporal train/val/test splits.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        DatasetDict with train, validation, and test splits.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df):,}")

    # Sort by timestamp for temporal split
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Select only columns to keep
    df = df[COLUMNS_TO_KEEP].copy()

    # 70/15/15 temporal split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df):,} | Validation: {len(val_df):,} | Test: {len(test_df):,}")

    # Get features schema
    features = get_features()

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, features=features, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, features=features, preserve_index=False),
            "test": Dataset.from_pandas(test_df, features=features, preserve_index=False),
        }
    )


def print_class_distribution(dataset_dict: DatasetDict) -> None:
    """Print class distribution for each split."""
    label_names = ["BUY", "HOLD", "SELL"]

    for split_name, split_data in dataset_dict.items():
        print(f"\n{split_name} class distribution:")
        labels = split_data["label_1d_3class"]
        total = len(labels)
        for i, name in enumerate(label_names):
            count = labels.count(i)
            pct = count / total * 100
            print(f"  {name}: {count:,} ({pct:.1f}%)")


def main() -> None:
    """Upload dataset to HuggingFace Hub."""
    # Load and prepare dataset
    dataset_dict = load_and_split_dataset(CSV_PATH)

    # Show class distribution
    print_class_distribution(dataset_dict)

    # Push to HuggingFace Hub
    print(f"\nPushing to HuggingFace Hub: {REPO_ID}...")
    dataset_dict.push_to_hub(
        REPO_ID,
        commit_message=f"Upload dataset v{VERSION}",
        private=False,
    )

    print(f"\n✅ Dataset uploaded successfully!")
    print(f"   View at: https://huggingface.co/datasets/{REPO_ID}")
    print(f"\nNext steps:")
    print(f"   1. Upload the dataset card (README.md) to the repository")
    print(f"   2. Create a version tag: huggingface-cli tag {REPO_ID} v{VERSION} --repo-type dataset")


if __name__ == "__main__":
    main()

