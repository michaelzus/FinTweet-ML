"""PyTorch Dataset for multi-modal tweet classification."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tweet_classifier.config import (
    LABEL_MAP,
    MAX_TEXT_LENGTH,
    NUMERICAL_FEATURES,
    TEXT_COLUMN,
)


class TweetDataset(Dataset):
    """PyTorch Dataset for tweet classification with multi-modal features.

    Handles:
    - Text tokenization (using provided tokenizer)
    - Numerical feature normalization
    - Categorical feature encoding (author, category)
    - Label encoding
    """

    def __init__(
        self,
        texts: Union[pd.Series, List[str]],
        numerical_features: Union[pd.DataFrame, np.ndarray],
        author_indices: Union[pd.Series, np.ndarray],
        category_indices: Union[pd.Series, np.ndarray],
        labels: Union[pd.Series, np.ndarray],
        tokenizer,
        max_length: int = MAX_TEXT_LENGTH,
    ):
        """Initialize the dataset.

        Args:
            texts: Series or list of tweet texts.
            numerical_features: DataFrame or array of numerical features.
            author_indices: Series or array of author indices (integers).
            category_indices: Series or array of category indices (integers).
            labels: Series or array of labels (strings or integers).
            tokenizer: HuggingFace tokenizer for text encoding.
            max_length: Maximum token length for text (default: 128).
        """
        # Convert texts to list
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Tokenize all texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # Convert numerical features
        if isinstance(numerical_features, pd.DataFrame):
            numerical_features = numerical_features.values
        self.numerical = torch.tensor(numerical_features, dtype=torch.float32)

        # Convert categorical indices
        if isinstance(author_indices, pd.Series):
            author_indices = author_indices.values
        if isinstance(category_indices, pd.Series):
            category_indices = category_indices.values
        self.author_idx = torch.tensor(author_indices, dtype=torch.long)
        self.category_idx = torch.tensor(category_indices, dtype=torch.long)

        # Convert labels
        if isinstance(labels, pd.Series):
            labels = labels.values
        if isinstance(labels[0], str):
            labels = np.array([LABEL_MAP[label] for label in labels])
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with:
                - input_ids: Token IDs for text
                - attention_mask: Attention mask for text
                - numerical: Numerical features tensor
                - author_idx: Author index
                - category_idx: Category index
                - labels: Label (integer)
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "numerical": self.numerical[idx],
            "author_idx": self.author_idx[idx],
            "category_idx": self.category_idx[idx],
            "labels": self.labels[idx],
        }


def create_categorical_encodings(df: pd.DataFrame) -> Dict[str, Dict]:
    """Create mappings from categorical values to indices.

    Args:
        df: DataFrame containing author and category columns.

    Returns:
        Dictionary with:
            - 'author_to_idx': Mapping from author name to index
            - 'category_to_idx': Mapping from category name to index
            - 'num_authors': Number of unique authors
            - 'num_categories': Number of unique categories
    """
    authors = df["author"].unique().tolist()
    categories = df["category"].unique().tolist()

    author_to_idx = {auth: i for i, auth in enumerate(authors)}
    category_to_idx = {cat: i for i, cat in enumerate(categories)}

    return {
        "author_to_idx": author_to_idx,
        "category_to_idx": category_to_idx,
        "num_authors": len(authors),
        "num_categories": len(categories),
    }


def encode_categorical(
    df: pd.DataFrame,
    author_to_idx: Dict[str, int],
    category_to_idx: Dict[str, int],
    handle_unknown: str = "default",
) -> pd.DataFrame:
    """Encode categorical columns to indices.

    Args:
        df: DataFrame with author and category columns.
        author_to_idx: Mapping from author name to index.
        category_to_idx: Mapping from category name to index.
        handle_unknown: How to handle unknown values:
            - 'default': Map to index 0
            - 'error': Raise ValueError

    Returns:
        DataFrame with additional author_idx and category_idx columns.
    """
    df = df.copy()

    if handle_unknown == "default":
        df["author_idx"] = df["author"].map(lambda x: author_to_idx.get(x, 0))
        df["category_idx"] = df["category"].map(lambda x: category_to_idx.get(x, 0))
    else:
        df["author_idx"] = df["author"].map(author_to_idx)
        df["category_idx"] = df["category"].map(category_to_idx)

        if df["author_idx"].isna().any():
            unknown = df[df["author_idx"].isna()]["author"].unique()
            raise ValueError(f"Unknown authors: {unknown}")
        if df["category_idx"].isna().any():
            unknown = df[df["category_idx"].isna()]["category"].unique()
            raise ValueError(f"Unknown categories: {unknown}")

    return df


def create_dataset_from_df(
    df: pd.DataFrame,
    tokenizer,
    author_to_idx: Dict[str, int],
    category_to_idx: Dict[str, int],
    scaler: Optional[object] = None,
    fit_scaler: bool = False,
) -> TweetDataset:
    """Create TweetDataset from DataFrame.

    Args:
        df: DataFrame with all required columns.
        tokenizer: HuggingFace tokenizer.
        author_to_idx: Author name to index mapping.
        category_to_idx: Category name to index mapping.
        scaler: Optional sklearn scaler for numerical features.
        fit_scaler: If True, fit the scaler on this data (for training set only).

    Returns:
        TweetDataset instance.
    """
    from sklearn.preprocessing import StandardScaler

    # Encode categorical features
    df = encode_categorical(df, author_to_idx, category_to_idx)

    # Scale numerical features
    numerical = df[NUMERICAL_FEATURES].fillna(0).values

    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    if fit_scaler:
        numerical = scaler.fit_transform(numerical)
    else:
        numerical = scaler.transform(numerical)

    # Create dataset
    from tweet_classifier.config import TARGET_COLUMN

    dataset = TweetDataset(
        texts=df[TEXT_COLUMN],
        numerical_features=numerical,
        author_indices=df["author_idx"],
        category_indices=df["category_idx"],
        labels=df[TARGET_COLUMN],
        tokenizer=tokenizer,
    )

    return dataset, scaler


def save_scaler(scaler: Any, path: Union[str, Path]) -> None:
    """Save a fitted scaler to disk using joblib.

    Args:
        scaler: Fitted sklearn scaler (e.g., StandardScaler).
        path: Path to save the scaler file (.pkl or .joblib).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Union[str, Path]) -> Any:
    """Load a fitted scaler from disk.

    Args:
        path: Path to the saved scaler file.

    Returns:
        Loaded sklearn scaler.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)


def save_categorical_encodings(encodings: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save categorical encodings to disk using joblib.

    Args:
        encodings: Dictionary containing author_to_idx, category_to_idx, etc.
        path: Path to save the encodings file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encodings, path)


def load_categorical_encodings(path: Union[str, Path]) -> Dict[str, Any]:
    """Load categorical encodings from disk.

    Args:
        path: Path to the saved encodings file.

    Returns:
        Dictionary containing categorical encodings.

    Raises:
        FileNotFoundError: If the encodings file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Encodings file not found: {path}")
    return joblib.load(path)


def save_preprocessing_artifacts(
    scaler: Any,
    encodings: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Save all preprocessing artifacts (scaler and encodings) to a directory.

    Args:
        scaler: Fitted sklearn scaler.
        encodings: Categorical encodings dictionary.
        output_dir: Directory to save artifacts.

    Returns:
        Tuple of (scaler_path, encodings_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = output_dir / "scaler.pkl"
    encodings_path = output_dir / "encodings.pkl"

    save_scaler(scaler, scaler_path)
    save_categorical_encodings(encodings, encodings_path)

    return scaler_path, encodings_path


def load_preprocessing_artifacts(
    input_dir: Union[str, Path],
) -> Tuple[Any, Dict[str, Any]]:
    """Load all preprocessing artifacts from a directory.

    Args:
        input_dir: Directory containing saved artifacts.

    Returns:
        Tuple of (scaler, encodings).
    """
    input_dir = Path(input_dir)

    scaler_path = input_dir / "scaler.pkl"
    encodings_path = input_dir / "encodings.pkl"

    scaler = load_scaler(scaler_path)
    encodings = load_categorical_encodings(encodings_path)

    return scaler, encodings

