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
    TARGET_COLUMN,
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
        market_regime_indices: Union[pd.Series, np.ndarray],  # Phase 1
        sector_indices: Union[pd.Series, np.ndarray],  # Phase 1
        market_cap_indices: Union[pd.Series, np.ndarray],  # Phase 1
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
            market_regime_indices: Series or array of market regime indices. [Phase 1]
            sector_indices: Series or array of sector indices. [Phase 1]
            market_cap_indices: Series or array of market cap indices. [Phase 1]
            labels: Series or array of labels (strings or integers).
            tokenizer: HuggingFace tokenizer for text encoding.
            max_length: Maximum token length for text (default: 128).

        Raises:
            ValueError: If labels array is empty.
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
        if isinstance(market_regime_indices, pd.Series):
            market_regime_indices = market_regime_indices.values
        if isinstance(sector_indices, pd.Series):
            sector_indices = sector_indices.values
        if isinstance(market_cap_indices, pd.Series):
            market_cap_indices = market_cap_indices.values
            
        self.author_idx = torch.tensor(author_indices, dtype=torch.long)
        self.category_idx = torch.tensor(category_indices, dtype=torch.long)
        self.market_regime_idx = torch.tensor(market_regime_indices, dtype=torch.long)
        self.sector_idx = torch.tensor(sector_indices, dtype=torch.long)
        self.market_cap_idx = torch.tensor(market_cap_indices, dtype=torch.long)

        # Convert labels
        if isinstance(labels, pd.Series):
            labels = labels.values
        if len(labels) == 0:
            raise ValueError("Labels array cannot be empty")
        if isinstance(labels[0], str):
            labels = np.array([LABEL_MAP[label] for label in labels])
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
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
                - market_regime_idx: Market regime index [Phase 1]
                - sector_idx: Sector index [Phase 1]
                - market_cap_idx: Market cap index [Phase 1]
                - labels: Label (integer)
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "numerical": self.numerical[idx],
            "author_idx": self.author_idx[idx],
            "category_idx": self.category_idx[idx],
            "market_regime_idx": self.market_regime_idx[idx],
            "sector_idx": self.sector_idx[idx],
            "market_cap_idx": self.market_cap_idx[idx],
            "labels": self.labels[idx],
        }


def create_categorical_encodings(df: pd.DataFrame) -> Dict[str, Dict]:
    """Create mappings from categorical values to indices.

    Args:
        df: DataFrame containing author, category, and Phase 1 categorical columns.

    Returns:
        Dictionary with:
            - 'author_to_idx': Mapping from author name to index
            - 'category_to_idx': Mapping from category name to index
            - 'market_regime_to_idx': Mapping from market regime to index [Phase 1]
            - 'sector_to_idx': Mapping from sector to index [Phase 1]
            - 'market_cap_to_idx': Mapping from market cap bucket to index [Phase 1]
            - 'num_authors': Number of unique authors
            - 'num_categories': Number of unique categories
            - 'num_market_regimes': Number of unique market regimes [Phase 1]
            - 'num_sectors': Number of unique sectors [Phase 1]
            - 'num_market_caps': Number of unique market caps [Phase 1]
    """
    authors = df["author"].unique().tolist()
    categories = df["category"].unique().tolist()
    market_regimes = df["market_regime"].fillna("calm").unique().tolist()  # Phase 1
    sectors = df["sector"].fillna("Other").unique().tolist()  # Phase 1
    market_caps = df["market_cap_bucket"].fillna("unknown").unique().tolist()  # Phase 1

    author_to_idx = {auth: i for i, auth in enumerate(authors)}
    category_to_idx = {cat: i for i, cat in enumerate(categories)}
    market_regime_to_idx = {reg: i for i, reg in enumerate(market_regimes)}
    sector_to_idx = {sec: i for i, sec in enumerate(sectors)}
    market_cap_to_idx = {cap: i for i, cap in enumerate(market_caps)}

    return {
        "author_to_idx": author_to_idx,
        "category_to_idx": category_to_idx,
        "market_regime_to_idx": market_regime_to_idx,
        "sector_to_idx": sector_to_idx,
        "market_cap_to_idx": market_cap_to_idx,
        "num_authors": len(authors),
        "num_categories": len(categories),
        "num_market_regimes": len(market_regimes),
        "num_sectors": len(sectors),
        "num_market_caps": len(market_caps),
    }


def encode_categorical(
    df: pd.DataFrame,
    author_to_idx: Dict[str, int],
    category_to_idx: Dict[str, int],
    market_regime_to_idx: Dict[str, int],  # Phase 1
    sector_to_idx: Dict[str, int],  # Phase 1
    market_cap_to_idx: Dict[str, int],  # Phase 1
    handle_unknown: str = "default",
) -> pd.DataFrame:
    """Encode categorical columns to indices.

    Args:
        df: DataFrame with author, category, and Phase 1 categorical columns.
        author_to_idx: Mapping from author name to index.
        category_to_idx: Mapping from category name to index.
        market_regime_to_idx: Mapping from market regime to index. [Phase 1]
        sector_to_idx: Mapping from sector to index. [Phase 1]
        market_cap_to_idx: Mapping from market cap to index. [Phase 1]
        handle_unknown: How to handle unknown values:
            - 'default': Map to index 0
            - 'error': Raise ValueError

    Returns:
        DataFrame with additional *_idx columns for all categorical features.

    Raises:
        ValueError: If handle_unknown='error' and unknown values found.
    """
    df = df.copy()

    # Fill NaN values for Phase 1 features
    df["market_regime"] = df["market_regime"].fillna("calm")
    df["sector"] = df["sector"].fillna("Other")
    df["market_cap_bucket"] = df["market_cap_bucket"].fillna("unknown")

    if handle_unknown == "default":
        df["author_idx"] = df["author"].map(lambda x: author_to_idx.get(x, 0))
        df["category_idx"] = df["category"].map(lambda x: category_to_idx.get(x, 0))
        df["market_regime_idx"] = df["market_regime"].map(lambda x: market_regime_to_idx.get(x, 0))
        df["sector_idx"] = df["sector"].map(lambda x: sector_to_idx.get(x, 0))
        df["market_cap_idx"] = df["market_cap_bucket"].map(lambda x: market_cap_to_idx.get(x, 0))
    else:
        df["author_idx"] = df["author"].map(author_to_idx)
        df["category_idx"] = df["category"].map(category_to_idx)
        df["market_regime_idx"] = df["market_regime"].map(market_regime_to_idx)
        df["sector_idx"] = df["sector"].map(sector_to_idx)
        df["market_cap_idx"] = df["market_cap_bucket"].map(market_cap_to_idx)

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
    market_regime_to_idx: Dict[str, int],  # Phase 1
    sector_to_idx: Dict[str, int],  # Phase 1
    market_cap_to_idx: Dict[str, int],  # Phase 1
    scaler: Optional[Any] = None,
    fit_scaler: bool = False,
) -> Tuple[TweetDataset, Any]:
    """Create TweetDataset from DataFrame.

    Args:
        df: DataFrame with all required columns.
        tokenizer: HuggingFace tokenizer.
        author_to_idx: Author name to index mapping.
        category_to_idx: Category name to index mapping.
        market_regime_to_idx: Market regime to index mapping. [Phase 1]
        sector_to_idx: Sector to index mapping. [Phase 1]
        market_cap_to_idx: Market cap to index mapping. [Phase 1]
        scaler: Optional sklearn scaler for numerical features.
        fit_scaler: If True, fit the scaler on this data (for training set only).

    Returns:
        Tuple of (TweetDataset instance, fitted scaler).
    """
    from sklearn.preprocessing import StandardScaler

    # Encode categorical features
    df = encode_categorical(df, author_to_idx, category_to_idx, market_regime_to_idx, sector_to_idx, market_cap_to_idx)

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
    dataset = TweetDataset(
        texts=df[TEXT_COLUMN],
        numerical_features=numerical,
        author_indices=df["author_idx"],
        category_indices=df["category_idx"],
        market_regime_indices=df["market_regime_idx"],  # Phase 1
        sector_indices=df["sector_idx"],  # Phase 1
        market_cap_indices=df["market_cap_idx"],  # Phase 1
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
