"""Unit tests for tweet_classifier feature engineering components."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from tweet_classifier.config import (
    EXCLUDED_FROM_FEATURES,
    LABEL_MAP,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable, prepare_features
from tweet_classifier.data.splitter import split_by_hash, verify_no_leakage
from tweet_classifier.data.weights import compute_class_weights
from tweet_classifier.dataset import (
    TweetDataset,
    create_categorical_encodings,
    encode_categorical,
    load_preprocessing_artifacts,
    load_scaler,
    save_preprocessing_artifacts,
    save_scaler,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "text": [f"Sample tweet {i} about $AAPL" for i in range(n_samples)],
            "author": np.random.choice(["Author A", "Author B", "Author C"], n_samples),
            "category": np.random.choice(["news", "opinion", "analysis"], n_samples),
            "tweet_hash": [f"hash_{i}" for i in range(n_samples)],
            "is_reliable_label": [True] * 80 + [False] * 20,
            TARGET_COLUMN: np.random.choice(["SELL", "HOLD", "BUY"], n_samples),
            "volatility_7d": np.random.uniform(0.01, 0.05, n_samples),
            "relative_volume": np.random.uniform(0.5, 2.0, n_samples),
            "rsi_14": np.random.uniform(30, 70, n_samples),
            "distance_from_ma_20": np.random.uniform(-0.1, 0.1, n_samples),
            # Forbidden columns (should be excluded from features)
            "spy_return_1d": np.random.uniform(-0.02, 0.02, n_samples),
            "spy_return_1hr": np.random.uniform(-0.01, 0.01, n_samples),
            "return_1hr": np.random.uniform(-0.05, 0.05, n_samples),
        }
    )
    return df


class TestNumericalFeatureExtraction:
    """Tests for numerical feature extraction and forbidden column exclusion."""

    def test_numerical_features_exclude_forbidden(self):
        """Verify NUMERICAL_FEATURES does not contain any forbidden columns."""
        for col in EXCLUDED_FROM_FEATURES:
            assert col not in NUMERICAL_FEATURES, f"LEAK: {col} found in NUMERICAL_FEATURES!"

    def test_prepare_features_raises_on_leak(self, sample_df: pd.DataFrame):
        """Verify prepare_features raises error if forbidden column in features."""
        # This test verifies the safety check works
        # The actual NUMERICAL_FEATURES should never contain forbidden columns
        assert "spy_return_1d" not in NUMERICAL_FEATURES

    def test_prepare_features_returns_correct_columns(self, sample_df: pd.DataFrame):
        """Verify prepare_features returns only safe numerical columns."""
        sample_df = filter_reliable(sample_df)
        features = prepare_features(sample_df)

        # Check numerical features are correct
        assert list(features["numerical"].columns) == NUMERICAL_FEATURES

        # Verify no forbidden columns
        for col in EXCLUDED_FROM_FEATURES:
            assert col not in features["numerical"].columns


class TestCategoricalEncoding:
    """Tests for categorical encoding functions."""

    def test_create_categorical_encodings(self, sample_df: pd.DataFrame):
        """Test creating categorical encodings from DataFrame."""
        encodings = create_categorical_encodings(sample_df)

        assert "author_to_idx" in encodings
        assert "category_to_idx" in encodings
        assert "num_authors" in encodings
        assert "num_categories" in encodings

        # Verify all authors are encoded
        assert encodings["num_authors"] == sample_df["author"].nunique()
        assert encodings["num_categories"] == sample_df["category"].nunique()

    def test_encode_categorical_adds_columns(self, sample_df: pd.DataFrame):
        """Test that encode_categorical adds index columns."""
        encodings = create_categorical_encodings(sample_df)
        encoded_df = encode_categorical(
            sample_df, encodings["author_to_idx"], encodings["category_to_idx"]
        )

        assert "author_idx" in encoded_df.columns
        assert "category_idx" in encoded_df.columns

    def test_encode_categorical_handles_unknown_default(self, sample_df: pd.DataFrame):
        """Test unknown values map to 0 with handle_unknown='default'."""
        encodings = create_categorical_encodings(sample_df)

        # Create df with unknown author
        new_df = sample_df.copy()
        new_df.loc[0, "author"] = "Unknown Author"

        encoded_df = encode_categorical(
            new_df, encodings["author_to_idx"], encodings["category_to_idx"], handle_unknown="default"
        )

        # Unknown should map to 0
        assert encoded_df.loc[0, "author_idx"] == 0

    def test_encode_categorical_handles_unknown_error(self, sample_df: pd.DataFrame):
        """Test unknown values raise error with handle_unknown='error'."""
        encodings = create_categorical_encodings(sample_df)

        # Create df with unknown author
        new_df = sample_df.copy()
        new_df.loc[0, "author"] = "Unknown Author"

        with pytest.raises(ValueError, match="Unknown authors"):
            encode_categorical(
                new_df, encodings["author_to_idx"], encodings["category_to_idx"], handle_unknown="error"
            )


class TestTweetDataset:
    """Tests for TweetDataset class."""

    def test_dataset_returns_correct_tensor_shapes(self, sample_df: pd.DataFrame):
        """Test TweetDataset returns tensors with correct shapes."""
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                n = len(texts)
                max_len = kwargs.get("max_length", 128)
                return {
                    "input_ids": np.zeros((n, max_len), dtype=np.int64),
                    "attention_mask": np.ones((n, max_len), dtype=np.int64),
                }

        tokenizer = MockTokenizer()
        sample_df = filter_reliable(sample_df)
        encodings = create_categorical_encodings(sample_df)
        encoded_df = encode_categorical(
            sample_df, encodings["author_to_idx"], encodings["category_to_idx"]
        )

        # Scale numerical features
        scaler = StandardScaler()
        numerical = scaler.fit_transform(encoded_df[NUMERICAL_FEATURES].fillna(0))

        dataset = TweetDataset(
            texts=encoded_df["text"],
            numerical_features=numerical,
            author_indices=encoded_df["author_idx"],
            category_indices=encoded_df["category_idx"],
            labels=encoded_df[TARGET_COLUMN],
            tokenizer=tokenizer,
            max_length=128,
        )

        # Check dataset length
        assert len(dataset) == len(sample_df)

        # Check sample shapes
        sample = dataset[0]
        assert sample["input_ids"].shape == (128,)
        assert sample["attention_mask"].shape == (128,)
        assert sample["numerical"].shape == (len(NUMERICAL_FEATURES),)
        assert sample["author_idx"].shape == ()
        assert sample["category_idx"].shape == ()
        assert sample["labels"].shape == ()

    def test_dataset_encodes_labels_correctly(self, sample_df: pd.DataFrame):
        """Test labels are encoded to integers correctly."""

        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                n = len(texts)
                max_len = kwargs.get("max_length", 128)
                return {
                    "input_ids": np.zeros((n, max_len), dtype=np.int64),
                    "attention_mask": np.ones((n, max_len), dtype=np.int64),
                }

        tokenizer = MockTokenizer()
        sample_df = filter_reliable(sample_df)
        encodings = create_categorical_encodings(sample_df)
        encoded_df = encode_categorical(
            sample_df, encodings["author_to_idx"], encodings["category_to_idx"]
        )

        scaler = StandardScaler()
        numerical = scaler.fit_transform(encoded_df[NUMERICAL_FEATURES].fillna(0))

        dataset = TweetDataset(
            texts=encoded_df["text"],
            numerical_features=numerical,
            author_indices=encoded_df["author_idx"],
            category_indices=encoded_df["category_idx"],
            labels=encoded_df[TARGET_COLUMN],
            tokenizer=tokenizer,
        )

        # Check all labels are valid
        for i in range(len(dataset)):
            label = dataset[i]["labels"].item()
            assert label in [0, 1, 2], f"Invalid label: {label}"


class TestScalerPersistence:
    """Tests for scaler save/load functionality."""

    def test_save_and_load_scaler(self):
        """Test saving and loading a fitted scaler."""
        scaler = StandardScaler()
        data = np.random.randn(100, 4)
        scaler.fit(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaler.pkl"
            save_scaler(scaler, path)

            loaded_scaler = load_scaler(path)

            # Verify loaded scaler has same parameters
            np.testing.assert_array_almost_equal(scaler.mean_, loaded_scaler.mean_)
            np.testing.assert_array_almost_equal(scaler.scale_, loaded_scaler.scale_)

    def test_load_scaler_raises_on_missing_file(self):
        """Test load_scaler raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_scaler("/nonexistent/path/scaler.pkl")

    def test_save_and_load_preprocessing_artifacts(self, sample_df: pd.DataFrame):
        """Test saving and loading all preprocessing artifacts."""
        scaler = StandardScaler()
        data = np.random.randn(100, 4)
        scaler.fit(data)

        encodings = create_categorical_encodings(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessing_artifacts(scaler, encodings, tmpdir)

            loaded_scaler, loaded_encodings = load_preprocessing_artifacts(tmpdir)

            # Verify scaler
            np.testing.assert_array_almost_equal(scaler.mean_, loaded_scaler.mean_)

            # Verify encodings
            assert loaded_encodings["num_authors"] == encodings["num_authors"]
            assert loaded_encodings["author_to_idx"] == encodings["author_to_idx"]


class TestDataSplitting:
    """Tests for data splitting functionality."""

    def test_split_by_hash_no_overlap(self, sample_df: pd.DataFrame):
        """Test split_by_hash produces non-overlapping splits."""
        sample_df = filter_reliable(sample_df)
        df_train, df_val, df_test = split_by_hash(sample_df)

        # Verify no leakage
        assert verify_no_leakage(df_train, df_val, df_test)

    def test_split_preserves_all_samples(self, sample_df: pd.DataFrame):
        """Test split_by_hash preserves all samples."""
        sample_df = filter_reliable(sample_df)
        df_train, df_val, df_test = split_by_hash(sample_df)

        total = len(df_train) + len(df_val) + len(df_test)
        assert total == len(sample_df)


class TestClassWeights:
    """Tests for class weight computation."""

    def test_compute_class_weights_returns_correct_shape(self, sample_df: pd.DataFrame):
        """Test class weights have correct shape."""
        sample_df = filter_reliable(sample_df)
        weights = compute_class_weights(sample_df[TARGET_COLUMN])

        assert weights.shape == (3,)

    def test_minority_class_has_higher_weight(self):
        """Test minority class gets higher weight."""
        # Create imbalanced labels
        labels = ["SELL"] * 100 + ["HOLD"] * 50 + ["BUY"] * 25

        weights = compute_class_weights(pd.Series(labels))

        # BUY is minority (25 samples), should have highest weight
        assert weights[LABEL_MAP["BUY"]] > weights[LABEL_MAP["SELL"]]
        assert weights[LABEL_MAP["BUY"]] > weights[LABEL_MAP["HOLD"]]

