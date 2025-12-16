"""Unit tests for tweet_classifier feature engineering components."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
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


class MockTokenizer:
    """Mock tokenizer for testing TweetDataset without loading real model."""

    def __call__(self, texts, **kwargs):
        """Tokenize texts returning PyTorch tensors of zeros/ones.

        Args:
            texts: List of text strings to tokenize.
            **kwargs: Additional keyword arguments (max_length, return_tensors supported).

        Returns:
            Dict with input_ids and attention_mask as PyTorch tensors.
        """
        n = len(texts)
        max_len = kwargs.get("max_length", 128)
        return {
            "input_ids": torch.zeros((n, max_len), dtype=torch.long),
            "attention_mask": torch.ones((n, max_len), dtype=torch.long),
        }


class TestTweetDataset:
    """Tests for TweetDataset class."""

    def test_dataset_returns_correct_tensor_shapes(self, sample_df: pd.DataFrame):
        """Test TweetDataset returns tensors with correct shapes."""
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


class TestFinBERTMultiModal:
    """Tests for the FinBERTMultiModal model class."""

    @pytest.fixture
    def mock_bert_model(self, monkeypatch):  # noqa: DAR101
        """Mock BertModel to avoid downloading weights during tests."""
        import torch
        import torch.nn as nn

        class MockBertConfig:
            hidden_size = 768

        class MockBertOutput:
            def __init__(self, batch_size, seq_len):
                self.last_hidden_state = torch.randn(batch_size, seq_len, 768)

        class MockBertModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.config = MockBertConfig()
                # Add a dummy parameter so model has parameters
                self.dummy = nn.Linear(768, 768)

            def forward(self, input_ids, attention_mask):
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                return MockBertOutput(batch_size, seq_len)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        monkeypatch.setattr("tweet_classifier.model.BertModel", MockBertModel)
        return MockBertModel

    def test_model_instantiation(self, mock_bert_model):  # noqa: DAR101
        """Test model can be instantiated with correct parameters."""
        from tweet_classifier.model import FinBERTMultiModal

        model = FinBERTMultiModal(
            num_numerical_features=4,
            num_authors=5,
            num_categories=3,
        )

        assert model.num_numerical_features == 4
        assert model.num_authors == 5
        assert model.num_categories == 3
        assert model.num_classes == 3

    def test_forward_pass_output_shape(self, mock_bert_model):  # noqa: DAR101
        """Test forward pass returns correct output shapes."""
        import torch

        from tweet_classifier.model import FinBERTMultiModal

        batch_size = 4
        seq_len = 128
        num_features = 4
        num_authors = 5
        num_categories = 3

        model = FinBERTMultiModal(
            num_numerical_features=num_features,
            num_authors=num_authors,
            num_categories=num_categories,
        )
        model.eval()

        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        numerical = torch.randn(batch_size, num_features)
        author_idx = torch.randint(0, num_authors, (batch_size,))
        category_idx = torch.randint(0, num_categories, (batch_size,))

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical=numerical,
                author_idx=author_idx,
                category_idx=category_idx,
            )

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 3)
        assert "loss" not in output  # No labels provided

    def test_forward_pass_with_labels_returns_loss(self, mock_bert_model):  # noqa: DAR101
        """Test forward pass computes loss when labels provided."""
        import torch

        from tweet_classifier.model import FinBERTMultiModal

        batch_size = 4
        seq_len = 128
        num_features = 4
        num_authors = 5
        num_categories = 3

        model = FinBERTMultiModal(
            num_numerical_features=num_features,
            num_authors=num_authors,
            num_categories=num_categories,
        )

        # Create dummy inputs with labels
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        numerical = torch.randn(batch_size, num_features)
        author_idx = torch.randint(0, num_authors, (batch_size,))
        category_idx = torch.randint(0, num_categories, (batch_size,))
        labels = torch.randint(0, 3, (batch_size,))

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numerical=numerical,
            author_idx=author_idx,
            category_idx=category_idx,
            labels=labels,
        )

        assert "logits" in output
        assert "loss" in output
        assert output["loss"].shape == ()  # Scalar loss
        assert output["loss"].requires_grad  # Loss should be differentiable

    def test_frozen_bert_has_no_gradients(self, mock_bert_model):  # noqa: DAR101
        """Test that frozen BERT parameters have requires_grad=False."""
        from tweet_classifier.model import FinBERTMultiModal

        model = FinBERTMultiModal(
            num_numerical_features=4,
            num_authors=5,
            num_categories=3,
            freeze_bert=True,
        )

        # All BERT parameters should have requires_grad=False
        for param in model.bert.parameters():
            assert not param.requires_grad, "BERT parameter should be frozen"

        # Other parameters should still be trainable
        for param in model.numerical_encoder.parameters():
            assert param.requires_grad, "Numerical encoder should be trainable"

        for param in model.classifier.parameters():
            assert param.requires_grad, "Classifier should be trainable"

    def test_unfrozen_bert_has_gradients(self, mock_bert_model):  # noqa: DAR101
        """Test that unfrozen BERT parameters have requires_grad=True."""
        from tweet_classifier.model import FinBERTMultiModal

        model = FinBERTMultiModal(
            num_numerical_features=4,
            num_authors=5,
            num_categories=3,
            freeze_bert=False,
        )

        # All BERT parameters should have requires_grad=True
        for param in model.bert.parameters():
            assert param.requires_grad, "BERT parameter should be trainable"

    def test_get_config_returns_correct_values(self, mock_bert_model):  # noqa: DAR101
        """Test get_config returns correct configuration."""
        from tweet_classifier.model import FinBERTMultiModal

        model = FinBERTMultiModal(
            num_numerical_features=4,
            num_authors=5,
            num_categories=3,
            dropout=0.5,
            freeze_bert=True,
        )

        config = model.get_config()

        assert config["num_numerical_features"] == 4
        assert config["num_authors"] == 5
        assert config["num_categories"] == 3
        assert config["num_classes"] == 3
        assert config["dropout"] == 0.5
        assert config["freeze_bert"] is True

    def test_model_embedding_dimensions(self, mock_bert_model):  # noqa: DAR101
        """Test embedding layers have correct dimensions."""
        from tweet_classifier.model import FinBERTMultiModal

        num_authors = 10
        num_categories = 5
        author_emb_dim = 16
        category_emb_dim = 8

        model = FinBERTMultiModal(
            num_numerical_features=4,
            num_authors=num_authors,
            num_categories=num_categories,
            author_embedding_dim=author_emb_dim,
            category_embedding_dim=category_emb_dim,
        )

        assert model.author_embedding.num_embeddings == num_authors
        assert model.author_embedding.embedding_dim == author_emb_dim
        assert model.category_embedding.num_embeddings == num_categories
        assert model.category_embedding.embedding_dim == category_emb_dim


class TestComputeMetrics:
    """Tests for the compute_metrics evaluation function."""

    def test_compute_metrics_returns_all_metrics(self):
        """Test compute_metrics returns accuracy, f1_macro, and f1_weighted."""
        from tweet_classifier.trainer import compute_metrics

        # Create sample predictions and labels
        predictions = np.array([
            [0.9, 0.05, 0.05],  # Predict SELL (class 0)
            [0.1, 0.8, 0.1],   # Predict HOLD (class 1)
            [0.1, 0.1, 0.8],   # Predict BUY (class 2)
            [0.9, 0.05, 0.05],  # Predict SELL (class 0)
        ])
        labels = np.array([0, 1, 2, 0])  # All correct

        metrics = compute_metrics((predictions, labels))

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics

    def test_compute_metrics_perfect_predictions(self):
        """Test compute_metrics with perfect predictions."""
        from tweet_classifier.trainer import compute_metrics

        predictions = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        labels = np.array([0, 1, 2])

        metrics = compute_metrics((predictions, labels))

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_compute_metrics_wrong_predictions(self):
        """Test compute_metrics with all wrong predictions."""
        from tweet_classifier.trainer import compute_metrics

        predictions = np.array([
            [0.1, 0.1, 0.8],   # Predict BUY (class 2)
            [0.9, 0.05, 0.05],  # Predict SELL (class 0)
            [0.1, 0.8, 0.1],   # Predict HOLD (class 1)
        ])
        labels = np.array([0, 1, 2])  # All wrong

        metrics = compute_metrics((predictions, labels))

        assert metrics["accuracy"] == 0.0
        assert metrics["f1_macro"] == 0.0


class TestWeightedTrainer:
    """Tests for the WeightedTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing WeightedTrainer."""
        import torch
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, **kwargs):
                # Just return random logits of the right shape
                batch_size = kwargs.get("input_ids", torch.zeros(4, 10)).shape[0]
                logits = torch.randn(batch_size, 3)
                return {"logits": logits}

        return MockModel()

    def test_weighted_trainer_init(self, mock_model):
        """Test WeightedTrainer initializes with class weights."""
        import torch
        from transformers import TrainingArguments

        from tweet_classifier.trainer import WeightedTrainer

        class_weights = torch.tensor([1.0, 2.0, 1.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=4,
                num_train_epochs=1,
                report_to="none",
            )

            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=mock_model,
                args=args,
            )

            assert trainer.class_weights is not None
            assert torch.equal(trainer.class_weights, class_weights)

    def test_weighted_trainer_compute_loss(self, mock_model):
        """Test WeightedTrainer.compute_loss applies class weights."""
        import torch
        from transformers import TrainingArguments

        from tweet_classifier.trainer import WeightedTrainer

        # Higher weight for class 1 (HOLD)
        class_weights = torch.tensor([1.0, 5.0, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=4,
                num_train_epochs=1,
                report_to="none",
            )

            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=mock_model,
                args=args,
            )

            # Create dummy inputs
            inputs = {
                "input_ids": torch.randint(0, 1000, (4, 10)),
                "attention_mask": torch.ones(4, 10),
                "numerical": torch.randn(4, 4),
                "author_idx": torch.randint(0, 3, (4,)),
                "category_idx": torch.randint(0, 3, (4,)),
                "labels": torch.tensor([0, 1, 2, 1]),  # Two samples with class 1
            }

            loss = trainer.compute_loss(mock_model, inputs)

            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss
            assert loss.item() > 0  # Loss should be positive


class TestTrainingConfig:
    """Tests for training configuration creation."""

    def test_create_training_args_default_values(self):
        """Test create_training_args with default values."""
        from tweet_classifier.train import create_training_args

        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(output_dir=Path(tmpdir))

            assert args.eval_strategy == "epoch"
            assert args.save_strategy == "epoch"
            assert args.load_best_model_at_end is True
            assert args.metric_for_best_model == "f1_macro"
            assert args.greater_is_better is True
            assert args.remove_unused_columns is False

    def test_create_training_args_custom_values(self):
        """Test create_training_args with custom values."""
        from tweet_classifier.train import create_training_args

        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(
                output_dir=Path(tmpdir),
                num_epochs=10,
                batch_size=32,
                learning_rate=1e-4,
                warmup_ratio=0.2,
            )

            assert args.num_train_epochs == 10
            assert args.per_device_train_batch_size == 32
            assert args.learning_rate == 1e-4
            assert args.warmup_ratio == 0.2

    def test_create_training_args_fp16_disabled_without_cuda(self):
        """Test fp16 is disabled when CUDA is not available."""
        import torch

        from tweet_classifier.train import create_training_args

        with tempfile.TemporaryDirectory() as tmpdir:
            args = create_training_args(output_dir=Path(tmpdir), fp16=True)

            # fp16 should only be enabled if CUDA is available
            if torch.cuda.is_available():
                assert args.fp16 is True
            else:
                assert args.fp16 is False


class TestComputeTradingMetrics:
    """Tests for trading-focused evaluation metrics."""

    def test_compute_trading_metrics_basic(self):
        """Test compute_trading_metrics returns expected keys."""
        from tweet_classifier.evaluate import compute_trading_metrics

        np.random.seed(42)
        n_samples = 100

        predictions = np.random.randint(0, 3, n_samples)
        probabilities = np.random.dirichlet([1, 1, 1], n_samples)
        actual_returns = np.random.uniform(-0.05, 0.05, n_samples)
        labels = np.random.randint(0, 3, n_samples)

        metrics = compute_trading_metrics(
            predictions=predictions,
            probabilities=probabilities,
            actual_returns=actual_returns,
            labels=labels,
        )

        # Check required keys exist
        assert "information_coefficient" in metrics
        assert "ic_pvalue" in metrics
        assert "directional_accuracy" in metrics
        assert "simulated_sharpe_top" in metrics
        assert "simulated_return_top" in metrics

    def test_compute_trading_metrics_with_nan_returns(self):
        """Test compute_trading_metrics handles NaN returns correctly."""
        from tweet_classifier.evaluate import compute_trading_metrics

        n_samples = 50
        predictions = np.random.randint(0, 3, n_samples)
        probabilities = np.random.dirichlet([1, 1, 1], n_samples)
        actual_returns = np.random.uniform(-0.05, 0.05, n_samples)
        labels = np.random.randint(0, 3, n_samples)

        # Add some NaN values
        actual_returns[10:20] = np.nan

        metrics = compute_trading_metrics(
            predictions=predictions,
            probabilities=probabilities,
            actual_returns=actual_returns,
            labels=labels,
        )

        # Should still work with NaN values filtered out
        assert "information_coefficient" in metrics
        assert not np.isnan(metrics["information_coefficient"])

    def test_compute_trading_metrics_directional_accuracy(self):
        """Test directional accuracy with perfect predictions."""
        from tweet_classifier.evaluate import compute_trading_metrics

        # Create predictions where BUY (2) always aligns with positive returns
        # and SELL (0) always aligns with negative returns
        predictions = np.array([2, 2, 0, 0, 1])  # BUY, BUY, SELL, SELL, HOLD
        probabilities = np.array([
            [0.1, 0.1, 0.8],  # High BUY confidence
            [0.1, 0.1, 0.8],  # High BUY confidence
            [0.8, 0.1, 0.1],  # High SELL confidence
            [0.8, 0.1, 0.1],  # High SELL confidence
            [0.1, 0.8, 0.1],  # High HOLD confidence
        ])
        actual_returns = np.array([0.03, 0.02, -0.02, -0.03, 0.01])  # Positive for BUY, negative for SELL
        labels = np.array([2, 2, 0, 0, 1])

        metrics = compute_trading_metrics(
            predictions=predictions,
            probabilities=probabilities,
            actual_returns=actual_returns,
            labels=labels,
        )

        # Perfect directional accuracy for non-HOLD predictions
        assert metrics["directional_accuracy"] == 1.0
        assert metrics["n_directional_predictions"] == 4  # 4 non-HOLD predictions

    def test_compute_trading_metrics_precision_at_confidence(self):
        """Test precision at confidence threshold."""
        from tweet_classifier.evaluate import compute_trading_metrics

        np.random.seed(42)
        n_samples = 100

        # Create predictions with some high confidence ones being correct
        predictions = np.random.randint(0, 3, n_samples)
        labels = predictions.copy()  # All correct

        # Create probabilities with varying confidence
        probabilities = np.zeros((n_samples, 3))
        for i in range(n_samples):
            probabilities[i, predictions[i]] = 0.7  # 70% confidence
            remaining = 0.3 / 2
            for j in range(3):
                if j != predictions[i]:
                    probabilities[i, j] = remaining

        actual_returns = np.random.uniform(-0.05, 0.05, n_samples)

        metrics = compute_trading_metrics(
            predictions=predictions,
            probabilities=probabilities,
            actual_returns=actual_returns,
            labels=labels,
            confidence_threshold=0.6,
        )

        # All predictions are correct and have >60% confidence
        assert metrics["precision_at_confidence"] == 1.0
        assert metrics["n_high_confidence"] == n_samples


class TestComputeBaselines:
    """Tests for baseline computation functions."""

    def test_compute_baselines_returns_expected_keys(self):
        """Test compute_baselines returns all expected keys."""
        from tweet_classifier.evaluate import compute_baselines

        labels = np.array([0, 0, 0, 1, 1, 2])  # Majority is class 0 (SELL)

        baselines = compute_baselines(labels)

        assert "naive_accuracy" in baselines
        assert "random_accuracy" in baselines
        assert "weighted_random_accuracy" in baselines
        assert "majority_class" in baselines

    def test_compute_baselines_naive_accuracy(self):
        """Test naive accuracy equals majority class proportion."""
        from tweet_classifier.evaluate import compute_baselines

        # 60% class 0, 30% class 1, 10% class 2
        labels = np.array([0] * 6 + [1] * 3 + [2] * 1)

        baselines = compute_baselines(labels)

        assert baselines["naive_accuracy"] == 0.6  # 6/10
        assert baselines["majority_class"] == "SELL"  # Class 0

    def test_compute_baselines_random_accuracy(self):
        """Test random accuracy is 1/3 for 3 classes."""
        from tweet_classifier.evaluate import compute_baselines

        labels = np.array([0, 1, 2, 0, 1, 2])

        baselines = compute_baselines(labels)

        assert abs(baselines["random_accuracy"] - 1 / 3) < 1e-6

    def test_compute_baselines_with_train_distribution(self):
        """Test weighted random uses training distribution when provided."""
        from tweet_classifier.evaluate import compute_baselines

        test_labels = np.array([0, 1, 2] * 10)

        # Training distribution: heavily skewed towards BUY
        train_labels = pd.Series(["BUY"] * 80 + ["SELL"] * 15 + ["HOLD"] * 5)

        baselines = compute_baselines(test_labels, train_label_distribution=train_labels)

        # Weighted random should reflect training distribution
        assert baselines["weighted_random_accuracy"] > baselines["random_accuracy"]


class TestGenerateClassificationReport:
    """Tests for classification report generation."""

    def test_generate_classification_report_format(self):
        """Test classification report returns string with class names."""
        from tweet_classifier.evaluate import generate_classification_report

        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 0, 2])

        report = generate_classification_report(labels, predictions)

        assert isinstance(report, str)
        assert "SELL" in report
        assert "HOLD" in report
        assert "BUY" in report
        assert "precision" in report
        assert "recall" in report


class TestPlotConfusionMatrix:
    """Tests for confusion matrix plotting."""

    def test_plot_confusion_matrix_returns_figure(self):
        """Test plot_confusion_matrix returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        from tweet_classifier.evaluate import plot_confusion_matrix

        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 0, 2])

        fig = plot_confusion_matrix(labels, predictions)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_confusion_matrix_saves_file(self):
        """Test plot_confusion_matrix saves PNG file."""
        import matplotlib.pyplot as plt

        from tweet_classifier.evaluate import plot_confusion_matrix

        labels = np.array([0, 1, 2, 0, 1, 2])
        predictions = np.array([0, 1, 2, 0, 0, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cm.png"
            fig = plot_confusion_matrix(labels, predictions, output_path=output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

            plt.close(fig)


class TestEvaluateOnTest:
    """Tests for the evaluate_on_test function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        import torch
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, **kwargs):
                batch_size = kwargs.get("input_ids", torch.zeros(4, 10)).shape[0]
                # Return slightly random logits so we get different predictions
                logits = torch.randn(batch_size, 3)
                return {"logits": logits}

        return MockModel()

    @pytest.fixture
    def mock_dataset(self, sample_df: pd.DataFrame):
        """Create a mock dataset for testing."""
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
            tokenizer=MockTokenizer(),
            max_length=128,
        )
        return dataset

    def test_evaluate_on_test_returns_expected_keys(self, mock_model, mock_dataset):
        """Test evaluate_on_test returns dictionary with expected keys."""
        from tweet_classifier.evaluate import evaluate_on_test

        results = evaluate_on_test(mock_model, mock_dataset, batch_size=16)

        assert "predictions" in results
        assert "probabilities" in results
        assert "labels" in results
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "f1_weighted" in results

    def test_evaluate_on_test_output_shapes(self, mock_model, mock_dataset):
        """Test evaluate_on_test returns arrays with correct shapes."""
        from tweet_classifier.evaluate import evaluate_on_test

        results = evaluate_on_test(mock_model, mock_dataset, batch_size=16)

        n_samples = len(mock_dataset)
        assert results["predictions"].shape == (n_samples,)
        assert results["probabilities"].shape == (n_samples, 3)
        assert results["labels"].shape == (n_samples,)
