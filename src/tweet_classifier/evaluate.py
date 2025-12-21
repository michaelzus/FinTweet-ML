"""Evaluation module for FinBERT tweet classifier.

This module provides comprehensive evaluation utilities including:
- Test set evaluation with standard ML metrics
- Confusion matrix visualization
- Trading-focused metrics (IC, directional accuracy, Sharpe)
- Baseline comparisons

Usage:
    python -m tweet_classifier.evaluate --model-dir models/finbert-tweet-classifier/final
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import BertTokenizer

from tweet_classifier.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    FINBERT_MODEL_NAME,
    LABEL_MAP,
    LABEL_MAP_INV,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.data.splitter import split_by_hash
from tweet_classifier.dataset import (
    create_dataset_from_df,
    load_preprocessing_artifacts,
)
from tweet_classifier.model import FinBERTMultiModal

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model_for_evaluation(
    model_dir: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[FinBERTMultiModal, Any, Dict[str, Any]]:
    """Load a trained model and preprocessing artifacts for evaluation.

    Args:
        model_dir: Directory containing saved model and artifacts.
        device: Torch device to load model on. Defaults to CUDA if available.

    Returns:
        Tuple of (model, scaler, encodings).

    Raises:
        FileNotFoundError: If model config or weights not found.
    """
    model_dir = Path(model_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config
    config_path = model_dir.parent / "model_config.json"
    if not config_path.exists():
        config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at {config_path}")

    with open(config_path) as f:
        model_config = json.load(f)

    # Load preprocessing artifacts
    artifacts_dir = model_dir.parent if model_dir.name == "final" else model_dir
    scaler, encodings = load_preprocessing_artifacts(artifacts_dir)

    # Initialize model
    model = FinBERTMultiModal(
        num_numerical_features=model_config["num_numerical_features"],
        num_authors=model_config["num_authors"],
        num_categories=model_config["num_categories"],
        num_market_regimes=model_config.get("num_market_regimes", 5),
        num_sectors=model_config.get("num_sectors", 12),
        num_market_caps=model_config.get("num_market_caps", 5),
        num_classes=model_config.get("num_classes", 3),
        freeze_bert=model_config.get("freeze_bert", False),
        dropout=model_config.get("dropout", 0.3),
        author_embedding_dim=model_config.get("author_embedding_dim", 16),
        category_embedding_dim=model_config.get("category_embedding_dim", 8),
        market_regime_embedding_dim=model_config.get("market_regime_embedding_dim", 4),
        sector_embedding_dim=model_config.get("sector_embedding_dim", 8),
        market_cap_embedding_dim=model_config.get("market_cap_embedding_dim", 4),
        numerical_hidden_dim=model_config.get("numerical_hidden_dim", 32),
    )

    # Load weights
    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        # Try loading via HuggingFace method
        state_dict_path = list(model_dir.glob("*.bin")) + list(
            model_dir.glob("*.safetensors")
        )
        if state_dict_path:
            weights_path = state_dict_path[0]
        else:
            raise FileNotFoundError(f"Model weights not found in {model_dir}")

    # Load state dict based on file format
    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device=str(device))
    else:
        # Use weights_only=True for security - only load tensors, not arbitrary Python objects
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, scaler, encodings


def evaluate_on_test(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run model predictions on test dataset and compute metrics.

    Args:
        model: Trained FinBERTMultiModal model.
        test_dataset: TweetDataset with test samples.
        device: Torch device for inference.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with:
            - predictions: Array of predicted class indices
            - probabilities: Array of softmax probabilities (n_samples, n_classes)
            - labels: Array of true labels
            - accuracy: Overall accuracy
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical = batch["numerical"].to(device)
            author_idx = batch["author_idx"].to(device)
            category_idx = batch["category_idx"].to(device)
            market_regime_idx = batch["market_regime_idx"].to(device)
            sector_idx = batch["sector_idx"].to(device)
            market_cap_idx = batch["market_cap_idx"].to(device)
            labels = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical=numerical,
                author_idx=author_idx,
                category_idx=category_idx,
                market_regime_idx=market_regime_idx,
                sector_idx=sector_idx,
                market_cap_idx=market_cap_idx,
            )

            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels,
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def generate_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> str:
    """Generate detailed classification report with per-class metrics.

    Args:
        labels: True labels (integers).
        predictions: Predicted labels (integers).

    Returns:
        Formatted classification report string.
    """
    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]
    return classification_report(labels, predictions, target_names=target_names)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Generate and optionally save confusion matrix visualization.

    Args:
        labels: True labels (integers).
        predictions: Predicted labels (integers).
        output_path: Optional path to save the figure.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure object.
    """
    cm = confusion_matrix(labels, predictions)
    class_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV.keys())]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {output_path}")

    return fig


def compute_trading_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    actual_returns: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.6,
    top_pct: float = 0.3,
) -> Dict[str, Any]:
    """Compute trading-relevant metrics beyond standard ML metrics.

    Args:
        predictions: Model predictions (0=SELL, 1=HOLD, 2=BUY).
        probabilities: Softmax probabilities (n_samples, 3).
        actual_returns: Actual returns (e.g., return_to_next_open).
        labels: True labels for accuracy calculation.
        transaction_cost: Cost per trade as decimal (default 0.1%).
        confidence_threshold: Threshold for high-confidence predictions.
        top_pct: Top percentage for Sharpe calculation.

    Returns:
        Dictionary with trading metrics:
            - information_coefficient: IC (correlation of confidence vs returns)
            - ic_pvalue: P-value of IC
            - directional_accuracy: Accuracy on non-HOLD predictions
            - simulated_sharpe_top: Sharpe ratio on top-confidence trades
            - simulated_return_top: Annualized return on top-confidence trades
            - precision_at_confidence: Accuracy at high confidence
            - n_high_confidence: Number of high-confidence predictions
    """
    results: Dict[str, Any] = {}

    # Filter out NaN returns
    valid_mask = ~np.isnan(actual_returns)
    if valid_mask.sum() == 0:
        logger.warning("No valid returns for trading metrics computation")
        return results

    valid_predictions = predictions[valid_mask]
    valid_probabilities = probabilities[valid_mask]
    valid_returns = actual_returns[valid_mask]
    valid_labels = labels[valid_mask]

    # 1. Information Coefficient (IC)
    # Correlation between model's bullish confidence and actual returns
    buy_confidence = valid_probabilities[:, 2] - valid_probabilities[:, 0]  # BUY - SELL
    ic, ic_pvalue = spearmanr(buy_confidence, valid_returns)
    results["information_coefficient"] = float(ic) if not np.isnan(ic) else 0.0
    results["ic_pvalue"] = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0

    # 2. Directional Accuracy (ignoring HOLD predictions)
    non_hold_mask = valid_predictions != 1
    if non_hold_mask.sum() > 0:
        # Model predicts BUY (+1) or SELL (-1)
        predicted_direction = np.where(valid_predictions[non_hold_mask] == 2, 1, -1)
        actual_direction = np.sign(valid_returns[non_hold_mask])

        # Count correct direction (excluding zero returns)
        non_zero_returns = actual_direction != 0
        if non_zero_returns.sum() > 0:
            directional_correct = (
                predicted_direction[non_zero_returns]
                == actual_direction[non_zero_returns]
            ).sum()
            results["directional_accuracy"] = float(
                directional_correct / non_zero_returns.sum()
            )
            results["n_directional_predictions"] = int(non_zero_returns.sum())
        else:
            results["directional_accuracy"] = 0.0
            results["n_directional_predictions"] = 0
    else:
        results["directional_accuracy"] = 0.0
        results["n_directional_predictions"] = 0

    # 3. Simulated Trading Sharpe (Top-% confidence predictions)
    n_top = max(1, int(top_pct * len(valid_predictions)))
    max_confidence = valid_probabilities.max(axis=1)
    top_indices = np.argsort(max_confidence)[-n_top:]

    simulated_returns = []
    for idx in top_indices:
        pred = valid_predictions[idx]
        ret = valid_returns[idx]

        if pred == 2:  # BUY
            simulated_returns.append(ret - transaction_cost)
        elif pred == 0:  # SELL
            simulated_returns.append(-ret - transaction_cost)
        else:  # HOLD
            simulated_returns.append(0)

    simulated_returns = np.array(simulated_returns)
    if len(simulated_returns) > 0 and simulated_returns.std() > 0:
        sharpe = (simulated_returns.mean() / simulated_returns.std()) * np.sqrt(252)
        results["simulated_sharpe_top"] = float(sharpe)
        results["simulated_return_top"] = float(simulated_returns.mean() * 252)
    else:
        results["simulated_sharpe_top"] = 0.0
        results["simulated_return_top"] = 0.0
    results["n_top_trades"] = n_top

    # 4. Precision @ High Confidence
    high_conf_mask = max_confidence > confidence_threshold
    n_high_conf = high_conf_mask.sum()
    results["n_high_confidence"] = int(n_high_conf)

    if n_high_conf >= 10:
        high_conf_preds = valid_predictions[high_conf_mask]
        high_conf_labels = valid_labels[high_conf_mask]
        precision_high_conf = (high_conf_preds == high_conf_labels).mean()
        results["precision_at_confidence"] = float(precision_high_conf)
        results["confidence_threshold"] = confidence_threshold
    else:
        results["precision_at_confidence"] = None
        results["confidence_threshold"] = confidence_threshold

    return results


def compute_baselines(
    labels: np.ndarray,
    train_label_distribution: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute baseline accuracies for comparison.

    Args:
        labels: True test labels (integers).
        train_label_distribution: Optional label distribution from training set.
            If None, uses test set distribution for weighted random baseline.

    Returns:
        Dictionary with baseline metrics:
            - naive_accuracy: Always predicting majority class
            - random_accuracy: Random uniform prediction (1/3)
            - weighted_random_accuracy: Random weighted by class distribution
            - majority_class: The majority class label
    """
    results: Dict[str, float] = {}

    # Count label distribution in test set
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))

    # Naive baseline: always predict majority class
    majority_class = max(label_dist.keys(), key=lambda k: label_dist[k])
    naive_accuracy = label_dist[majority_class] / len(labels)
    results["naive_accuracy"] = float(naive_accuracy)
    results["majority_class"] = LABEL_MAP_INV[majority_class]

    # Random baseline: uniform random (1/3 for 3 classes)
    num_classes = len(LABEL_MAP)
    results["random_accuracy"] = 1.0 / num_classes

    # Weighted random baseline
    if train_label_distribution is not None:
        class_probs = train_label_distribution.value_counts(normalize=True)
        weighted_random_acc = (class_probs**2).sum()
        results["weighted_random_accuracy"] = float(weighted_random_acc)
    else:
        class_probs = counts / counts.sum()
        weighted_random_acc = (class_probs**2).sum()
        results["weighted_random_accuracy"] = float(weighted_random_acc)

    return results


def run_full_evaluation(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    df_test: pd.DataFrame,
    df_train: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run complete evaluation pipeline and save results.

    Args:
        model: Trained model.
        test_dataset: Test dataset.
        df_test: Test DataFrame with return columns for trading metrics.
        df_train: Optional training DataFrame for baseline computation.
        output_dir: Directory to save evaluation results.
        device: Torch device.

    Returns:
        Dictionary with all evaluation results.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # 1. Standard evaluation
    logger.info("Running test set evaluation...")
    eval_results = evaluate_on_test(model, test_dataset, device=device)
    results["accuracy"] = eval_results["accuracy"]
    results["f1_macro"] = eval_results["f1_macro"]
    results["f1_weighted"] = eval_results["f1_weighted"]

    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"Test F1 (weighted): {results['f1_weighted']:.4f}")

    # 2. Classification report
    logger.info("\n" + "=" * 50)
    logger.info("Classification Report:")
    report = generate_classification_report(
        eval_results["labels"], eval_results["predictions"]
    )
    logger.info("\n" + report)
    results["classification_report"] = report

    # 3. Confusion matrix
    if output_dir is not None:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            eval_results["labels"], eval_results["predictions"], cm_path
        )
        results["confusion_matrix_path"] = str(cm_path)

    # 4. Trading metrics
    return_col = "return_to_next_open"
    if return_col in df_test.columns:
        logger.info("\n" + "=" * 50)
        logger.info("Computing trading metrics...")
        actual_returns = df_test[return_col].values
        trading_metrics = compute_trading_metrics(
            predictions=eval_results["predictions"],
            probabilities=eval_results["probabilities"],
            actual_returns=actual_returns,
            labels=eval_results["labels"],
        )
        results["trading_metrics"] = trading_metrics

        logger.info(
            f"Information Coefficient: {trading_metrics['information_coefficient']:.4f} "
            f"(p={trading_metrics['ic_pvalue']:.4f})"
        )
        logger.info(
            f"Directional Accuracy: {trading_metrics.get('directional_accuracy', 0):.2%}"
        )
        logger.info(
            f"Simulated Sharpe (top 30%): {trading_metrics['simulated_sharpe_top']:.2f}"
        )
        logger.info(
            f"Annualized Return (top 30%): {trading_metrics['simulated_return_top']:.2%}"
        )
        if trading_metrics.get("precision_at_confidence") is not None:
            logger.info(
                f"Precision @ 60% conf: {trading_metrics['precision_at_confidence']:.2%} "
                f"(n={trading_metrics['n_high_confidence']})"
            )
    else:
        logger.warning(f"Column '{return_col}' not found, skipping trading metrics")
        results["trading_metrics"] = {}

    # 5. Baseline comparisons
    logger.info("\n" + "=" * 50)
    logger.info("Computing baseline comparisons...")
    train_labels = df_train[TARGET_COLUMN] if df_train is not None else None
    baselines = compute_baselines(eval_results["labels"], train_labels)
    results["baselines"] = baselines

    logger.info(f"Model Accuracy:      {results['accuracy']:.2%}")
    logger.info(
        f"Naive ({baselines['majority_class']}):  {baselines['naive_accuracy']:.2%}"
    )
    logger.info(f"Random:              {baselines['random_accuracy']:.2%}")
    logger.info(f"Weighted Random:     {baselines['weighted_random_accuracy']:.2%}")

    # Improvement metrics
    improvement_vs_naive = (
        results["accuracy"] - baselines["naive_accuracy"]
    ) / baselines["naive_accuracy"]
    improvement_vs_random = (
        results["accuracy"] - baselines["random_accuracy"]
    ) / baselines["random_accuracy"]
    results["improvement_vs_naive"] = improvement_vs_naive
    results["improvement_vs_random"] = improvement_vs_random

    logger.info(f"\nImprovement vs Naive:  {improvement_vs_naive:+.1%}")
    logger.info(f"Improvement vs Random: {improvement_vs_random:+.1%}")

    if results["accuracy"] <= baselines["naive_accuracy"]:
        logger.warning("Model performs WORSE than always predicting majority class!")
    elif improvement_vs_naive < 0.05:
        logger.warning("Model barely beats naive baseline (<5% improvement)")
    else:
        logger.info("Model shows meaningful improvement over baselines")

    # Save results
    if output_dir is not None:
        results_path = output_dir / "evaluation_results.json"
        # Convert numpy types for JSON serialization
        serializable_results = _make_serializable(results)
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"\nSaved evaluation results to {results_path}")

    return results


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert.

    Returns:
        Serializable version of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def evaluate(
    model_dir: Optional[Path] = None,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run evaluation on a trained model.

    Args:
        model_dir: Path to saved model directory.
        data_path: Path to enriched CSV data file.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary with evaluation results.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR / "final"
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if output_dir is None:
        output_dir = DEFAULT_MODEL_DIR / "evaluation"

    model_dir = Path(model_dir)
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and artifacts
    logger.info(f"Loading model from {model_dir}")
    model, scaler, encodings = load_model_for_evaluation(model_dir, device)

    # Load and prepare data
    logger.info(f"Loading data from {data_path}")
    df = load_enriched_data(data_path)
    df_reliable = filter_reliable(df)

    # Split data
    df_train, df_val, df_test = split_by_hash(df_reliable)
    logger.info(f"Test set: {len(df_test)} samples")

    # Create test dataset
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    test_dataset, _ = create_dataset_from_df(
        df_test,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        scaler=scaler,
        fit_scaler=False,
    )

    # Run evaluation
    results = run_full_evaluation(
        model=model,
        test_dataset=test_dataset,
        df_test=df_test,
        df_train=df_train,
        output_dir=output_dir,
        device=device,
    )

    return results


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained FinBERT tweet classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR / "final",
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to enriched CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR / "evaluation",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
