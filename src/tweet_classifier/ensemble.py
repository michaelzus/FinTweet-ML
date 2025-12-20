"""Ensemble module for averaging predictions from multiple trained models.

This module provides utilities for loading multiple trained models and
averaging their predictions to reduce variance and improve stability.

Usage:
    from tweet_classifier.ensemble import load_ensemble, ensemble_predict
    
    models = load_ensemble(["models/run-1/final", "models/run-3/final"])
    avg_probs = ensemble_predict(models, batch)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

from tweet_classifier.evaluate import load_model_for_evaluation
from tweet_classifier.model import FinBERTMultiModal

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Best models from 12-run consistency test (IC > 0.10, p < 0.01)
DEFAULT_ENSEMBLE_MODELS = [
    "models/consistency-test-run-1/final",
    "models/consistency-test-run-3/final",
    "models/consistency-test-run-6/final",
    "models/consistency-test-run-7/final",
    "models/consistency-test-run-8/final",
    "models/consistency-test-run-10/final",
]


def load_ensemble(
    model_dirs: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[FinBERTMultiModal], Any, Dict[str, Any]]:
    """Load multiple trained models for ensemble inference.

    Args:
        model_dirs: List of model directory paths. If None, uses DEFAULT_ENSEMBLE_MODELS.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Tuple of (list of models, scaler, encodings).
        Note: scaler and encodings are taken from the first model.
    """
    if model_dirs is None:
        model_dirs = DEFAULT_ENSEMBLE_MODELS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    scaler = None
    encodings = None

    for i, model_dir in enumerate(model_dirs):
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning(f"Model not found at {model_dir}, skipping...")
            continue

        logger.info(f"Loading model {i + 1}/{len(model_dirs)}: {model_dir}")
        model, model_scaler, model_encodings = load_model_for_evaluation(model_path, device)
        models.append(model)

        # Use scaler and encodings from first model
        if scaler is None:
            scaler = model_scaler
            encodings = model_encodings

    if len(models) == 0:
        raise ValueError("No models were loaded successfully")

    logger.info(f"Loaded {len(models)} models for ensemble")
    return models, scaler, encodings


def ensemble_predict_batch(
    models: List[FinBERTMultiModal],
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ensemble prediction on a single batch.

    Args:
        models: List of trained models.
        batch: Dictionary with input tensors (input_ids, attention_mask, etc.).
        device: Torch device for inference.

    Returns:
        Tuple of (averaged probabilities, predicted classes).
    """
    if device is None:
        device = next(models[0].parameters()).device

    all_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                numerical=batch["numerical"].to(device),
                author_idx=batch["author_idx"].to(device),
                category_idx=batch["category_idx"].to(device),
                market_regime_idx=batch["market_regime_idx"].to(device),
                sector_idx=batch["sector_idx"].to(device),
                market_cap_idx=batch["market_cap_idx"].to(device),
            )
            probs = F.softmax(outputs["logits"], dim=1)
            all_probs.append(probs.cpu().numpy())

    # Average probabilities across models
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    return avg_probs, predictions


def evaluate_ensemble(
    models: List[FinBERTMultiModal],
    test_dataset: torch.utils.data.Dataset,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate ensemble on test dataset.

    Args:
        models: List of trained models.
        test_dataset: Test dataset.
        device: Torch device for inference.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with evaluation results (predictions, probabilities, accuracy, etc.).
    """
    if device is None:
        device = next(models[0].parameters()).device

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_probabilities = []
    all_labels = []

    for batch in dataloader:
        avg_probs, predictions = ensemble_predict_batch(models, batch, device)
        all_predictions.extend(predictions)
        all_probabilities.extend(avg_probs)
        all_labels.extend(batch["labels"].numpy())

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


def compute_ensemble_trading_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    actual_returns: np.ndarray,
    labels: np.ndarray,
    transaction_cost: float = 0.001,
    top_pct: float = 0.3,
) -> Dict[str, Any]:
    """Compute trading metrics for ensemble predictions.

    Args:
        predictions: Ensemble predictions.
        probabilities: Averaged probabilities.
        actual_returns: Actual returns from data.
        labels: True labels.
        transaction_cost: Cost per trade.
        top_pct: Percentage of top-confidence trades to consider.

    Returns:
        Dictionary with trading metrics (IC, directional accuracy, Sharpe, etc.).
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

    # 1. Information Coefficient (IC)
    buy_confidence = valid_probabilities[:, 2] - valid_probabilities[:, 0]  # BUY - SELL
    ic, ic_pvalue = spearmanr(buy_confidence, valid_returns)
    results["information_coefficient"] = float(ic) if not np.isnan(ic) else 0.0
    results["ic_pvalue"] = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0

    # 2. Directional Accuracy (ignoring HOLD predictions)
    non_hold_mask = valid_predictions != 1
    if non_hold_mask.sum() > 0:
        predicted_direction = np.where(valid_predictions[non_hold_mask] == 2, 1, -1)
        actual_direction = np.sign(valid_returns[non_hold_mask])
        non_zero_returns = actual_direction != 0
        if non_zero_returns.sum() > 0:
            directional_correct = (predicted_direction[non_zero_returns] == actual_direction[non_zero_returns]).sum()
            results["directional_accuracy"] = float(directional_correct / non_zero_returns.sum())
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

    return results



