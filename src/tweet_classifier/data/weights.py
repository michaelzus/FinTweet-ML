"""Class weight computation for imbalanced classification."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from tweet_classifier.config import LABEL_MAP, NUM_CLASSES


def compute_class_weights(
    labels: Union[pd.Series, np.ndarray],
    method: str = "balanced",
) -> np.ndarray:
    """Compute class weights for imbalanced classification.

    Args:
        labels: Array-like of string labels (SELL, HOLD, BUY) or integer labels.
        method: Weight computation method. Options:
            - 'balanced': sklearn balanced weights (inversely proportional to frequency)
            - 'sqrt': Square root of balanced weights (less aggressive)

    Returns:
        numpy array of class weights indexed by class (0=SELL, 1=HOLD, 2=BUY).
    """
    # Convert string labels to integers if needed
    if isinstance(labels, pd.Series):
        if labels.dtype == object:
            y = labels.map(LABEL_MAP).values
        else:
            y = labels.values
    else:
        if isinstance(labels[0], str):
            y = np.array([LABEL_MAP[label] for label in labels])
        else:
            y = np.array(labels)

    # Compute balanced weights
    classes = np.array(list(range(NUM_CLASSES)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    if method == "sqrt":
        # Less aggressive weighting
        weights = np.sqrt(weights)
        # Renormalize so mean weight is 1
        weights = weights / weights.mean()

    return weights


def weights_to_tensor(weights: np.ndarray):
    """Convert numpy weights to PyTorch tensor.

    Args:
        weights: numpy array of class weights.

    Returns:
        PyTorch tensor of weights.
    """
    import torch

    return torch.tensor(weights, dtype=torch.float32)


def apply_buy_boost(weights: np.ndarray, boost: float) -> np.ndarray:
    """Apply a multiplicative boost to BUY class weight.

    This is useful when BUY recall is too low - increasing BUY weight
    makes the model more aggressive at predicting BUY at the cost of precision.

    Args:
        weights: Array of class weights [SELL, HOLD, BUY].
        boost: Multiplier for BUY weight (e.g., 1.4 = 40% increase).

    Returns:
        Modified weights array with boosted BUY weight.
    """
    boosted = weights.copy()
    boosted[2] *= boost  # Index 2 = BUY
    return boosted


def get_weight_summary(
    labels: Union[pd.Series, np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Generate summary of class weights.

    Args:
        labels: Array-like of labels.
        weights: Pre-computed weights array. If None, computes from labels.

    Returns:
        Dictionary with weight information.
    """
    if weights is None:
        weights = compute_class_weights(labels)

    # Convert labels to integers for counting
    if isinstance(labels, pd.Series):
        if labels.dtype == object:
            y = labels.map(LABEL_MAP).values
        else:
            y = labels.values
    else:
        if isinstance(labels[0], str):
            y = np.array([LABEL_MAP[label] for label in labels])
        else:
            y = np.array(labels)

    # Count each class
    unique, counts = np.unique(y, return_counts=True)

    summary = {
        "SELL": {"count": 0, "weight": weights[0], "class_id": 0},
        "HOLD": {"count": 0, "weight": weights[1], "class_id": 1},
        "BUY": {"count": 0, "weight": weights[2], "class_id": 2},
    }

    for cls, count in zip(unique, counts):
        label_name = {0: "SELL", 1: "HOLD", 2: "BUY"}[cls]
        summary[label_name]["count"] = int(count)

    return summary
