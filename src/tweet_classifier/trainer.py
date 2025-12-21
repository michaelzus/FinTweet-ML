"""Custom Trainer with class-weighted loss for tweet classification.

This module provides:
- WeightedTrainer: HuggingFace Trainer with class-weighted cross-entropy loss
- compute_metrics: Evaluation metrics function for accuracy and F1 scores
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer


class WeightedTrainer(Trainer):
    """Custom Trainer that uses class-weighted cross-entropy loss.

    This trainer is designed for imbalanced classification tasks where
    minority classes should have higher importance in the loss function.
    """

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any):
        """Initialize the WeightedTrainer.

        Args:
            class_weights: Tensor of shape (num_classes,) with weight for each class.
                           Higher weights give more importance to that class in the loss.
            *args: Positional arguments passed to parent Trainer.
            **kwargs: Keyword arguments passed to parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute class-weighted cross-entropy loss.

        Args:
            model: The model being trained.
            inputs: Dictionary containing model inputs and labels.
            return_outputs: If True, return (loss, outputs) tuple.
            num_items_in_batch: Number of items in batch for gradient accumulation.

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True.
        """
        labels = inputs.pop("labels")

        # Forward pass without labels to get logits
        outputs = model(**inputs)
        logits = outputs["logits"]

        # Compute weighted cross-entropy loss
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
            reduction=reduction,
        )

        # Normalize by batch size if using gradient accumulation
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics for classification.

    Args:
        eval_pred: Tuple of (predictions, labels) where predictions are logits
                   of shape (n_samples, n_classes) and labels are integers.

    Returns:
        Dictionary with accuracy, macro F1, and weighted F1 scores.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
