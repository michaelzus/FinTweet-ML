"""Data utilities for tweet classification.

Provides functions for loading, filtering, splitting, and preparing data.
"""

from tweet_classifier.data.loader import (
    filter_reliable,
    get_data_summary,
    load_enriched_data,
    prepare_features,
)
from tweet_classifier.data.splitter import (
    get_split_summary,
    split_by_hash,
    verify_no_leakage,
)
from tweet_classifier.data.weights import (
    compute_class_weights,
    get_weight_summary,
    weights_to_tensor,
)

__all__ = [
    # Loader
    "load_enriched_data",
    "filter_reliable",
    "prepare_features",
    "get_data_summary",
    # Splitter
    "split_by_hash",
    "get_split_summary",
    "verify_no_leakage",
    # Weights
    "compute_class_weights",
    "get_weight_summary",
    "weights_to_tensor",
]
