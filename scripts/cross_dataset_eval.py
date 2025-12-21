"""Cross-dataset evaluation: evaluate a model on a completely different dataset.

This script loads a model trained on one dataset (e.g., 180-day) and evaluates
it on a different dataset (e.g., 2025 data) to test temporal generalization.

Usage:
    python scripts/cross_dataset_eval.py \
        --model-dir models/180day-temporal-split/final \
        --data-path output/2025_enrich_2.csv \
        --output-dir evaluation/180day-on-2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from transformers import BertTokenizer

from tweet_classifier.config import FINBERT_MODEL_NAME
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.dataset import create_dataset_from_df
from tweet_classifier.evaluate import (
    load_model_for_evaluation,
    run_full_evaluation,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def cross_dataset_evaluate(
    model_dir: Path,
    data_path: Path,
    output_dir: Path,
) -> dict:
    """Evaluate a model on a completely different dataset.

    Args:
        model_dir: Path to trained model directory (with final/ subfolder).
        data_path: Path to CSV data file to evaluate on.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary with evaluation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"Using device: {device}")

    # Load model and artifacts
    logger.info(f"Loading model from {model_dir}")
    model, scaler, encodings = load_model_for_evaluation(model_dir, device)

    # Load evaluation data (NO SPLITTING - use all data as test)
    logger.info(f"Loading evaluation data from {data_path}")
    df = load_enriched_data(data_path)
    logger.info(f"Total samples loaded: {len(df)}")

    df_reliable = filter_reliable(df)
    logger.info(f"Reliable samples (after filtering): {len(df_reliable)}")

    if len(df_reliable) == 0:
        logger.error("No reliable samples found in the dataset!")
        return {}

    # Log date range
    if "timestamp" in df_reliable.columns:
        min_date = pd.to_datetime(df_reliable["timestamp"]).min()
        max_date = pd.to_datetime(df_reliable["timestamp"]).max()
        logger.info(f"Data date range: {min_date.date()} to {max_date.date()}")

    # Log label distribution
    label_col = "label_1d_3class"
    if label_col in df_reliable.columns:
        label_dist = df_reliable[label_col].value_counts(normalize=True)
        logger.info(f"Label distribution:\n{label_dist}")

    # Handle unknown categories (authors/categories not seen during training)
    df_eval = df_reliable.copy()
    unknown_authors = 0
    unknown_categories = 0

    if "author" in df_eval.columns:
        known_authors = set(encodings["author_to_idx"].keys())
        mask = ~df_eval["author"].isin(known_authors)
        unknown_authors = mask.sum()
        if unknown_authors > 0:
            df_eval.loc[mask, "author"] = "UNKNOWN"
            logger.warning(f"Found {unknown_authors} samples with unknown authors -> mapped to UNKNOWN")

    if "category" in df_eval.columns:
        known_categories = set(encodings["category_to_idx"].keys())
        mask = ~df_eval["category"].isin(known_categories)
        unknown_categories = mask.sum()
        if unknown_categories > 0:
            df_eval.loc[mask, "category"] = "UNKNOWN"
            logger.warning(f"Found {unknown_categories} samples with unknown categories -> mapped to UNKNOWN")

    # Create dataset using model's preprocessing artifacts
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    test_dataset, _ = create_dataset_from_df(
        df_eval,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        scaler=scaler,
        fit_scaler=False,  # Use existing scaler, don't refit
    )

    logger.info(f"Created test dataset with {len(test_dataset)} samples")

    # Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-DATASET EVALUATION")
    logger.info(f"Model: {model_dir}")
    logger.info(f"Eval data: {data_path}")
    logger.info("=" * 60 + "\n")

    results = run_full_evaluation(
        model=model,
        test_dataset=test_dataset,
        df_test=df_eval,
        df_train=None,  # No training data for baselines
        output_dir=output_dir,
        device=device,
    )

    # Add metadata to results
    results["cross_dataset_eval"] = {
        "model_dir": str(model_dir),
        "eval_data_path": str(data_path),
        "total_samples": len(df),
        "reliable_samples": len(df_reliable),
        "unknown_authors": unknown_authors,
        "unknown_categories": unknown_categories,
    }

    # Save extended results
    results_path = output_dir / "cross_dataset_results.json"
    with open(results_path, "w") as f:
        # Filter out non-serializable items
        serializable = {k: v for k, v in results.items() if k != "classification_report"}
        serializable["classification_report"] = results.get("classification_report", "")
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved cross-dataset results to {results_path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a different dataset (cross-dataset evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to trained model directory (e.g., models/180day-temporal-split/final)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to CSV data file to evaluate on (e.g., output/2025_enrich_2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/cross-dataset"),
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    cross_dataset_evaluate(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
