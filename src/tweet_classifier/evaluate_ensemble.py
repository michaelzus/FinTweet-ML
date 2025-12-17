"""Ensemble evaluation script for averaging predictions from multiple models.

This script loads multiple trained models, averages their predictions,
and evaluates the ensemble on the test set.

Usage:
    python -m tweet_classifier.evaluate_ensemble --data-path output/test2_entry_fix.csv
    python -m tweet_classifier.evaluate_ensemble --model-dirs models/run-1/final models/run-3/final
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import BertTokenizer

from tweet_classifier.config import (
    DEFAULT_DATA_PATH,
    FINBERT_MODEL_NAME,
    LABEL_MAP_INV,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.data.splitter import split_by_hash
from tweet_classifier.dataset import create_dataset_from_df
from tweet_classifier.ensemble import (
    DEFAULT_ENSEMBLE_MODELS,
    compute_ensemble_trading_metrics,
    evaluate_ensemble,
    load_ensemble,
)
from tweet_classifier.evaluate import compute_baselines, generate_classification_report, plot_confusion_matrix

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_ensemble_evaluation(
    data_path: Path,
    model_dirs: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full evaluation on ensemble of models.

    Args:
        data_path: Path to enriched CSV data.
        model_dirs: List of model directories. If None, uses DEFAULT_ENSEMBLE_MODELS.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary with evaluation results.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load ensemble
    logger.info("Loading ensemble models...")
    models, scaler, encodings = load_ensemble(model_dirs, device)
    logger.info(f"Loaded {len(models)} models")

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = load_enriched_data(data_path)
    df_reliable = filter_reliable(df)
    logger.info(f"Reliable samples: {len(df_reliable)}")

    # Split data (use same split as training)
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

    # Evaluate ensemble
    logger.info("Running ensemble evaluation...")
    eval_results = evaluate_ensemble(models, test_dataset, device)

    results: Dict[str, Any] = {
        "n_models": len(models),
        "accuracy": eval_results["accuracy"],
        "f1_macro": eval_results["f1_macro"],
        "f1_weighted": eval_results["f1_weighted"],
    }

    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Number of models: {len(models)}")
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"Test F1 (weighted): {results['f1_weighted']:.4f}")

    # Classification report
    logger.info("\n" + "=" * 50)
    logger.info("Classification Report:")
    report = generate_classification_report(eval_results["labels"], eval_results["predictions"])
    logger.info("\n" + report)
    results["classification_report"] = report

    # Confusion matrix
    if output_dir is not None:
        cm_path = output_dir / "ensemble_confusion_matrix.png"
        plot_confusion_matrix(eval_results["labels"], eval_results["predictions"], cm_path)
        results["confusion_matrix_path"] = str(cm_path)

    # Trading metrics
    return_col = "return_to_next_open"
    if return_col in df_test.columns:
        logger.info("\n" + "=" * 50)
        logger.info("Trading Metrics (Ensemble):")
        trading_metrics = compute_ensemble_trading_metrics(
            predictions=eval_results["predictions"],
            probabilities=eval_results["probabilities"],
            actual_returns=df_test[return_col].values,
            labels=eval_results["labels"],
        )
        results["trading_metrics"] = trading_metrics

        logger.info(f"Information Coefficient: {trading_metrics['information_coefficient']:.4f} "
                    f"(p={trading_metrics['ic_pvalue']:.4f})")
        ic_significant = "YES ✓" if trading_metrics['ic_pvalue'] < 0.05 else "NO ✗"
        logger.info(f"IC Statistically Significant: {ic_significant}")
        logger.info(f"Directional Accuracy: {trading_metrics.get('directional_accuracy', 0):.2%}")
        logger.info(f"Simulated Sharpe (top 30%): {trading_metrics['simulated_sharpe_top']:.2f}")
        logger.info(f"Annualized Return (top 30%): {trading_metrics['simulated_return_top']:.2%}")

    # Baselines
    logger.info("\n" + "=" * 50)
    logger.info("Baseline Comparisons:")
    baselines = compute_baselines(eval_results["labels"])
    results["baselines"] = baselines

    logger.info(f"Ensemble Accuracy:   {results['accuracy']:.2%}")
    logger.info(f"Naive ({baselines['majority_class']}):  {baselines['naive_accuracy']:.2%}")
    logger.info(f"Random:              {baselines['random_accuracy']:.2%}")

    improvement_vs_naive = (results["accuracy"] - baselines["naive_accuracy"]) / baselines["naive_accuracy"]
    results["improvement_vs_naive"] = improvement_vs_naive
    logger.info(f"\nImprovement vs Naive:  {improvement_vs_naive:+.1%}")

    # Save results
    if output_dir is not None:
        results_path = output_dir / "ensemble_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(_make_serializable(results), f, indent=2)
        logger.info(f"\nSaved results to {results_path}")

    return results


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
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


def main() -> None:
    """CLI entry point for ensemble evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble of trained FinBERT tweet classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to enriched CSV data file",
    )
    parser.add_argument(
        "--model-dirs",
        type=str,
        nargs="+",
        default=None,
        help="List of model directories (default: best 6 from consistency test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/ensemble"),
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    run_ensemble_evaluation(
        data_path=args.data_path,
        model_dirs=args.model_dirs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

