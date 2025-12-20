"""Walk-forward validation for tweet classifier.

This script implements rolling window training and evaluation:
- Train on 4 months
- Validate on 1 month  
- Test on 1 month
- Slide window forward by 1 month

Example windows for 2025:
  Window 1: Train [01-04], Val [05], Test [06]
  Window 2: Train [02-05], Val [06], Test [07]
  ...
  Window 7: Train [07-10], Val [11], Test [12]

Usage:
    python scripts/walk_forward_validation.py --data-path output/2025_enrich_2.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from transformers import BertTokenizer, TrainingArguments

from tweet_classifier.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    FINBERT_MODEL_NAME,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.data.weights import compute_class_weights, weights_to_tensor
from tweet_classifier.dataset import (
    create_categorical_encodings,
    create_dataset_from_df,
    save_preprocessing_artifacts,
)
from tweet_classifier.evaluate import evaluate_on_test, compute_trading_metrics
from tweet_classifier.model import FinBERTMultiModal
from tweet_classifier.trainer import WeightedTrainer, compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_windows(start_month: int = 1, end_month: int = 12, train_months: int = 4) -> List[Tuple[List[int], int, int]]:
    """Generate walk-forward windows.

    Args:
        start_month: First month of data (1-12).
        end_month: Last month of data (1-12).
        train_months: Number of months for training.

    Returns:
        List of tuples: (train_months_list, val_month, test_month).
    """
    windows = []
    for test_month in range(start_month + train_months + 1, end_month + 1):
        val_month = test_month - 1
        train_month_list = list(range(test_month - train_months - 1, test_month - 1))
        windows.append((train_month_list, val_month, test_month))
    return windows


def split_by_month(df: pd.DataFrame, train_months: List[int], val_month: int, test_month: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by month.

    Args:
        df: DataFrame with 'month' column.
        train_months: List of months for training.
        val_month: Month for validation.
        test_month: Month for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df_train = df[df["month"].isin(train_months)].copy()
    df_val = df[df["month"] == val_month].copy()
    df_test = df[df["month"] == test_month].copy()
    return df_train, df_val, df_test


def train_and_evaluate_window(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    num_epochs: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dropout: float = DEFAULT_DROPOUT,
) -> Dict:
    """Train model on one window and evaluate.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        output_dir: Directory to save model and results.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        dropout: Dropout rate.

    Returns:
        Dictionary with evaluation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Create encodings from training data
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    encodings = create_categorical_encodings(df_train)

    # Create datasets
    train_dataset, scaler = create_dataset_from_df(
        df_train, tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        fit_scaler=True,
    )

    val_dataset, _ = create_dataset_from_df(
        df_val, tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        scaler=scaler,
        fit_scaler=False,
    )

    test_dataset, _ = create_dataset_from_df(
        df_test, tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        scaler=scaler,
        fit_scaler=False,
    )

    # Compute class weights
    class_weights = compute_class_weights(df_train[TARGET_COLUMN])
    class_weights_tensor = weights_to_tensor(class_weights).to(device)

    # Initialize model
    model = FinBERTMultiModal(
        num_numerical_features=len(NUMERICAL_FEATURES),
        num_authors=len(encodings["author_to_idx"]),
        num_categories=len(encodings["category_to_idx"]),
        num_market_regimes=len(encodings["market_regime_to_idx"]),
        num_sectors=len(encodings["sector_to_idx"]),
        num_market_caps=len(encodings["market_cap_to_idx"]),
        num_classes=3,
        freeze_bert=False,
        dropout=dropout,
    )
    model.to(device)

    # Training arguments
    use_fp16 = torch.cuda.is_available()
    use_pin_memory = not torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=use_pin_memory,
    )

    # Train
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    eval_results = evaluate_on_test(model, test_dataset, device=device)

    # Trading metrics
    trading_metrics = {}
    if "return_to_next_open" in df_test.columns:
        trading_metrics = compute_trading_metrics(
            predictions=eval_results["predictions"],
            probabilities=eval_results["probabilities"],
            actual_returns=df_test["return_to_next_open"].values,
            labels=eval_results["labels"],
        )

    results = {
        "accuracy": eval_results["accuracy"],
        "f1_macro": eval_results["f1_macro"],
        "f1_weighted": eval_results["f1_weighted"],
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        **trading_metrics,
    }

    return results


def run_walk_forward(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 5,
    train_months: int = 4,
) -> List[Dict]:
    """Run walk-forward validation across all windows.

    Args:
        data_path: Path to enriched CSV data.
        output_dir: Base directory for outputs.
        num_epochs: Epochs per window.
        train_months: Number of training months per window.

    Returns:
        List of results for each window.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = load_enriched_data(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.month

    # Filter reliable
    df_reliable = filter_reliable(df)
    logger.info(f"Total reliable samples: {len(df_reliable)}")

    # Check available months
    available_months = sorted([int(m) for m in df_reliable["month"].unique()])
    logger.info(f"Available months: {available_months}")

    # Generate windows
    windows = get_windows(
        start_month=min(available_months),
        end_month=max(available_months),
        train_months=train_months,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD VALIDATION: {len(windows)} windows")
    logger.info(f"{'='*60}\n")

    all_results = []

    for i, (train_months_list, val_month, test_month) in enumerate(windows):
        window_name = f"window_{i+1}_train_{train_months_list[0]:02d}-{train_months_list[-1]:02d}_test_{test_month:02d}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {i+1}/{len(windows)}: Train [{train_months_list[0]:02d}-{train_months_list[-1]:02d}], Val [{val_month:02d}], Test [{test_month:02d}]")
        logger.info(f"{'='*60}")

        # Split data
        df_train, df_val, df_test = split_by_month(df_reliable, train_months_list, val_month, test_month)

        if len(df_train) < 100 or len(df_val) < 50 or len(df_test) < 50:
            logger.warning(f"Skipping window {i+1}: insufficient data")
            continue

        window_dir = output_dir / window_name

        # Train and evaluate
        results = train_and_evaluate_window(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            output_dir=window_dir,
            num_epochs=num_epochs,
        )

        results["window"] = i + 1
        results["train_months"] = f"{train_months_list[0]:02d}-{train_months_list[-1]:02d}"
        results["val_month"] = f"{val_month:02d}"
        results["test_month"] = f"{test_month:02d}"

        all_results.append(results)

        logger.info(f"\nWindow {i+1} Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.2%}")
        logger.info(f"  F1 Macro: {results['f1_macro']:.2%}")
        if "information_coefficient" in results:
            logger.info(f"  IC: {results['information_coefficient']:.4f}")
            logger.info(f"  Directional Acc: {results.get('directional_accuracy', 0):.2%}")
            logger.info(f"  Sharpe: {results.get('simulated_sharpe_top', 0):.2f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("WALK-FORWARD SUMMARY")
    logger.info(f"{'='*60}")

    if all_results:
        avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
        avg_f1 = sum(r["f1_macro"] for r in all_results) / len(all_results)
        avg_ic = sum(r.get("information_coefficient", 0) for r in all_results) / len(all_results)
        avg_sharpe = sum(r.get("simulated_sharpe_top", 0) for r in all_results) / len(all_results)

        logger.info(f"Windows completed: {len(all_results)}")
        logger.info(f"Avg Accuracy: {avg_accuracy:.2%}")
        logger.info(f"Avg F1 Macro: {avg_f1:.2%}")
        logger.info(f"Avg IC: {avg_ic:.4f}")
        logger.info(f"Avg Sharpe: {avg_sharpe:.2f}")

        # Save summary
        summary = {
            "windows": all_results,
            "summary": {
                "num_windows": len(all_results),
                "avg_accuracy": avg_accuracy,
                "avg_f1_macro": avg_f1,
                "avg_ic": avg_ic,
                "avg_sharpe": avg_sharpe,
            },
        }

        summary_path = output_dir / "walk_forward_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"\nSaved summary to {summary_path}")

    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for tweet classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to enriched CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/walk-forward"),
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per window",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=4,
        help="Number of months for training in each window",
    )

    args = parser.parse_args()

    run_walk_forward(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        train_months=args.train_months,
    )


if __name__ == "__main__":
    main()

