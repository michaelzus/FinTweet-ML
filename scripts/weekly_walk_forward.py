"""Weekly walk-forward validation with daily-sliding monthly windows.

This script implements rolling window training and evaluation:
- Train on ~120 days (~4 months)
- Validate on ~30 days (~1 month)
- Test on ~30 days (~1 month)
- Slide window forward by 7 days (weekly)

Usage:
    python scripts/weekly_walk_forward.py --data-path output/2025_enrich_2.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import BertTokenizer, TrainingArguments

from tweet_classifier.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    FINBERT_MODEL_NAME,
    LABEL_MAP,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable
from tweet_classifier.data.weights import compute_class_weights, weights_to_tensor
from tweet_classifier.dataset import create_categorical_encodings, create_dataset_from_df
from tweet_classifier.model import FinBERTMultiModal
from tweet_classifier.trainer import WeightedTrainer, compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Weekly walk-forward validation")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to enriched CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weekly-walk-forward",
        help="Output directory for models and results",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=120,
        help="Number of days for training (~4 months)",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=30,
        help="Number of days for validation (~1 month)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Number of days for testing (~1 month)",
    )
    parser.add_argument(
        "--slide-days",
        type=int,
        default=7,
        help="Number of days to slide window (7 = weekly)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    return parser.parse_args()


def get_date_ranges(
    dates: List,
    train_days: int,
    val_days: int,
    test_days: int,
    slide_days: int,
) -> List[Dict]:
    """Generate date ranges for each window.

    Args:
        dates: Sorted list of unique dates in the data.
        train_days: Number of days for training.
        val_days: Number of days for validation.
        test_days: Number of days for testing.
        slide_days: Number of days to slide window.

    Returns:
        List of window info dictionaries.
    """
    total_days = train_days + val_days + test_days
    windows = []

    i = 0
    window_num = 1
    while i + total_days <= len(dates):
        train_start = dates[i]
        train_end = dates[i + train_days - 1]
        val_start = dates[i + train_days]
        val_end = dates[i + train_days + val_days - 1]
        test_start = dates[i + train_days + val_days]
        test_end = dates[i + total_days - 1]

        windows.append({
            "window": window_num,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        i += slide_days
        window_num += 1

    return windows


def filter_by_date_range(
    df: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Filter dataframe to date range.

    Args:
        df: DataFrame with 'date' column.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).

    Returns:
        Filtered DataFrame.
    """
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].copy()


def compute_trading_metrics(
    predictions: np.ndarray,
    returns: np.ndarray,
    probabilities: np.ndarray,
    transaction_cost: float = 0.001,
) -> Dict:
    """Compute trading-specific metrics.

    Args:
        predictions: Array of predicted classes (0=SELL, 1=HOLD, 2=BUY).
        returns: Array of actual returns.
        probabilities: Array of prediction probabilities (N x 3).
        transaction_cost: Transaction cost per trade.

    Returns:
        Dictionary of trading metrics.
    """
    # Filter valid entries
    valid_mask = ~np.isnan(returns)
    valid_predictions = predictions[valid_mask]
    valid_returns = returns[valid_mask]
    valid_probs = probabilities[valid_mask]

    if len(valid_returns) == 0:
        return {}

    # Information Coefficient
    pred_direction = np.where(valid_predictions == 2, 1, np.where(valid_predictions == 0, -1, 0))
    ic, ic_pvalue = stats.spearmanr(pred_direction, valid_returns)

    # Directional accuracy (excluding HOLD)
    non_hold_mask = valid_predictions != 1
    if non_hold_mask.sum() > 0:
        pred_dir = pred_direction[non_hold_mask]
        actual_dir = np.sign(valid_returns[non_hold_mask])
        directional_acc = (pred_dir == actual_dir).mean()
        n_directional = non_hold_mask.sum()
    else:
        directional_acc = 0.0
        n_directional = 0

    # Simulated returns for top predictions
    max_probs = valid_probs.max(axis=1)
    top_count = max(1, int(len(max_probs) * 0.3))
    top_indices = np.argsort(max_probs)[::-1][:top_count]

    simulated_returns = []
    for idx in top_indices:
        pred = valid_predictions[idx]
        ret = valid_returns[idx]
        if pred == 2:  # BUY
            simulated_returns.append(ret - transaction_cost)
        elif pred == 0:  # SELL
            simulated_returns.append(-ret - transaction_cost)
        else:
            simulated_returns.append(0)

    simulated_returns = np.array(simulated_returns)
    sharpe = np.mean(simulated_returns) / (np.std(simulated_returns) + 1e-8) * np.sqrt(252)
    total_return = np.sum(simulated_returns)

    # Precision at confidence threshold
    conf_threshold = 0.6
    high_conf_mask = max_probs >= conf_threshold
    if high_conf_mask.sum() > 0:
        high_conf_preds = valid_predictions[high_conf_mask]
        high_conf_returns = valid_returns[high_conf_mask]
        correct = ((high_conf_preds == 2) & (high_conf_returns > 0)) | \
                  ((high_conf_preds == 0) & (high_conf_returns < 0))
        precision_at_conf = correct.mean()
        n_high_conf = high_conf_mask.sum()
    else:
        precision_at_conf = 0.0
        n_high_conf = 0

    return {
        "information_coefficient": float(ic) if not np.isnan(ic) else 0.0,
        "ic_pvalue": float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0,
        "directional_accuracy": float(directional_acc),
        "n_directional_predictions": int(n_directional),
        "simulated_sharpe_top": float(sharpe),
        "simulated_return_top": float(total_return),
        "n_top_trades": len(top_indices),
        "n_high_confidence": int(n_high_conf),
        "precision_at_confidence": float(precision_at_conf),
        "confidence_threshold": conf_threshold,
    }


def train_and_evaluate_window(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    window_info: Dict,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """Train model on window and evaluate.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        window_info: Window metadata.
        output_dir: Output directory for this window.
        args: Command line arguments.
        device: Torch device.

    Returns:
        Dictionary of results.
    """
    window_num = window_info["window"]
    logger.info(f"Window {window_num}: Train {len(df_train)}, Val {len(df_val)}, Test {len(df_test)}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)

    # Create categorical encodings from training data
    encodings = create_categorical_encodings(df_train)

    # Create datasets with proper categorical encodings
    train_dataset, scaler = create_dataset_from_df(
        df_train,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        fit_scaler=True,
    )

    val_dataset, _ = create_dataset_from_df(
        df_val,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],
        encodings["sector_to_idx"],
        encodings["market_cap_to_idx"],
        scaler=scaler,
        fit_scaler=False,
    )

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

    # Initialize model with correct parameters
    model = FinBERTMultiModal(
        num_numerical_features=len(NUMERICAL_FEATURES),
        num_authors=len(encodings["author_to_idx"]),
        num_categories=len(encodings["category_to_idx"]),
        num_market_regimes=len(encodings["market_regime_to_idx"]),
        num_sectors=len(encodings["sector_to_idx"]),
        num_market_caps=len(encodings["market_cap_to_idx"]),
        num_classes=len(LABEL_MAP),
        freeze_bert=False,  # Fine-tuning is essential for performance
        dropout=DEFAULT_DROPOUT,
    )
    model.to(device)

    # Compute class weights
    class_weights = compute_class_weights(df_train[TARGET_COLUMN])
    class_weights_tensor = weights_to_tensor(class_weights).to(device)

    # Training arguments
    window_output_dir = output_dir / f"window_{window_num}"
    use_fp16 = torch.cuda.is_available()
    use_pin_memory = not torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=str(window_output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        logging_dir=str(window_output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=use_fp16,
        dataloader_pin_memory=use_pin_memory,
        remove_unused_columns=False,
    )

    # Create trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    # Get predictions for trading metrics
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions.argmax(axis=1)
    probabilities = torch.softmax(torch.tensor(predictions_output.predictions), dim=1).numpy()

    # Get returns
    returns = df_test["return_to_next_open"].values

    # Compute trading metrics
    trading_metrics = compute_trading_metrics(predictions, returns, probabilities)

    # Combine results
    results = {
        "window": window_num,
        "train_start": str(window_info["train_start"]),
        "train_end": str(window_info["train_end"]),
        "val_start": str(window_info["val_start"]),
        "val_end": str(window_info["val_end"]),
        "test_start": str(window_info["test_start"]),
        "test_end": str(window_info["test_end"]),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "accuracy": test_results["eval_accuracy"],
        "f1_macro": test_results["eval_f1_macro"],
        "f1_weighted": test_results["eval_f1_weighted"],
        "val_accuracy": trainer.state.best_metric,
        **trading_metrics,
    }

    return results


def main():
    """Run weekly walk-forward validation."""
    args = parse_args()

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Filter reliable labels
    df_reliable = filter_reliable(df)
    logger.info(f"Reliable samples: {len(df_reliable)}")

    # Get unique dates
    dates = sorted(df_reliable["date"].unique())
    logger.info(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Generate windows
    windows = get_date_ranges(
        dates,
        args.train_days,
        args.val_days,
        args.test_days,
        args.slide_days,
    )
    logger.info(f"Total windows: {len(windows)}")

    if len(windows) == 0:
        logger.error(
            f"No windows can be created. Need at least {args.train_days + args.val_days + args.test_days} days, "
            f"but only have {len(dates)} days."
        )
        return

    # Run walk-forward validation
    all_results = []

    for window_info in windows:
        window_num = window_info["window"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Window {window_num}/{len(windows)}")
        logger.info(f"Train: {window_info['train_start']} to {window_info['train_end']}")
        logger.info(f"Val:   {window_info['val_start']} to {window_info['val_end']}")
        logger.info(f"Test:  {window_info['test_start']} to {window_info['test_end']}")
        logger.info("=" * 60)

        # Split data
        df_train = filter_by_date_range(df_reliable, window_info["train_start"], window_info["train_end"])
        df_val = filter_by_date_range(df_reliable, window_info["val_start"], window_info["val_end"])
        df_test = filter_by_date_range(df_reliable, window_info["test_start"], window_info["test_end"])

        if len(df_train) < 100 or len(df_val) < 50 or len(df_test) < 50:
            logger.warning(f"Skipping window {window_num} - insufficient data")
            continue

        # Train and evaluate
        try:
            results = train_and_evaluate_window(
                df_train, df_val, df_test,
                window_info, output_dir, args, device,
            )
            all_results.append(results)

            # Log results
            logger.info(f"\nWindow {window_num} Results:")
            logger.info(f"  Accuracy: {results['accuracy']*100:.2f}%")
            logger.info(f"  F1 Macro: {results['f1_macro']*100:.2f}%")
            logger.info(f"  IC: {results.get('information_coefficient', 0):.4f} (p={results.get('ic_pvalue', 1):.4f})")
            logger.info(f"  Directional Acc: {results.get('directional_accuracy', 0)*100:.2f}%")
            logger.info(f"  Sharpe: {results.get('simulated_sharpe_top', 0):.2f}")

            # Save intermediate results
            summary_path = output_dir / "weekly_walk_forward_summary.json"
            _save_summary(all_results, args, summary_path)

        except Exception as e:
            logger.error(f"Error in window {window_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 60)

    if all_results:
        _print_summary(all_results)

        # Save final results
        summary_path = output_dir / "weekly_walk_forward_summary.json"
        _save_summary(all_results, args, summary_path)
        logger.info(f"\nSaved summary to {summary_path}")


def _save_summary(all_results: List[Dict], args: argparse.Namespace, summary_path: Path) -> None:
    """Save summary to JSON file."""
    avg_accuracy = np.mean([r["accuracy"] for r in all_results])
    avg_f1 = np.mean([r["f1_macro"] for r in all_results])
    avg_ic = np.mean([r.get("information_coefficient", 0) for r in all_results])
    avg_sharpe = np.mean([r.get("simulated_sharpe_top", 0) for r in all_results])
    avg_dir_acc = np.mean([r.get("directional_accuracy", 0) for r in all_results])

    # IC significance analysis
    ic_values = [r.get("information_coefficient", 0) for r in all_results]
    ic_pvalues = [r.get("ic_pvalue", 1) for r in all_results]
    significant_ics = sum(1 for p in ic_pvalues if p < 0.05)

    summary = {
        "windows": all_results,
        "summary": {
            "num_windows": len(all_results),
            "avg_accuracy": float(avg_accuracy),
            "avg_f1_macro": float(avg_f1),
            "avg_ic": float(avg_ic),
            "avg_sharpe": float(avg_sharpe),
            "avg_directional_accuracy": float(avg_dir_acc),
            "ic_std": float(np.std(ic_values)),
            "significant_ic_count": significant_ics,
            "significant_ic_pct": float(significant_ics / len(all_results)) if all_results else 0.0,
            "train_days": args.train_days,
            "val_days": args.val_days,
            "test_days": args.test_days,
            "slide_days": args.slide_days,
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _print_summary(all_results: List[Dict]) -> None:
    """Print summary statistics."""
    avg_accuracy = np.mean([r["accuracy"] for r in all_results])
    avg_f1 = np.mean([r["f1_macro"] for r in all_results])
    avg_ic = np.mean([r.get("information_coefficient", 0) for r in all_results])
    avg_sharpe = np.mean([r.get("simulated_sharpe_top", 0) for r in all_results])
    avg_dir_acc = np.mean([r.get("directional_accuracy", 0) for r in all_results])

    ic_values = [r.get("information_coefficient", 0) for r in all_results]
    ic_pvalues = [r.get("ic_pvalue", 1) for r in all_results]
    significant_ics = sum(1 for p in ic_pvalues if p < 0.05)

    logger.info(f"Windows completed: {len(all_results)}")
    logger.info(f"Avg Accuracy: {avg_accuracy*100:.2f}%")
    logger.info(f"Avg F1 Macro: {avg_f1*100:.2f}%")
    logger.info(f"Avg IC: {avg_ic:.4f} (std: {np.std(ic_values):.4f})")
    logger.info(f"Avg Directional Accuracy: {avg_dir_acc*100:.2f}%")
    logger.info(f"Avg Sharpe: {avg_sharpe:.2f}")
    logger.info(f"Significant IC (p<0.05): {significant_ics}/{len(all_results)} ({100*significant_ics/len(all_results):.1f}%)")

    # Best and worst windows
    best_ic_window = max(all_results, key=lambda x: x.get("information_coefficient", -999))
    worst_ic_window = min(all_results, key=lambda x: x.get("information_coefficient", 999))
    logger.info(f"\nBest IC window: {best_ic_window['window']} (IC={best_ic_window.get('information_coefficient', 0):.4f}, test: {best_ic_window['test_start']} to {best_ic_window['test_end']})")
    logger.info(f"Worst IC window: {worst_ic_window['window']} (IC={worst_ic_window.get('information_coefficient', 0):.4f}, test: {worst_ic_window['test_start']} to {worst_ic_window['test_end']})")


if __name__ == "__main__":
    main()
