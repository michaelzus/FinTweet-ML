"""Evaluate weekly walk-forward models on their next trading week.

This script evaluates each trained model on the first 7 trading days
of its test period - simulating a real trading scenario where you
train weekly and trade the next week.

Usage:
    python scripts/evaluate_weekly_walk_forward.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.stats import spearmanr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import BertTokenizer

from tweet_classifier.config import (
    FINBERT_MODEL_NAME,
    LABEL_MAP,
    LABEL_MAP_INV,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable
from tweet_classifier.dataset import create_categorical_encodings, create_dataset_from_df
from tweet_classifier.model import FinBERTMultiModal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_walk_forward_summary(model_dir: Path) -> Dict:
    """Load the walk-forward summary with window information."""
    summary_path = model_dir / "weekly_walk_forward_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Filter dataframe to date range (inclusive)."""
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].copy()


def get_first_n_trading_days(df: pd.DataFrame, start_date, n_days: int = 7) -> pd.DataFrame:
    """Get samples from the first N trading days starting from start_date."""
    dates = sorted(df[df["date"] >= start_date]["date"].unique())
    if len(dates) < n_days:
        logger.warning(f"Only {len(dates)} trading days available, need {n_days}")
        end_date = dates[-1] if dates else start_date
    else:
        end_date = dates[n_days - 1]

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask].copy()


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    encodings: Dict,
    device: torch.device,
) -> FinBERTMultiModal:
    """Load model from checkpoint directory."""
    model = FinBERTMultiModal(
        num_numerical_features=len(NUMERICAL_FEATURES),
        num_authors=len(encodings["author_to_idx"]),
        num_categories=len(encodings["category_to_idx"]),
        num_market_regimes=len(encodings["market_regime_to_idx"]),
        num_sectors=len(encodings["sector_to_idx"]),
        num_market_caps=len(encodings["market_cap_to_idx"]),
        num_classes=len(LABEL_MAP),
        freeze_bert=False,
    )

    # Load weights
    weights_path = checkpoint_dir / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(weights_path, device=str(device))
    else:
        weights_path = checkpoint_dir / "pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def evaluate_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    df_test: pd.DataFrame,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run evaluation and compute all metrics."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    # Standard metrics
    results = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted")),
        "n_samples": len(labels),
    }

    # Trading metrics
    return_col = "return_to_next_open"
    if return_col in df_test.columns:
        actual_returns = df_test[return_col].values
        valid_mask = ~np.isnan(actual_returns)

        if valid_mask.sum() > 0:
            valid_predictions = predictions[valid_mask]
            valid_probabilities = probabilities[valid_mask]
            valid_returns = actual_returns[valid_mask]
            valid_labels = labels[valid_mask]

            # Information Coefficient
            buy_confidence = valid_probabilities[:, 2] - valid_probabilities[:, 0]
            ic, ic_pvalue = spearmanr(buy_confidence, valid_returns)
            results["information_coefficient"] = float(ic) if not np.isnan(ic) else 0.0
            results["ic_pvalue"] = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0

            # Directional accuracy (excluding HOLD)
            non_hold_mask = valid_predictions != 1
            if non_hold_mask.sum() > 0:
                predicted_direction = np.where(valid_predictions[non_hold_mask] == 2, 1, -1)
                actual_direction = np.sign(valid_returns[non_hold_mask])
                non_zero_returns = actual_direction != 0
                if non_zero_returns.sum() > 0:
                    directional_correct = (predicted_direction[non_zero_returns] == actual_direction[non_zero_returns]).sum()
                    results["directional_accuracy"] = float(directional_correct / non_zero_returns.sum())
                    results["n_directional"] = int(non_zero_returns.sum())
                else:
                    results["directional_accuracy"] = 0.0
                    results["n_directional"] = 0
            else:
                results["directional_accuracy"] = 0.0
                results["n_directional"] = 0

            # Simulated returns (top 30% confidence)
            n_top = max(1, int(0.3 * len(valid_predictions)))
            max_confidence = valid_probabilities.max(axis=1)
            top_indices = np.argsort(max_confidence)[-n_top:]

            simulated_returns = []
            for idx in top_indices:
                pred = valid_predictions[idx]
                ret = valid_returns[idx]
                if pred == 2:  # BUY
                    simulated_returns.append(ret - 0.001)
                elif pred == 0:  # SELL
                    simulated_returns.append(-ret - 0.001)
                else:
                    simulated_returns.append(0)

            simulated_returns = np.array(simulated_returns)
            if len(simulated_returns) > 0 and simulated_returns.std() > 0:
                sharpe = (simulated_returns.mean() / simulated_returns.std()) * np.sqrt(252)
                results["simulated_sharpe"] = float(sharpe)
                results["simulated_total_return"] = float(simulated_returns.sum())
            else:
                results["simulated_sharpe"] = 0.0
                results["simulated_total_return"] = 0.0

            results["n_top_trades"] = n_top

    # Per-class predictions
    for i, label_name in LABEL_MAP_INV.items():
        results[f"pred_{label_name}"] = int((predictions == i).sum())
        results[f"true_{label_name}"] = int((labels == i).sum())

    return results


def main():
    """Evaluate all weekly walk-forward models on their next trading week."""
    # Configuration
    model_dir = Path("models/weekly-walk-forward")
    data_path = Path("output/2025_enrich_2.csv")
    n_trading_days = 7  # Evaluate on first week

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

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Filter reliable labels
    df_reliable = filter_reliable(df)
    logger.info(f"Reliable samples: {len(df_reliable)}")

    # Load summary
    summary = load_walk_forward_summary(model_dir)
    windows = summary["windows"]

    logger.info(f"\n{'='*70}")
    logger.info("WEEKLY WALK-FORWARD EVALUATION - NEXT TRADING WEEK")
    logger.info(f"Evaluating {len(windows)} models on first {n_trading_days} trading days of test period")
    logger.info(f"{'='*70}\n")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)

    # Evaluate each window
    all_results = []

    for window_info in windows:
        window_num = window_info["window"]
        train_start = datetime.strptime(window_info["train_start"], "%Y-%m-%d").date()
        train_end = datetime.strptime(window_info["train_end"], "%Y-%m-%d").date()
        test_start = datetime.strptime(window_info["test_start"], "%Y-%m-%d").date()

        logger.info(f"\n{'='*60}")
        logger.info(f"Window {window_num}")
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Evaluating on: {n_trading_days} trading days from {test_start}")
        logger.info("=" * 60)

        # Get training data to recreate encodings
        df_train = filter_by_date_range(df_reliable, train_start, train_end)
        logger.info(f"Training data: {len(df_train)} samples")

        # Get first week of test data
        df_test_week = get_first_n_trading_days(df_reliable, test_start, n_trading_days)
        test_dates = sorted(df_test_week["date"].unique())
        test_end = test_dates[-1] if test_dates else test_start
        logger.info(f"Test week: {test_start} to {test_end} ({len(test_dates)} days, {len(df_test_week)} samples)")

        if len(df_test_week) < 10:
            logger.warning(f"Skipping window {window_num} - insufficient test data")
            continue

        # Create encodings from training data
        encodings = create_categorical_encodings(df_train)

        # Create datasets
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

        test_dataset, _ = create_dataset_from_df(
            df_test_week,
            tokenizer,
            encodings["author_to_idx"],
            encodings["category_to_idx"],
            encodings["market_regime_to_idx"],
            encodings["sector_to_idx"],
            encodings["market_cap_to_idx"],
            scaler=scaler,
            fit_scaler=False,
        )

        # Find checkpoint directory
        window_dir = model_dir / f"window_{window_num}"
        checkpoint_dirs = list(window_dir.glob("checkpoint-*"))
        if not checkpoint_dirs:
            logger.warning(f"No checkpoint found for window {window_num}")
            continue
        checkpoint_dir = checkpoint_dirs[0]
        logger.info(f"Loading model from {checkpoint_dir.name}")

        # Load model
        model = load_model_from_checkpoint(checkpoint_dir, encodings, device)

        # Evaluate
        results = evaluate_model(model, test_dataset, df_test_week, device)

        # Add window metadata
        results["window"] = window_num
        results["train_start"] = str(train_start)
        results["train_end"] = str(train_end)
        results["test_start"] = str(test_start)
        results["test_end"] = str(test_end)
        results["n_trading_days"] = len(test_dates)

        all_results.append(results)

        # Log results
        logger.info(f"\nWindow {window_num} Results:")
        logger.info(f"  Samples: {results['n_samples']}")
        logger.info(f"  Accuracy: {results['accuracy']*100:.2f}%")
        logger.info(f"  F1 Macro: {results['f1_macro']*100:.2f}%")
        logger.info(f"  IC: {results.get('information_coefficient', 0):.4f} (p={results.get('ic_pvalue', 1):.4f})")
        logger.info(f"  Directional Acc: {results.get('directional_accuracy', 0)*100:.2f}%")
        logger.info(f"  Sharpe: {results.get('simulated_sharpe', 0):.2f}")
        logger.info(f"  Total Return: {results.get('simulated_total_return', 0)*100:.2f}%")

        # Clean up model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY - NEXT WEEK EVALUATION")
    logger.info(f"{'='*70}")

    if all_results:
        avg_accuracy = np.mean([r["accuracy"] for r in all_results])
        avg_f1 = np.mean([r["f1_macro"] for r in all_results])
        avg_ic = np.mean([r.get("information_coefficient", 0) for r in all_results])
        avg_dir_acc = np.mean([r.get("directional_accuracy", 0) for r in all_results])
        avg_sharpe = np.mean([r.get("simulated_sharpe", 0) for r in all_results])
        total_return = sum([r.get("simulated_total_return", 0) for r in all_results])

        ic_values = [r.get("information_coefficient", 0) for r in all_results]
        ic_pvalues = [r.get("ic_pvalue", 1) for r in all_results]
        significant_ics = sum(1 for p in ic_pvalues if p < 0.05)

        logger.info(f"\nWindows evaluated: {len(all_results)}")
        logger.info(f"Avg Accuracy: {avg_accuracy*100:.2f}%")
        logger.info(f"Avg F1 Macro: {avg_f1*100:.2f}%")
        logger.info(f"Avg IC: {avg_ic:.4f} (std: {np.std(ic_values):.4f})")
        logger.info(f"Avg Directional Accuracy: {avg_dir_acc*100:.2f}%")
        logger.info(f"Avg Sharpe: {avg_sharpe:.2f}")
        logger.info(f"Cumulative Return (top 30%): {total_return*100:.2f}%")
        logger.info(f"Significant IC (p<0.05): {significant_ics}/{len(all_results)}")

        # Best and worst
        best_ic = max(all_results, key=lambda x: x.get("information_coefficient", -999))
        worst_ic = min(all_results, key=lambda x: x.get("information_coefficient", 999))
        logger.info(f"\nBest IC: Window {best_ic['window']} (IC={best_ic.get('information_coefficient', 0):.4f})")
        logger.info(f"Worst IC: Window {worst_ic['window']} (IC={worst_ic.get('information_coefficient', 0):.4f})")

        # Save results
        output_path = model_dir / "next_week_evaluation.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "windows": all_results,
                    "summary": {
                        "n_windows": len(all_results),
                        "n_trading_days_per_window": n_trading_days,
                        "avg_accuracy": float(avg_accuracy),
                        "avg_f1_macro": float(avg_f1),
                        "avg_ic": float(avg_ic),
                        "avg_directional_accuracy": float(avg_dir_acc),
                        "avg_sharpe": float(avg_sharpe),
                        "cumulative_return": float(total_return),
                        "significant_ic_count": significant_ics,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
