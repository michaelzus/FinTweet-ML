#!/usr/bin/env python3
"""Evaluate model metrics at various confidence thresholds.

This script analyzes how IC, directional accuracy, and trading metrics
improve when filtering to high-confidence predictions only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tweet_classifier.config import FINBERT_MODEL_NAME
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.data.splitter import split_by_time
from tweet_classifier.dataset import create_dataset_from_df
from tweet_classifier.evaluate import evaluate_on_test, load_model_for_evaluation


def compute_metrics_at_threshold(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    threshold: float,
    transaction_cost: float = 0.001,
) -> Dict[str, Any]:
    """Compute all metrics for predictions above confidence threshold."""
    max_conf = probabilities.max(axis=1)
    mask = max_conf >= threshold

    n_samples = mask.sum()
    total_samples = len(predictions)
    coverage = mask.mean()

    if n_samples < 20:
        return {
            "threshold": threshold,
            "n_samples": int(n_samples),
            "coverage": float(coverage),
            "insufficient_samples": True,
        }

    # Filter data
    filt_preds = predictions[mask]
    filt_labels = labels[mask]
    filt_probs = probabilities[mask]
    filt_returns = returns[mask]

    # Remove NaN returns
    valid_ret_mask = ~np.isnan(filt_returns)
    if valid_ret_mask.sum() < 20:
        return {
            "threshold": threshold,
            "n_samples": int(n_samples),
            "coverage": float(coverage),
            "insufficient_valid_returns": True,
        }

    valid_preds = filt_preds[valid_ret_mask]
    valid_labels = filt_labels[valid_ret_mask]
    valid_probs = filt_probs[valid_ret_mask]
    valid_returns = filt_returns[valid_ret_mask]

    results: Dict[str, Any] = {
        "threshold": threshold,
        "n_samples": int(n_samples),
        "n_valid_returns": int(valid_ret_mask.sum()),
        "coverage": float(coverage),
    }

    # 1. Classification Metrics
    results["accuracy"] = float(accuracy_score(filt_labels, filt_preds))
    results["f1_macro"] = float(f1_score(filt_labels, filt_preds, average="macro", zero_division=0))

    # Per-class accuracy
    for i, name in enumerate(["SELL", "HOLD", "BUY"]):
        class_mask = filt_labels == i
        if class_mask.sum() > 0:
            results[f"{name.lower()}_accuracy"] = float((filt_preds[class_mask] == i).mean())
            results[f"{name.lower()}_count"] = int(class_mask.sum())
        else:
            results[f"{name.lower()}_accuracy"] = None
            results[f"{name.lower()}_count"] = 0

    # 2. Information Coefficient (on filtered data)
    buy_confidence = valid_probs[:, 2] - valid_probs[:, 0]  # BUY - SELL prob
    ic, ic_pvalue = spearmanr(buy_confidence, valid_returns)
    results["ic"] = float(ic) if not np.isnan(ic) else 0.0
    results["ic_pvalue"] = float(ic_pvalue) if not np.isnan(ic_pvalue) else 1.0
    results["ic_significant"] = results["ic_pvalue"] < 0.05

    # 3. Directional Accuracy (non-HOLD predictions only)
    non_hold_mask = valid_preds != 1
    if non_hold_mask.sum() > 10:
        pred_direction = np.where(valid_preds[non_hold_mask] == 2, 1, -1)
        actual_direction = np.sign(valid_returns[non_hold_mask])
        non_zero = actual_direction != 0
        if non_zero.sum() > 0:
            dir_correct = (pred_direction[non_zero] == actual_direction[non_zero]).sum()
            results["directional_accuracy"] = float(dir_correct / non_zero.sum())
            results["n_directional"] = int(non_zero.sum())
        else:
            results["directional_accuracy"] = None
            results["n_directional"] = 0
    else:
        results["directional_accuracy"] = None
        results["n_directional"] = 0

    # 4. Simulated Trading Returns
    sim_returns = []
    for i in range(len(valid_preds)):
        if valid_preds[i] == 2:  # BUY
            sim_returns.append(valid_returns[i] - transaction_cost)
        elif valid_preds[i] == 0:  # SELL
            sim_returns.append(-valid_returns[i] - transaction_cost)
        else:  # HOLD
            sim_returns.append(0)

    sim_returns = np.array(sim_returns)
    non_zero_trades = sim_returns != 0

    if non_zero_trades.sum() > 10 and sim_returns[non_zero_trades].std() > 0:
        active_returns = sim_returns[non_zero_trades]
        results["sharpe"] = float((active_returns.mean() / active_returns.std()) * np.sqrt(252))
        results["ann_return"] = float(active_returns.mean() * 252)
        results["win_rate"] = float((active_returns > 0).mean())
        results["n_trades"] = int(non_zero_trades.sum())
        results["avg_return_per_trade"] = float(active_returns.mean())
    else:
        results["sharpe"] = None
        results["ann_return"] = None
        results["win_rate"] = None
        results["n_trades"] = int(non_zero_trades.sum())

    # 5. Confidence stats for this threshold
    results["mean_confidence"] = float(max_conf[mask].mean())
    results["median_confidence"] = float(np.median(max_conf[mask]))

    return results


def print_results_table(results_list: List[Dict[str, Any]]) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 120)
    print("METRICS BY CONFIDENCE THRESHOLD")
    print("=" * 120)

    # Header
    headers = [
        "Thresh",
        "N",
        "Cover",
        "Acc",
        "F1",
        "IC",
        "IC p",
        "Sig?",
        "Dir Acc",
        "Sharpe",
        "Ann Ret",
        "Win%",
        "Trades",
    ]
    print(f"{'|'.join(f'{h:>8}' for h in headers)}")
    print("-" * 120)

    for r in results_list:
        if r.get("insufficient_samples") or r.get("insufficient_valid_returns"):
            print(
                f"{r['threshold']:>7.0%} | {r['n_samples']:>6} | {r['coverage']:>5.1%} | "
                f"{'Insufficient samples':^70}"
            )
            continue

        row = [
            f"{r['threshold']:.0%}",
            f"{r['n_samples']}",
            f"{r['coverage']:.1%}",
            f"{r['accuracy']:.1%}",
            f"{r['f1_macro']:.1%}",
            f"{r['ic']:.3f}",
            f"{r['ic_pvalue']:.3f}",
            "✓" if r["ic_significant"] else "✗",
            f"{r['directional_accuracy']:.1%}" if r.get("directional_accuracy") else "-",
            f"{r['sharpe']:.2f}" if r.get("sharpe") is not None else "-",
            f"{r['ann_return']:.1%}" if r.get("ann_return") is not None else "-",
            f"{r['win_rate']:.1%}" if r.get("win_rate") is not None else "-",
            f"{r['n_trades']}" if r.get("n_trades") else "-",
        ]
        print(" | ".join(f"{v:>7}" for v in row))

    print("=" * 120)


def print_class_breakdown(results_list: List[Dict[str, Any]]) -> None:
    """Print per-class accuracy breakdown."""
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY BY THRESHOLD")
    print("=" * 80)

    headers = ["Thresh", "SELL Acc", "SELL N", "HOLD Acc", "HOLD N", "BUY Acc", "BUY N"]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 80)

    for r in results_list:
        if r.get("insufficient_samples"):
            continue

        row = [
            f"{r['threshold']:.0%}",
            f"{r.get('sell_accuracy', 0):.1%}" if r.get("sell_accuracy") else "-",
            f"{r.get('sell_count', 0)}",
            f"{r.get('hold_accuracy', 0):.1%}" if r.get("hold_accuracy") else "-",
            f"{r.get('hold_count', 0)}",
            f"{r.get('buy_accuracy', 0):.1%}" if r.get("buy_accuracy") else "-",
            f"{r.get('buy_count', 0)}",
        ]
        print(" | ".join(f"{v:>10}" for v in row))

    print("=" * 80)


def find_best_thresholds(results_list: List[Dict[str, Any]]) -> None:
    """Find and print optimal thresholds for different objectives."""
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLDS")
    print("=" * 80)

    valid_results = [r for r in results_list if not r.get("insufficient_samples")]

    # Best IC (significant)
    sig_results = [r for r in valid_results if r.get("ic_significant")]
    if sig_results:
        best_ic = max(sig_results, key=lambda x: x["ic"])
        print(f"\n📈 Best IC (significant only):")
        print(f"   Threshold: {best_ic['threshold']:.0%}")
        print(f"   IC: {best_ic['ic']:.4f} (p={best_ic['ic_pvalue']:.4f})")
        print(f"   Coverage: {best_ic['coverage']:.1%} ({best_ic['n_samples']} samples)")
    else:
        print("\n⚠️  No threshold has statistically significant IC (p < 0.05)")

    # Best Sharpe (with min 100 trades)
    sharpe_results = [r for r in valid_results if r.get("sharpe") is not None and r.get("n_trades", 0) >= 100]
    if sharpe_results:
        best_sharpe = max(sharpe_results, key=lambda x: x["sharpe"])
        print(f"\n📊 Best Sharpe Ratio (min 100 trades):")
        print(f"   Threshold: {best_sharpe['threshold']:.0%}")
        print(f"   Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"   Ann. Return: {best_sharpe['ann_return']:.1%}")
        print(f"   Win Rate: {best_sharpe['win_rate']:.1%}")
        print(f"   Trades: {best_sharpe['n_trades']}")

    # Best accuracy (with min 500 samples)
    acc_results = [r for r in valid_results if r["n_samples"] >= 500]
    if acc_results:
        best_acc = max(acc_results, key=lambda x: x["accuracy"])
        print(f"\n🎯 Best Accuracy (min 500 samples):")
        print(f"   Threshold: {best_acc['threshold']:.0%}")
        print(f"   Accuracy: {best_acc['accuracy']:.1%}")
        print(f"   F1 Macro: {best_acc['f1_macro']:.1%}")
        print(f"   Coverage: {best_acc['coverage']:.1%} ({best_acc['n_samples']} samples)")

    # Best directional accuracy
    dir_results = [r for r in valid_results if r.get("directional_accuracy") and r.get("n_directional", 0) >= 100]
    if dir_results:
        best_dir = max(dir_results, key=lambda x: x["directional_accuracy"])
        print(f"\n🧭 Best Directional Accuracy (min 100 predictions):")
        print(f"   Threshold: {best_dir['threshold']:.0%}")
        print(f"   Directional Acc: {best_dir['directional_accuracy']:.1%}")
        print(f"   N predictions: {best_dir['n_directional']}")

    print("\n" + "=" * 80)


def main() -> None:
    """Run confidence-threshold evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate metrics at different confidence thresholds")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--data", type=Path, default=Path("output/dataset.csv"), help="Path to dataset")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        help="Confidence thresholds to evaluate",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_dir}")

    model, scaler, encodings = load_model_for_evaluation(args.model_dir, device)

    # Load data
    print(f"Loading data from {args.data}")
    df = load_enriched_data(args.data)
    df = filter_reliable(df)

    # Temporal split (same as training: 80/10/10)
    df_train, df_val, df_test = split_by_time(df, test_size=0.1, val_size=0.1)
    print(f"Test set: {len(df_test)} samples")
    print(f"Test date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

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

    # Get predictions
    print("\nRunning inference...")
    eval_results = evaluate_on_test(model, test_dataset, device)

    predictions = eval_results["predictions"]
    probabilities = eval_results["probabilities"]
    labels = eval_results["labels"]
    returns = df_test["return_to_next_open"].values

    # Confidence distribution
    max_conf = probabilities.max(axis=1)
    print(f"\nConfidence Distribution:")
    print(f"  Min: {max_conf.min():.3f}")
    print(f"  25%: {np.percentile(max_conf, 25):.3f}")
    print(f"  50%: {np.percentile(max_conf, 50):.3f}")
    print(f"  75%: {np.percentile(max_conf, 75):.3f}")
    print(f"  Max: {max_conf.max():.3f}")

    # Evaluate at each threshold
    print("\nEvaluating at each threshold...")
    results_list = []
    for thresh in args.thresholds:
        metrics = compute_metrics_at_threshold(predictions, probabilities, labels, returns, thresh)
        results_list.append(metrics)

    # Print results
    print_results_table(results_list)
    print_class_breakdown(results_list)
    find_best_thresholds(results_list)

    # Save to JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_list, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

