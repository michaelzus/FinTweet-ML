#!/usr/bin/env python3
"""Pre-process tweets with FinBERT sentiment analysis.

This script adds sentiment scores to tweets CSV for faster scanner execution.
Run this once to enrich your tweets, then use the enriched CSV with the scanner.

Usage:
    python scripts/preprocess_tweet_sentiment.py \
        --input output/tweets_2025_export.csv \
        --output output/tweets_with_sentiment.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_finbert():
    """Load FinBERT model from HuggingFace."""
    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        logger.info(f"Loading FinBERT model on {device_name}...")

        model = pipeline(
            "sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", truncation=True, max_length=512, device=device
        )
        logger.info("FinBERT loaded successfully")
        return model

    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install transformers torch")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading FinBERT: {e}")
        sys.exit(1)


def process_tweets(input_path: Path, output_path: Path, batch_size: int = 32, checkpoint_every: int = 5000) -> None:
    """
    Process tweets with FinBERT sentiment analysis.

    Args:
        input_path: Path to input tweets CSV
        output_path: Path to output enriched CSV
        batch_size: Number of tweets to process at once
        checkpoint_every: Save checkpoint every N tweets
    """
    # Load tweets
    logger.info(f"Loading tweets from {input_path}...")
    df = pd.read_csv(input_path)
    total_tweets = len(df)
    logger.info(f"Loaded {total_tweets} tweets")

    # Check for existing sentiment columns (resume support)
    if "sentiment_score" in df.columns:
        processed_mask = df["sentiment_score"].notna()
        already_processed = processed_mask.sum()
        if already_processed > 0:
            logger.info(f"Found {already_processed} already processed tweets, resuming...")
    else:
        df["sentiment_label"] = None
        df["sentiment_score"] = None
        df["sentiment_confidence"] = None
        processed_mask = pd.Series([False] * len(df))

    # Get rows that need processing
    to_process = df[~processed_mask].index.tolist()
    logger.info(f"Need to process {len(to_process)} tweets")

    if len(to_process) == 0:
        logger.info("All tweets already processed!")
        return

    # Load FinBERT
    finbert = load_finbert()

    # Label mapping
    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    # Process in batches
    processed_count = 0
    checkpoint_path = output_path.with_suffix(".checkpoint.csv")

    try:
        for batch_start in range(0, len(to_process), batch_size):
            batch_indices = to_process[batch_start : batch_start + batch_size]
            batch_texts = df.loc[batch_indices, "text"].tolist()

            # Handle empty/null texts
            batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]

            try:
                results = finbert(batch_texts)

                for idx, result in zip(batch_indices, results):
                    label = result["label"]
                    confidence = result["score"]
                    score = label_map[label] * confidence

                    df.at[idx, "sentiment_label"] = label
                    df.at[idx, "sentiment_score"] = round(score, 4)
                    df.at[idx, "sentiment_confidence"] = round(confidence, 4)

            except Exception as e:
                logger.warning(f"Error processing batch at {batch_start}: {e}")
                # Mark failed rows with neutral sentiment
                for idx in batch_indices:
                    df.at[idx, "sentiment_label"] = "neutral"
                    df.at[idx, "sentiment_score"] = 0.0
                    df.at[idx, "sentiment_confidence"] = 0.0

            processed_count += len(batch_indices)

            # Progress logging
            if processed_count % 1000 == 0 or processed_count == len(to_process):
                pct = (processed_count / len(to_process)) * 100
                logger.info(f"Processed {processed_count}/{len(to_process)} tweets ({pct:.1f}%)")

            # Checkpoint save
            if processed_count % checkpoint_every == 0:
                logger.info(f"Saving checkpoint at {processed_count} tweets...")
                df.to_csv(checkpoint_path, index=False)

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving progress...")
        df.to_csv(checkpoint_path, index=False)
        logger.info(f"Progress saved to {checkpoint_path}")
        logger.info("Run again to resume from checkpoint")
        sys.exit(0)

    # Save final output
    logger.info(f"Saving enriched tweets to {output_path}...")
    df.to_csv(output_path, index=False)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Cleaned up checkpoint file")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SENTIMENT PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    sentiment_counts = df["sentiment_label"].value_counts()
    logger.info(f"Total tweets processed: {total_tweets}")
    for label, count in sentiment_counts.items():
        pct = (count / total_tweets) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    avg_sentiment = df["sentiment_score"].mean()
    logger.info(f"Average sentiment score: {avg_sentiment:+.3f}")
    logger.info(f"\nOutput saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pre-process tweets with FinBERT sentiment analysis")
    parser.add_argument("--input", "-i", type=Path, default=Path("output/tweets_2025_export.csv"), help="Input tweets CSV path")
    parser.add_argument("--output", "-o", type=Path, default=Path("output/tweets_with_sentiment.csv"), help="Output enriched CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for FinBERT processing (default: 32)")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Save checkpoint every N tweets (default: 5000)")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Process tweets
    process_tweets(input_path=args.input, output_path=args.output, batch_size=args.batch_size, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
