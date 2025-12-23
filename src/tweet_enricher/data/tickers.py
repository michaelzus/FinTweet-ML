"""Ticker filtering utilities."""

import logging
from pathlib import Path
from typing import List

import pandas as pd

from tweet_enricher.config import DAILY_DATA_DIR

logger = logging.getLogger(__name__)


def filter_tickers_by_volume(data_dir: str = str(DAILY_DATA_DIR), min_avg_volume: float = 1_000_000) -> List[str]:
    """
    Filter tickers based on average daily trading volume.

    Reads feather files from the specified directory, calculates average daily volume
    for each ticker, and returns tickers with average volume above threshold.

    Args:
        data_dir: Directory containing feather files with historical data (default: data/daily)
        min_avg_volume: Minimum average daily volume threshold (default: 1,000,000)

    Returns:
        List of ticker symbols meeting the volume criteria

    Raises:
        FileNotFoundError: If data directory doesn't exist
    """
    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        filtered_tickers = []
        feather_files = list(data_path.glob("*.feather"))

        if not feather_files:
            logger.warning(f"No feather files found in {data_dir}")
            return []

        logger.info(f"Analyzing {len(feather_files)} feather files in {data_dir}...")

        for feather_file in feather_files:
            try:
                df = pd.read_feather(feather_file)

                # Check if volume column exists
                if "volume" not in df.columns:
                    logger.warning(f"No 'volume' column in {feather_file.name}, skipping")
                    continue

                # Calculate average volume
                avg_volume = df["volume"].mean()

                # Extract symbol from filename (e.g., AAPL.feather -> AAPL)
                symbol = feather_file.stem

                if avg_volume >= min_avg_volume:
                    filtered_tickers.append(symbol)
                    logger.debug(f"{symbol}: avg volume = {avg_volume:,.0f} (PASS)")
                else:
                    logger.debug(f"{symbol}: avg volume = {avg_volume:,.0f} (FAIL)")

            except Exception as e:
                logger.error(f"Error processing {feather_file.name}: {e}")
                continue

        logger.info(f"Found {len(filtered_tickers)} tickers with avg volume >= {min_avg_volume:,.0f}")
        return filtered_tickers

    except Exception as e:
        logger.error(f"Error filtering tickers by volume: {e}")
        raise
