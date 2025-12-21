"""Feather file I/O operations for daily and intraday OHLCV data.

Note on timezone handling:
    Data from IB (Interactive Brokers) is assumed to be in US Eastern Time.
    This module normalizes all timestamps to timezone-aware ET before saving
    to ensure consistent timezone handling across the pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tweet_enricher.config import DAILY_DATA_DIR
from tweet_enricher.utils.timezone import normalize_dataframe_timezone

logger = logging.getLogger(__name__)


def save_daily_data(symbol: str, df: pd.DataFrame, data_dir: Path = DAILY_DATA_DIR) -> None:
    """
    Save daily OHLCV data to feather format.

    Args:
        symbol: Stock ticker symbol
        df: DataFrame with OHLCV data (index should be 'date')
        data_dir: Directory to save feather files (default: data/daily)

    Note:
        Timestamps are normalized to timezone-aware US Eastern Time before saving.
        IB data is assumed to already be in ET (naive or aware).
    """
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_path = data_dir / f"{symbol}.feather"

        # Normalize timezone to ET before saving (DST-safe)
        df = normalize_dataframe_timezone(df.copy())

        # Reset index to save it as a column (Feather doesn't preserve index)
        df_to_save = df.reset_index()

        # Ensure first column is named 'date'
        if df_to_save.columns[0] != "date":
            df_to_save.columns = ["date"] + list(df_to_save.columns[1:])

        df_to_save.to_feather(cache_path)
        logger.debug(f"Saved {len(df)} daily bars to {cache_path}")

    except Exception as e:
        logger.error(f"Error saving daily data for {symbol}: {e}")


def load_daily_data(symbol: str, data_dir: Path = DAILY_DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Load daily OHLCV data from feather format.

    Args:
        symbol: Stock ticker symbol
        data_dir: Directory containing feather files (default: data/daily)

    Returns:
        DataFrame with OHLCV data (index is 'date'), or None if not found
    """
    cache_path = data_dir / f"{symbol}.feather"

    if not cache_path.exists():
        return None

    try:
        df = pd.read_feather(cache_path)

        # Set 'date' as index
        if "date" in df.columns:
            df = df.set_index("date")

        logger.debug(f"Loaded {len(df)} daily bars from {cache_path}")
        return df

    except Exception as e:
        logger.warning(f"Failed to load daily data for {symbol}: {e}")
        return None


def save_intraday_data(symbol: str, df: pd.DataFrame, data_dir: Path) -> None:
    """
    Save intraday data to disk cache (Feather format).

    Args:
        symbol: Stock ticker symbol
        df: DataFrame to save
        data_dir: Directory to save feather files

    Note:
        Timestamps are normalized to timezone-aware US Eastern Time before saving.
        IB data is assumed to already be in ET (naive or aware).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{symbol}.feather"

    try:
        # Normalize timezone to ET before saving (DST-safe)
        df = normalize_dataframe_timezone(df.copy())

        # Reset index to save it as a column (Feather doesn't preserve index)
        df_to_save = df.reset_index()
        df_to_save.columns = ["date"] + list(df_to_save.columns[1:])
        df_to_save.to_feather(cache_path)
        logger.debug(f"Saved {len(df)} intraday bars to cache for {symbol}")
    except Exception as e:
        logger.error(f"Failed to save intraday cache for {symbol}: {e}")


def load_intraday_data(symbol: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load intraday data from disk cache (Feather format).

    Args:
        symbol: Stock ticker symbol
        data_dir: Directory containing feather files

    Returns:
        DataFrame if cache exists and valid, None otherwise

    Note:
        Feather format preserves timezone-aware datetime columns natively.
    """
    cache_path = data_dir / f"{symbol}.feather"

    if not cache_path.exists():
        return None

    try:
        df = pd.read_feather(cache_path)

        # Feather doesn't preserve the index, so set it
        df = df.set_index("date")

        logger.debug(f"Loaded {len(df)} intraday bars from cache for {symbol}")
        return df

    except Exception as e:
        logger.warning(f"Failed to load intraday cache for {symbol}: {e}")
        return None
