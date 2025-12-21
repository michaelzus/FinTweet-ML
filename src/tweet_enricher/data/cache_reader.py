"""Read-only cache reader for OHLCV data.

Provides access to cached feather files without any API dependencies.
This enables the dataset preparation flow to work completely offline.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tweet_enricher.config import DAILY_DATA_DIR, INTRADAY_CACHE_DIR
from tweet_enricher.io.feather import load_daily_data, load_intraday_data
from tweet_enricher.market.session import normalize_dataframe_timezone

logger = logging.getLogger(__name__)


@dataclass
class CoverageInfo:
    """Data coverage information for a single symbol."""

    symbol: str
    has_daily: bool
    has_intraday: bool
    daily_start: Optional[date] = None
    daily_end: Optional[date] = None
    intraday_start: Optional[date] = None
    intraday_end: Optional[date] = None
    daily_bars: int = 0
    intraday_bars: int = 0


@dataclass
class ValidationReport:
    """Validation report for dataset preparation."""

    total_tickers: int
    available_tickers: int
    missing_tickers: List[str]
    partial_coverage: List[str]  # Have some data but may have gaps
    coverage_details: Dict[str, CoverageInfo]

    @property
    def is_valid(self) -> bool:
        """Check if all required data is available."""
        return len(self.missing_tickers) == 0


class CacheReader:
    """
    Read-only access to cached OHLCV data.

    This class provides a clean interface for reading cached data without
    any dependency on the IB API. It's used by the DatasetBuilder for
    offline dataset preparation.
    """

    def __init__(
        self,
        daily_dir: Path = DAILY_DATA_DIR,
        intraday_dir: Path = INTRADAY_CACHE_DIR,
    ):
        """
        Initialize the cache reader.

        Args:
            daily_dir: Directory containing daily OHLCV feather files
            intraday_dir: Directory containing intraday feather files
        """
        self.daily_dir = Path(daily_dir)
        self.intraday_dir = Path(intraday_dir)

        # In-memory caches for performance
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._intraday_cache: Dict[str, pd.DataFrame] = {}

    def get_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load daily data from feather cache.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data (index is datetime), or None if not found
        """
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]

        df = load_daily_data(symbol, self.daily_dir)
        if df is not None:
            df = normalize_dataframe_timezone(df)
            self._daily_cache[symbol] = df

        return df

    def get_intraday(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load intraday data from feather cache.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data (index is datetime), or None if not found
        """
        if symbol in self._intraday_cache:
            return self._intraday_cache[symbol]

        df = load_intraday_data(symbol, self.intraday_dir)
        if df is not None:
            df = normalize_dataframe_timezone(df)
            self._intraday_cache[symbol] = df

        return df

    def preload_symbols(self, symbols: List[str]) -> None:
        """
        Preload data for multiple symbols into memory cache.

        Args:
            symbols: List of ticker symbols to preload
        """
        logger.info(f"Preloading data for {len(symbols)} symbols...")

        for symbol in symbols:
            self.get_daily(symbol)
            self.get_intraday(symbol)

        loaded_daily = sum(1 for s in symbols if s in self._daily_cache)
        loaded_intraday = sum(1 for s in symbols if s in self._intraday_cache)

        logger.info(f"Preloaded: {loaded_daily} daily, {loaded_intraday} intraday")

    def get_coverage(self, symbol: str) -> CoverageInfo:
        """
        Get data coverage information for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CoverageInfo with availability and date range details
        """
        daily_df = self.get_daily(symbol)
        intraday_df = self.get_intraday(symbol)

        info = CoverageInfo(
            symbol=symbol,
            has_daily=daily_df is not None and not daily_df.empty,
            has_intraday=intraday_df is not None and not intraday_df.empty,
        )

        if info.has_daily and daily_df is not None:
            info.daily_start = daily_df.index.min().date()
            info.daily_end = daily_df.index.max().date()
            info.daily_bars = len(daily_df)

        if info.has_intraday and intraday_df is not None:
            info.intraday_start = intraday_df.index.min().date()
            info.intraday_end = intraday_df.index.max().date()
            info.intraday_bars = len(intraday_df)

        return info

    def validate_coverage(
        self,
        symbols: List[str],
        require_intraday: bool = True,
    ) -> ValidationReport:
        """
        Validate data coverage for multiple symbols.

        Args:
            symbols: List of ticker symbols to validate
            require_intraday: Whether intraday data is required

        Returns:
            ValidationReport with coverage details and missing tickers
        """
        missing = []
        partial = []
        coverage_details = {}

        for symbol in symbols:
            info = self.get_coverage(symbol)
            coverage_details[symbol] = info

            if not info.has_daily:
                missing.append(symbol)
            elif require_intraday and not info.has_intraday:
                partial.append(symbol)

        return ValidationReport(
            total_tickers=len(symbols),
            available_tickers=len(symbols) - len(missing),
            missing_tickers=missing,
            partial_coverage=partial,
            coverage_details=coverage_details,
        )

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about available cache files.

        Returns:
            Dictionary with cache statistics
        """
        daily_files = list(self.daily_dir.glob("*.feather")) if self.daily_dir.exists() else []
        intraday_files = list(self.intraday_dir.glob("*.feather")) if self.intraday_dir.exists() else []

        return {
            "daily_symbols": len(daily_files),
            "intraday_symbols": len(intraday_files),
            "daily_dir": str(self.daily_dir),
            "intraday_dir": str(self.intraday_dir),
        }

    def list_available_symbols(self, data_type: str = "daily") -> List[str]:
        """
        List all symbols available in cache.

        Args:
            data_type: 'daily' or 'intraday'

        Returns:
            List of symbol names (without .feather extension)
        """
        if data_type == "daily":
            cache_dir = self.daily_dir
        elif data_type == "intraday":
            cache_dir = self.intraday_dir
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Use 'daily' or 'intraday'")

        if not cache_dir.exists():
            return []

        return sorted([f.stem for f in cache_dir.glob("*.feather")])
