"""Data caching layer for daily and intraday OHLCV data."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from tweet_enricher.config import DAILY_DATA_DIR, INTRADAY_CACHE_DIR, MARKET_CLOSE
from tweet_enricher.data.ib_fetcher import IBHistoricalDataFetcher
from tweet_enricher.io.feather import (
    load_daily_data,
    load_intraday_data,
    save_daily_data,
    save_intraday_data,
)
from tweet_enricher.market.session import normalize_dataframe_timezone, normalize_timestamp


class DataCache:
    """
    Manages disk and memory caching for daily and intraday OHLCV data.

    Provides a unified interface for fetching data with automatic caching,
    supporting both full fetches and incremental updates.
    """

    def __init__(
        self,
        ib_fetcher: IBHistoricalDataFetcher,
        daily_dir: Path = DAILY_DATA_DIR,
        intraday_dir: Path = INTRADAY_CACHE_DIR,
    ):
        """
        Initialize the DataCache.

        Args:
            ib_fetcher: IBHistoricalDataFetcher instance for fetching data
            daily_dir: Directory for daily data cache
            intraday_dir: Directory for intraday data cache
        """
        self.ib_fetcher = ib_fetcher
        self.daily_dir = daily_dir
        self.intraday_dir = intraday_dir
        self.logger = logging.getLogger(__name__)

        # In-memory caches
        self.daily_data_cache: Dict[str, pd.DataFrame] = {}
        self.intraday_data_cache: Dict[str, pd.DataFrame] = {}

    def get_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get daily data for a symbol from memory cache.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame if cached, None otherwise
        """
        return self.daily_data_cache.get(symbol)

    def get_intraday(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get intraday data for a symbol from memory cache.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame if cached, None otherwise
        """
        return self.intraday_data_cache.get(symbol)

    async def prefetch_all_daily_data(
        self,
        symbols: list,
        max_date: datetime,
        duration: str = "3 M",
        batch_size: int = 50,
    ) -> None:
        """
        Pre-fetch daily data for all symbols using disk cache with parallel batch fetching.

        This method implements a smart caching strategy with parallel fetching:
        1. Phase 1: Check all caches and categorize symbols
        2. Phase 2: Batch fetch symbols needing full data in parallel
        3. Phase 3: Batch fetch symbols needing incremental updates in parallel

        Args:
            symbols: List of ticker symbols to fetch
            max_date: Maximum date for data range
            duration: Full duration string (default: 3 M) - used for initial fetch
            batch_size: Number of symbols to fetch per batch (default: 50)
        """
        loaded_from_cache = 0
        fetched_full = 0
        updated_incremental = 0
        failed = 0

        # Normalize max_date to ET
        max_date_normalized = normalize_timestamp(max_date)
        target_date = max_date_normalized.date() if hasattr(max_date_normalized, "date") else max_date_normalized

        # ========== PHASE 1: Check all caches and categorize ==========
        self.logger.info("  Phase 1: Checking caches...")

        cached_fresh: list = []  # Symbols with up-to-date cache
        needs_full_fetch: list = []  # Symbols needing full fetch (no cache or too old)
        needs_incremental: list = []  # Symbols needing incremental update
        incremental_info: Dict[str, Tuple[pd.DataFrame, int]] = {}  # symbol -> (cached_df, days_missing)

        for symbol in symbols:
            cached_df = load_daily_data(symbol, self.daily_dir)

            if cached_df is None:
                needs_full_fetch.append(symbol)
            else:
                cached_max_date = cached_df.index.max()
                cached_date = cached_max_date.date() if hasattr(cached_max_date, "date") else cached_max_date

                if cached_date >= target_date:
                    # Cache is fresh
                    cached_fresh.append(symbol)
                    self.daily_data_cache[symbol] = cached_df
                    loaded_from_cache += 1
                else:
                    days_missing = (target_date - cached_date).days
                    if days_missing > 90:
                        # Cache too old, treat as full fetch
                        needs_full_fetch.append(symbol)
                    else:
                        # Needs incremental update
                        needs_incremental.append(symbol)
                        incremental_info[symbol] = (cached_df, days_missing)

        self.logger.info(
            f"  Phase 1 complete: {len(cached_fresh)} fresh, {len(needs_full_fetch)} full fetch, {len(needs_incremental)} incremental"
        )

        # ========== PHASE 2: Batch fetch symbols needing full data ==========
        if needs_full_fetch:
            self.logger.info(f"  Phase 2: Fetching {len(needs_full_fetch)} symbols (full data, {duration})...")

            # Callback to save immediately after each batch (preserves progress on interrupt)
            def save_daily_full_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal fetched_full
                for symbol, df in batch_data.items():
                    normalized_df = normalize_dataframe_timezone(df)
                    save_daily_data(symbol, normalized_df, self.daily_dir)
                    self.daily_data_cache[symbol] = normalized_df
                    fetched_full += 1

            await self.ib_fetcher.fetch_multiple_stocks(
                symbols=needs_full_fetch,
                bar_size="1 day",
                duration=duration,
                use_rth=True,
                batch_size=batch_size,
                on_batch_complete=save_daily_full_batch,
            )

            # Count failures (symbols that weren't saved by callback)
            failed = len(needs_full_fetch) - fetched_full

            self.logger.info(f"  Phase 2 complete: {fetched_full} fetched, {failed} failed")

        # ========== PHASE 3: Batch fetch incremental updates ==========
        if needs_incremental:
            self.logger.info(f"  Phase 3: Fetching {len(needs_incremental)} incremental updates...")

            # For incremental, we fetch with a common duration that covers all
            max_days_missing = max(info[1] for info in incremental_info.values())
            incremental_duration = f"{max_days_missing + 10} D"  # +10 days safety buffer

            # Callback to merge and save immediately after each batch
            def save_daily_incremental_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal updated_incremental, loaded_from_cache
                for symbol, df in batch_data.items():
                    if symbol not in incremental_info:
                        continue
                    cached_df, _ = incremental_info[symbol]
                    # Normalize cached_df to ensure timezone-aware index for comparison
                    cached_df = normalize_dataframe_timezone(cached_df)
                    cached_max_date = cached_df.index.max()

                    normalized_df = normalize_dataframe_timezone(df)

                    # Merge with cached data (keep only new rows)
                    new_rows = normalized_df[normalized_df.index > cached_max_date]

                    if not new_rows.empty:
                        merged_df = pd.concat([cached_df, new_rows])
                        merged_df = merged_df.sort_index()
                        merged_df = merged_df[~merged_df.index.duplicated(keep="last")]

                        save_daily_data(symbol, merged_df, self.daily_dir)
                        self.daily_data_cache[symbol] = merged_df
                        updated_incremental += 1
                    else:
                        # No new data, use cached
                        self.daily_data_cache[symbol] = cached_df
                        loaded_from_cache += 1

            await self.ib_fetcher.fetch_multiple_stocks(
                symbols=needs_incremental,
                bar_size="1 day",
                duration=incremental_duration,
                use_rth=True,
                batch_size=batch_size,
                on_batch_complete=save_daily_incremental_batch,
            )

            # Handle symbols that failed to fetch (use stale cache)
            for symbol in needs_incremental:
                if symbol not in self.daily_data_cache:
                    cached_df, _ = incremental_info[symbol]
                    self.daily_data_cache[symbol] = cached_df
                    loaded_from_cache += 1

            self.logger.info(f"  Phase 3 complete: {updated_incremental} updated")

        # ========== Summary ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Daily Data Summary:")
        self.logger.info(f"   - Loaded from cache: {loaded_from_cache}")
        self.logger.info(f"   - Fetched full: {fetched_full}")
        self.logger.info(f"   - Updated incremental: {updated_incremental}")
        self.logger.info(f"   - Failed: {failed}")
        self.logger.info(f"   - Total in memory cache: {len(self.daily_data_cache)}")
        self.logger.info("=" * 80)

    async def prefetch_all_intraday_data(
        self,
        symbols: list,
        max_date: datetime,
        duration: str = "60 D",
        batch_size: int = 3,
    ) -> None:
        """
        Pre-fetch intraday data for all symbols using disk cache with parallel batch fetching.

        This method implements a smart caching strategy with parallel fetching:
        1. Phase 1: Check all caches and categorize symbols
        2. Phase 2: Batch fetch symbols needing full data in parallel
        3. Phase 3: Batch fetch symbols needing incremental updates in parallel

        Args:
            symbols: List of ticker symbols to fetch
            max_date: Maximum date for data range
            duration: Full duration string (default: 60 D) - used for initial fetch
            batch_size: Number of symbols to fetch per batch (default: 3)
        """
        loaded_from_cache = 0
        fetched_full = 0
        updated_incremental = 0
        failed = 0

        # Normalize max_date to ET
        max_date_normalized = normalize_timestamp(max_date)
        target_date = max_date_normalized.date() if hasattr(max_date_normalized, "date") else max_date_normalized

        # ========== PHASE 1: Check all caches and categorize ==========
        self.logger.info("  Phase 1: Checking caches...")

        cached_fresh: list = []  # Symbols with up-to-date cache
        needs_full_fetch: list = []  # Symbols needing full fetch (no cache or too old)
        needs_incremental: list = []  # Symbols needing incremental update
        incremental_info: Dict[str, Tuple[pd.DataFrame, int]] = {}  # symbol -> (cached_df, days_missing)

        for symbol in symbols:
            cached_df = load_intraday_data(symbol, self.intraday_dir)

            if cached_df is None:
                needs_full_fetch.append(symbol)
            else:
                cached_max_date = cached_df.index.max()
                cached_date = cached_max_date.date() if hasattr(cached_max_date, "date") else cached_max_date

                # For intraday data, check if we have data up to end of trading on target date
                is_cache_fresh = False
                if cached_date > target_date:
                    is_cache_fresh = True
                elif cached_date == target_date:
                    # Check if we have data past market close (afterhours data)
                    cached_time_minutes = cached_max_date.hour * 60 + cached_max_date.minute
                    if cached_time_minutes >= MARKET_CLOSE:  # Has data at or after 4:00 PM
                        is_cache_fresh = True

                if is_cache_fresh:
                    # Cache is fresh
                    cached_fresh.append(symbol)
                    self.intraday_data_cache[symbol] = cached_df
                    loaded_from_cache += 1
                else:
                    days_missing = (target_date - cached_date).days
                    if days_missing > 60:
                        # Cache too old, treat as full fetch
                        needs_full_fetch.append(symbol)
                    else:
                        # Needs incremental update
                        needs_incremental.append(symbol)
                        incremental_info[symbol] = (cached_df, days_missing)

        self.logger.info(
            f"  Phase 1 complete: {len(cached_fresh)} fresh, {len(needs_full_fetch)} full fetch, {len(needs_incremental)} incremental"
        )

        # ========== PHASE 2: Batch fetch symbols needing full data ==========
        if needs_full_fetch:
            self.logger.info(f"  Phase 2: Fetching {len(needs_full_fetch)} symbols (full data, {duration})...")

            # Callback to save immediately after each batch (preserves progress on interrupt)
            def save_full_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal fetched_full
                for symbol, df in batch_data.items():
                    normalized_df = normalize_dataframe_timezone(df)
                    save_intraday_data(symbol, normalized_df, self.intraday_dir)
                    self.intraday_data_cache[symbol] = normalized_df
                    fetched_full += 1

            await self.ib_fetcher.fetch_multiple_stocks(
                symbols=needs_full_fetch,
                bar_size="15 mins",
                duration=duration,
                use_rth=False,  # Include extended hours
                batch_size=batch_size,
                delay_between_batches=5.0,  # Longer delay for large intraday data
                on_batch_complete=save_full_batch,
            )

            # Count failures (symbols that weren't saved by callback)
            failed = len(needs_full_fetch) - fetched_full

            self.logger.info(f"  Phase 2 complete: {fetched_full} fetched, {failed} failed")

        # ========== PHASE 3: Batch fetch incremental updates ==========
        if needs_incremental:
            self.logger.info(f"  Phase 3: Fetching {len(needs_incremental)} incremental updates...")

            # For incremental, we fetch with a common duration that covers all
            max_days_missing = max(info[1] for info in incremental_info.values())
            incremental_duration = f"{max_days_missing + 5} D"  # +5 days safety buffer

            # Callback to merge and save immediately after each batch (preserves progress on interrupt)
            def save_incremental_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal updated_incremental, loaded_from_cache
                for symbol, df in batch_data.items():
                    if symbol not in incremental_info:
                        continue
                    cached_df, _ = incremental_info[symbol]
                    # Normalize cached_df to ensure timezone-aware index for comparison
                    cached_df = normalize_dataframe_timezone(cached_df)
                    cached_max_date = cached_df.index.max()

                    normalized_df = normalize_dataframe_timezone(df)

                    # Merge with cached data (keep only new rows)
                    new_rows = normalized_df[normalized_df.index > cached_max_date]

                    if not new_rows.empty:
                        merged_df = pd.concat([cached_df, new_rows])
                        merged_df = merged_df.sort_index()
                        merged_df = merged_df[~merged_df.index.duplicated(keep="last")]

                        save_intraday_data(symbol, merged_df, self.intraday_dir)
                        self.intraday_data_cache[symbol] = merged_df
                        updated_incremental += 1
                    else:
                        # No new data, use cached
                        self.intraday_data_cache[symbol] = cached_df
                        loaded_from_cache += 1

            await self.ib_fetcher.fetch_multiple_stocks(
                symbols=needs_incremental,
                bar_size="15 mins",
                duration=incremental_duration,
                use_rth=False,  # Include extended hours
                batch_size=batch_size,
                delay_between_batches=5.0,  # Longer delay for large intraday data
                on_batch_complete=save_incremental_batch,
            )

            # Handle symbols that failed to fetch (use stale cache)
            for symbol in needs_incremental:
                if symbol not in self.intraday_data_cache:
                    cached_df, _ = incremental_info[symbol]
                    self.intraday_data_cache[symbol] = cached_df
                    loaded_from_cache += 1

            self.logger.info(f"  Phase 3 complete: {updated_incremental} updated")

        # ========== Summary ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Intraday Data Summary:")
        self.logger.info(f"   - Loaded from cache: {loaded_from_cache}")
        self.logger.info(f"   - Fetched full: {fetched_full}")
        self.logger.info(f"   - Updated incremental: {updated_incremental}")
        self.logger.info(f"   - Failed: {failed}")
        self.logger.info(f"   - Total in memory cache: {len(self.intraday_data_cache)}")
        self.logger.info("=" * 80)
