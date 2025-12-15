"""
Tweet enrichment script.

This script enriches tweet data with financial indicators and price information
from Interactive Brokers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pandas_market_calendars as mcal
import pytz

from ib_fetcher import IBHistoricalDataFetcher
from technical_indicators import TechnicalIndicators
from helpers import save_daily_data, load_daily_data, DAILY_DATA_DIR, EXCLUDED_TICKERS

# US Eastern timezone
ET = pytz.timezone("US/Eastern")

# Market hours definition (all in ET)
MARKET_OPEN = 9 * 60 + 30  # 9:30 AM in minutes
MARKET_CLOSE = 16 * 60  # 4:00 PM in minutes
PREMARKET_START = 4 * 60  # 4:00 AM in minutes
AFTERHOURS_END = 20 * 60  # 8:00 PM in minutes

# Cache directory for intraday data (daily uses shared DAILY_DATA_DIR from helpers)
INTRADAY_CACHE_DIR = Path("data/intraday")


class MarketSession(Enum):
    """Market session types."""

    REGULAR = "regular"
    PREMARKET = "premarket"
    AFTERHOURS = "afterhours"
    CLOSED = "closed"


class TweetEnricher:
    """
    Enriches tweet data with financial indicators and price information.

    Handles data fetching from IBKR, technical indicator calculation,
    and market session awareness.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize the TweetEnricher.

        Args:
            host: TWS/Gateway host address
            port: TWS/Gateway port number
            client_id: Unique client identifier
        """
        self.ib_fetcher = IBHistoricalDataFetcher(host, port, client_id)
        self.logger = self._setup_logger()
        self.tech_indicators = TechnicalIndicators()

        # Cache for fetched data
        self.daily_data_cache: Dict[str, pd.DataFrame] = {}
        self.intraday_data_cache: Dict[str, pd.DataFrame] = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # # Avoid duplicate handlers
        # if not logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        #     handler.setFormatter(formatter)
        #     logger.addHandler(handler)

        return logger

    async def connect(self) -> bool:
        """Establish connection to Interactive Brokers."""
        return await self.ib_fetcher.connect()

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        await self.ib_fetcher.disconnect()

    async def _fetch_batch(
        self,
        symbols: list,
        bar_size: str,
        duration: str,
        use_rth: bool = True,
        batch_size: int = 50,
        delay_between_batches: float = 2.0,
        on_batch_complete: Optional[callable] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel batches.

        Args:
            symbols: List of ticker symbols to fetch
            bar_size: Bar size (e.g., "1 day", "15 mins")
            duration: Duration string (e.g., "3 M", "60 D")
            use_rth: Use regular trading hours only
            batch_size: Number of symbols per batch (default: 50)
            delay_between_batches: Delay between batches in seconds (default: 2.0)
            on_batch_complete: Optional callback(batch_results: Dict[str, pd.DataFrame]) called after each batch

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(symbols)

        if total == 0:
            return results

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_symbols = symbols[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            self.logger.info(f"  📦 Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols in parallel...")

            # Create tasks for parallel execution
            tasks = [
                self.ib_fetcher.fetch_historical_data(
                    symbol=symbol,
                    exchange="SMART",
                    currency="USD",
                    duration=duration,
                    bar_size=bar_size,
                    use_rth=use_rth,
                )
                for symbol in batch_symbols
            ]

            # Execute all tasks in parallel
            batch_results = await asyncio.gather(*tasks)

            # Collect successful results and count failures
            successful = 0
            failed = 0
            batch_data: Dict[str, pd.DataFrame] = {}
            for symbol, df in zip(batch_symbols, batch_results):
                if df is not None and not df.empty:
                    results[symbol] = df
                    batch_data[symbol] = df
                    successful += 1
                else:
                    failed += 1

            if failed > 0:
                self.logger.warning(f"  📦 Batch {batch_num}/{total_batches}: {successful} ✅, {failed} ❌ failed")
            else:
                self.logger.info(f"  📦 Batch {batch_num}/{total_batches}: {successful} ✅ all successful")

            # Call callback with batch results (for immediate saving)
            if on_batch_complete and batch_data:
                on_batch_complete(batch_data)

            # Delay between batches (except after the last batch)
            if batch_end < total:
                await asyncio.sleep(delay_between_batches)

        return results

    def _get_intraday_cache_path(self, symbol: str) -> Path:
        """
        Get intraday cache file path for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Path to cache file
        """
        return INTRADAY_CACHE_DIR / f"{symbol}.feather"

    def _load_intraday_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load intraday data from disk cache (Feather format).

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame if cache exists and valid, None otherwise
        """
        cache_path = self._get_intraday_cache_path(symbol)

        if not cache_path.exists():
            return None

        try:
            # Feather format preserves timezone-aware datetime index natively!
            df = pd.read_feather(cache_path)

            # Feather doesn't preserve the index, so set it
            df = df.set_index('date')

            self.logger.debug(f"Loaded {len(df)} intraday bars from cache for {symbol}")
            return df

        except Exception as e:
            self.logger.warning(f"Failed to load intraday cache for {symbol}: {e}")
            return None

    def _save_intraday_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Save intraday data to disk cache (Feather format).

        Args:
            symbol: Stock ticker symbol
            df: DataFrame to save
        """
        # Ensure cache directory exists
        INTRADAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_intraday_cache_path(symbol)

        try:
            # Reset index to save it as a column (Feather doesn't preserve index)
            df_to_save = df.reset_index()
            df_to_save.columns = ['date'] + list(df_to_save.columns[1:])
            df_to_save.to_feather(cache_path)
            self.logger.debug(f"Saved {len(df)} intraday bars to cache for {symbol}")
        except Exception as e:
            self.logger.error(f"Failed to save intraday cache for {symbol}: {e}")

    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """
        Ensure timestamp is timezone-aware and in ET.

        Args:
            timestamp: Datetime object (may be naive or aware)

        Returns:
            Timezone-aware datetime in ET
        """
        if timestamp.tzinfo is None:
            return ET.localize(timestamp)
        else:
            return timestamp.astimezone(ET)

    def _normalize_dataframe_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame index is timezone-aware and in ET.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with timezone-aware index in ET
        """
        if df.index.tz is None:
            df.index = df.index.tz_localize(ET)
        else:
            df.index = df.index.tz_convert(ET)
        return df

    def get_market_session(self, timestamp: datetime) -> MarketSession:
        """
        Determine market session for a given timestamp.

        Args:
            timestamp: Timestamp to check

        Returns:
            MarketSession enum value
        """
        # Ensure timestamp is in ET
        timestamp = self._normalize_timestamp(timestamp)

        # Check if it's a weekend
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.CLOSED

        # Convert to minutes since midnight
        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute

        if MARKET_OPEN <= minutes_since_midnight < MARKET_CLOSE:
            return MarketSession.REGULAR
        elif PREMARKET_START <= minutes_since_midnight < MARKET_OPEN:
            return MarketSession.PREMARKET
        elif MARKET_CLOSE <= minutes_since_midnight < AFTERHOURS_END:
            return MarketSession.AFTERHOURS
        else:
            return MarketSession.CLOSED

    async def fetch_daily_data(self, symbol: str, max_date: datetime, duration: str = "3 M") -> Optional[pd.DataFrame]:
        """
        Fetch daily historical data for entire date range.

        **CRITICAL**: Returns FULL DataFrame - caller must slice by tweet date to avoid look-ahead bias!

        Args:
            symbol: Stock ticker symbol
            max_date: Maximum date in dataset (used for end_datetime)
            duration: Duration string (default: 3 M to cover full range)

        Returns:
            DataFrame with ALL daily OHLCV data (timezone-aware)
            CALLER MUST SLICE: df[df.index <= tweet_date]
        """
        if symbol in self.daily_data_cache:
            return self.daily_data_cache[symbol]

        try:
            # Fetch ENTIRE date range for this ticker
            end_datetime_str = max_date.strftime("%Y%m%d 23:59:59")

            df = await self.ib_fetcher.fetch_historical_data(
                symbol=symbol,
                exchange="SMART",
                currency="USD",
                duration=duration,
                bar_size="1 day",
                use_rth=True,
                end_datetime=end_datetime_str,
            )

            if df is None or df.empty:
                self.logger.warning(f"No daily data for {symbol}")
                return None

            # Ensure timezone-aware datetime
            df = self._normalize_dataframe_timezone(df)

            # Cache the FULL DataFrame
            self.daily_data_cache[symbol] = df
            self.logger.debug(f"Fetched {len(df)} daily bars for {symbol} (full range)")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None

    async def fetch_intraday_data(self, symbol: str, max_date: datetime, duration: str = "60 D") -> Optional[pd.DataFrame]:
        """
        Fetch intraday historical data (15-min bars) for entire date range with extended hours.

        **CRITICAL**: Returns FULL DataFrame - caller must slice by tweet timestamp to avoid look-ahead bias!

        Args:
            symbol: Stock ticker symbol
            max_date: Maximum date in dataset (used for end_datetime)
            duration: Duration string (default: 60 D - safe for 15-min bars)

        Returns:
            DataFrame with ALL 15-minute OHLCV data (timezone-aware)
            CALLER MUST SLICE: df[df.index <= tweet_timestamp]
        """

        if symbol in self.intraday_data_cache:
            return self.intraday_data_cache[symbol]

        try:
            # Fetch ENTIRE intraday range for this ticker
            # Add 1 day to max_date to ensure we have data for "1hr after" last tweet
            end_datetime_str = (max_date + timedelta(days=1)).strftime("%Y%m%d 23:59:59")

            df = await self.ib_fetcher.fetch_historical_data(
                symbol=symbol,
                exchange="SMART",
                currency="USD",
                duration=duration,
                bar_size="15 mins",  # Changed from "1 min" to "15 mins"
                use_rth=False,  # Include extended hours (pre-market, after-hours)
                end_datetime=end_datetime_str,
            )

            if df is None or df.empty:
                self.logger.warning(f"No intraday data for {symbol}")
                return None

            # Ensure timezone-aware datetime
            df = self._normalize_dataframe_timezone(df)

            # Cache the FULL DataFrame
            self.intraday_data_cache[symbol] = df
            self.logger.debug(f"Fetched {len(df)} intraday bars (15-min) for {symbol} (full range)")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    async def prefetch_all_daily_data(
        self, symbols: list, max_date: datetime, duration: str = "3 M", batch_size: int = 50
    ) -> None:
        """
        Pre-fetch daily data for all symbols using disk cache with parallel batch fetching.

        This method implements a smart caching strategy with parallel fetching:
        1. Phase 1: Check all caches and categorize symbols
        2. Phase 2: Batch fetch symbols needing full data in parallel
        3. Phase 3: Batch fetch symbols needing incremental updates in parallel
        4. Phase 4: Process results (normalize, merge, save)

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
        max_date_normalized = self._normalize_timestamp(max_date)
        target_date = max_date_normalized.date() if hasattr(max_date_normalized, 'date') else max_date_normalized

        # ========== PHASE 1: Check all caches and categorize ==========
        self.logger.info("  📋 Phase 1: Checking caches...")

        cached_fresh: list = []  # Symbols with up-to-date cache
        needs_full_fetch: list = []  # Symbols needing full fetch (no cache or too old)
        needs_incremental: list = []  # Symbols needing incremental update
        incremental_info: Dict[str, Tuple[pd.DataFrame, int]] = {}  # symbol -> (cached_df, days_missing)

        for symbol in symbols:
            cached_df = load_daily_data(symbol, DAILY_DATA_DIR)

            if cached_df is None:
                needs_full_fetch.append(symbol)
            else:
                cached_max_date = cached_df.index.max()
                cached_date = cached_max_date.date() if hasattr(cached_max_date, 'date') else cached_max_date

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
            f"  📋 Phase 1 complete: {len(cached_fresh)} fresh, {len(needs_full_fetch)} full fetch, {len(needs_incremental)} incremental"
        )

        # ========== PHASE 2: Batch fetch symbols needing full data ==========
        if needs_full_fetch:
            self.logger.info(f"  🚀 Phase 2: Fetching {len(needs_full_fetch)} symbols (full data, {duration})...")

            # Callback to save immediately after each batch (preserves progress on interrupt)
            def save_daily_full_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal fetched_full
                for symbol, df in batch_data.items():
                    normalized_df = self._normalize_dataframe_timezone(df)
                    save_daily_data(symbol, normalized_df, DAILY_DATA_DIR)
                    self.daily_data_cache[symbol] = normalized_df
                    fetched_full += 1

            await self._fetch_batch(
                symbols=needs_full_fetch,
                bar_size="1 day",
                duration=duration,
                use_rth=True,
                batch_size=batch_size,
                on_batch_complete=save_daily_full_batch,
            )

            # Count failures (symbols that weren't saved by callback)
            failed = len(needs_full_fetch) - fetched_full

            self.logger.info(f"  🚀 Phase 2 complete: {fetched_full} fetched, {failed} failed")

        # ========== PHASE 3: Batch fetch incremental updates ==========
        if needs_incremental:
            self.logger.info(f"  🔄 Phase 3: Fetching {len(needs_incremental)} incremental updates...")

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
                    cached_max_date = cached_df.index.max()

                    normalized_df = self._normalize_dataframe_timezone(df)

                    # Merge with cached data (keep only new rows)
                    new_rows = normalized_df[normalized_df.index > cached_max_date]

                    if not new_rows.empty:
                        merged_df = pd.concat([cached_df, new_rows])
                        merged_df = merged_df.sort_index()
                        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]

                        save_daily_data(symbol, merged_df, DAILY_DATA_DIR)
                        self.daily_data_cache[symbol] = merged_df
                        updated_incremental += 1
                    else:
                        # No new data, use cached
                        self.daily_data_cache[symbol] = cached_df
                        loaded_from_cache += 1

            await self._fetch_batch(
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

            self.logger.info(f"  🔄 Phase 3 complete: {updated_incremental} updated")

        # ========== Summary ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 Daily Data Summary:")
        self.logger.info(f"   • Loaded from cache: {loaded_from_cache}")
        self.logger.info(f"   • Fetched full: {fetched_full}")
        self.logger.info(f"   • Updated incremental: {updated_incremental}")
        self.logger.info(f"   • Failed: {failed}")
        self.logger.info(f"   • Total in memory cache: {len(self.daily_data_cache)}")
        self.logger.info("=" * 80)

    async def prefetch_all_intraday_data(
        self, symbols: list, max_date: datetime, duration: str = "60 D", batch_size: int = 3
    ) -> None:
        """
        Pre-fetch intraday data for all symbols using disk cache with parallel batch fetching.

        This method implements a smart caching strategy with parallel fetching:
        1. Phase 1: Check all caches and categorize symbols
        2. Phase 2: Batch fetch symbols needing full data in parallel
        3. Phase 3: Batch fetch symbols needing incremental updates in parallel
        4. Phase 4: Process results (normalize, merge, save)

        Args:
            symbols: List of ticker symbols to fetch
            max_date: Maximum date for data range
            duration: Full duration string (default: 60 D) - used for initial fetch
            batch_size: Number of symbols to fetch per batch (default: 50)
        """
        loaded_from_cache = 0
        fetched_full = 0
        updated_incremental = 0
        failed = 0

        # Normalize max_date to ET
        max_date_normalized = self._normalize_timestamp(max_date)
        target_date = max_date_normalized.date() if hasattr(max_date_normalized, 'date') else max_date_normalized

        # ========== PHASE 1: Check all caches and categorize ==========
        self.logger.info("  📋 Phase 1: Checking caches...")

        cached_fresh: list = []  # Symbols with up-to-date cache
        needs_full_fetch: list = []  # Symbols needing full fetch (no cache or too old)
        needs_incremental: list = []  # Symbols needing incremental update
        incremental_info: Dict[str, Tuple[pd.DataFrame, int]] = {}  # symbol -> (cached_df, days_missing)

        for symbol in symbols:
            cached_df = self._load_intraday_from_cache(symbol)

            if cached_df is None:
                needs_full_fetch.append(symbol)
            else:
                cached_max_date = cached_df.index.max()
                cached_date = cached_max_date.date() if hasattr(cached_max_date, 'date') else cached_max_date

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
            f"  📋 Phase 1 complete: {len(cached_fresh)} fresh, {len(needs_full_fetch)} full fetch, {len(needs_incremental)} incremental"
        )

        # ========== PHASE 2: Batch fetch symbols needing full data ==========
        if needs_full_fetch:
            self.logger.info(f"  🚀 Phase 2: Fetching {len(needs_full_fetch)} symbols (full data, {duration})...")

            # Callback to save immediately after each batch (preserves progress on interrupt)
            def save_full_batch(batch_data: Dict[str, pd.DataFrame]) -> None:
                nonlocal fetched_full
                for symbol, df in batch_data.items():
                    normalized_df = self._normalize_dataframe_timezone(df)
                    self._save_intraday_to_cache(symbol, normalized_df)
                    self.intraday_data_cache[symbol] = normalized_df
                    fetched_full += 1

            await self._fetch_batch(
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

            self.logger.info(f"  🚀 Phase 2 complete: {fetched_full} fetched, {failed} failed")

        # ========== PHASE 3: Batch fetch incremental updates ==========
        if needs_incremental:
            self.logger.info(f"  🔄 Phase 3: Fetching {len(needs_incremental)} incremental updates...")

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
                    cached_max_date = cached_df.index.max()

                    normalized_df = self._normalize_dataframe_timezone(df)

                    # Merge with cached data (keep only new rows)
                    new_rows = normalized_df[normalized_df.index > cached_max_date]

                    if not new_rows.empty:
                        merged_df = pd.concat([cached_df, new_rows])
                        merged_df = merged_df.sort_index()
                        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]

                        self._save_intraday_to_cache(symbol, merged_df)
                        self.intraday_data_cache[symbol] = merged_df
                        updated_incremental += 1
                    else:
                        # No new data, use cached
                        self.intraday_data_cache[symbol] = cached_df
                        loaded_from_cache += 1

            await self._fetch_batch(
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

            self.logger.info(f"  🔄 Phase 3 complete: {updated_incremental} updated")

        # ========== Summary ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 Intraday Data Summary:")
        self.logger.info(f"   • Loaded from cache: {loaded_from_cache}")
        self.logger.info(f"   • Fetched full: {fetched_full}")
        self.logger.info(f"   • Updated incremental: {updated_incremental}")
        self.logger.info(f"   • Failed: {failed}")
        self.logger.info(f"   • Total in memory cache: {len(self.intraday_data_cache)}")
        self.logger.info("=" * 80)

    def get_price_at_timestamp(
        self, daily_df: pd.DataFrame, intraday_df: Optional[pd.DataFrame], timestamp: datetime, session: MarketSession
    ) -> Tuple[Optional[float], str]:
        """
        Get price at specific timestamp based on market session.

        Args:
            daily_df: Daily OHLCV data
            intraday_df: Intraday OHLCV data (15-min bars)
            timestamp: Tweet timestamp
            session: Market session enum value

        Returns:
            Tuple of (price, data_quality_flag)
        """
        # Ensure timestamp is timezone-aware
        timestamp = self._normalize_timestamp(timestamp)

        # For regular hours and extended hours
        if session in [MarketSession.REGULAR, MarketSession.PREMARKET, MarketSession.AFTERHOURS]:
            if intraday_df is not None and not intraday_df.empty:
                # Find closest bar (within 15 minutes for 15-min bars)
                time_diff = abs(intraday_df.index - timestamp)
                min_diff = time_diff.min()

                if min_diff <= pd.Timedelta(minutes=15):
                    closest_idx = time_diff.argmin()
                    closest_time = intraday_df.index[closest_idx]
                    price = intraday_df.iloc[closest_idx]["close"]
                    self.logger.debug(f"price_at_tweet: ${price:.2f} at {closest_time} (tweet: {timestamp}, diff: {min_diff})")
                    return price, f"{session.value}_intraday"

            # Fallback to daily close
            if not daily_df.empty:
                price = daily_df.iloc[-1]["close"]
                return price, f"no_{session.value}_data_used_daily"

        # For closed market (overnight, weekends)
        elif session == MarketSession.CLOSED:
            if not daily_df.empty:
                # Use previous close
                price = daily_df.iloc[-1]["close"]
                return price, "market_closed_used_prev_close"

        return None, "no_data_available"

    def get_price_n_hr_after(
        self, intraday_df: Optional[pd.DataFrame], daily_df: pd.DataFrame, timestamp: datetime, hours_after: float = 1.0
    ) -> Tuple[Optional[float], str]:
        """
        Get price N hours after tweet.

        Args:
            intraday_df: Intraday OHLCV data
            daily_df: Daily OHLCV data
            timestamp: Tweet timestamp
            hours_after: Hours after tweet to get price (default: 1.0)

        Returns:
            Tuple of (price, data_quality_flag)
        """
        timestamp = self._normalize_timestamp(timestamp)
        target_time = timestamp + timedelta(hours=hours_after)

        if intraday_df is not None and not intraday_df.empty:
            # Find closest bar after target time (within 20 minutes for 15-min bars)
            future_bars = intraday_df[intraday_df.index >= target_time]

            if not future_bars.empty:
                closest_bar = future_bars.iloc[0]
                time_diff = abs(closest_bar.name - target_time)

                if time_diff <= pd.Timedelta(minutes=20):
                    price = closest_bar["close"]
                    bar_time = closest_bar.name
                    self.logger.debug(
                        f"price_{hours_after}hr_after: ${price:.2f} at {bar_time} (target: {target_time}, diff: {time_diff})"
                    )
                    return price, f"{hours_after}hr_after_intraday"

        # Fallback: use next day's open or current close
        if not daily_df.empty:
            return daily_df.iloc[-1]["close"], f"{hours_after}hr_after_unavailable_used_close"

        return None, "no_data_available"

    async def enrich_tweet(self, tweet_row: pd.Series, max_date: datetime) -> dict:
        """
        Enrich a single tweet with all required features.

        Args:
            tweet_row: Row from tweets DataFrame
            max_date: Maximum date in dataset (for fetching full range)

        Returns:
            Dictionary with all enriched features
        """
        ticker = tweet_row["ticker"]
        timestamp = pd.to_datetime(tweet_row["timestamp"])

        # ⚠️ CRITICAL: Normalize timestamp to ET BEFORE any calculations to avoid tz-naive/tz-aware comparison errors
        timestamp = self._normalize_timestamp(timestamp)

        self.logger.info(f"Processing {ticker} at {timestamp}")

        # Fetch FULL date range for ticker (cached if already fetched)
        daily_df_full = await self.fetch_daily_data(ticker, max_date)

        if daily_df_full is None or daily_df_full.empty:
            self.logger.warning(f"No data available for {ticker}")
            return self._get_empty_features()

        # ⚠️ CRITICAL: Slice to only include data UP TO tweet date (no look-ahead!)
        tweet_date = timestamp.date()
        daily_df = daily_df_full[daily_df_full.index.date <= tweet_date].copy()

        if daily_df.empty:
            self.logger.warning(f"No daily data before/on {tweet_date} for {ticker}")
            return self._get_empty_features()

        # Fetch FULL intraday range for ticker (cached if already fetched)
        intraday_df_full = await self.fetch_intraday_data(ticker, max_date)

        # ⚠️ CRITICAL: Slice intraday to only include data UP TO 1 hour after tweet (for price_1hr_after)
        # We need data up to tweet + 1 hour to calculate "price 1hr after"
        # timestamp is already timezone-aware (normalized above), so max_intraday_time will be too
        max_intraday_time = timestamp + timedelta(hours=1, minutes=15)  # +15 min buffer for 15-min bars
        intraday_df = None
        if intraday_df_full is not None and not intraday_df_full.empty:
            intraday_df = intraday_df_full[intraday_df_full.index <= max_intraday_time].copy()

        # Determine market session
        session = self.get_market_session(timestamp)

        # Get price at tweet time
        price_at_tweet, price_flag = self.get_price_at_timestamp(daily_df, intraday_df, timestamp, session)

        # Get price 1hr after
        price_1hr_after, price_1hr_flag = self.get_price_n_hr_after(intraday_df, daily_df, timestamp, hours_after=1.0)

        # Find closest daily bar for indicator calculation
        tweet_date = timestamp.date()
        daily_df_dates = daily_df.index.date

        # Get the most recent bar before or on tweet date
        valid_indices = [i for i, d in enumerate(daily_df_dates) if d <= tweet_date]

        if not valid_indices:
            self.logger.warning(f"No daily data before {tweet_date} for {ticker}")
            return self._get_empty_features()

        current_idx = valid_indices[-1]

        # Calculate technical indicators
        indicators = self.tech_indicators.calculate_all_indicators(daily_df, current_idx)

        # Fetch SPY FULL data (cached if already fetched)
        spy_df_full = await self.fetch_daily_data("SPY", max_date)
        spy_return_1d = None

        if spy_df_full is not None and not spy_df_full.empty:
            # ⚠️ CRITICAL: Slice SPY data to only include data UP TO tweet date
            spy_df = spy_df_full[spy_df_full.index.date <= tweet_date].copy()

            if not spy_df.empty:
                spy_dates = spy_df.index.date
                spy_valid_indices = [i for i, d in enumerate(spy_dates) if d <= tweet_date]

                if spy_valid_indices:
                    spy_idx = spy_valid_indices[-1]
                    self.logger.debug(f"SPY: Found {len(spy_df)} bars, using index {spy_idx} (date: {spy_dates[spy_idx]})")
                    spy_return_1d = self.tech_indicators.calculate_return(spy_df, spy_idx, periods=1)
                    self.logger.debug(f"SPY return_1d: {spy_return_1d}")
                else:
                    self.logger.warning(f"No SPY data found for date <= {tweet_date}")
            else:
                self.logger.warning(f"No SPY data before/on {tweet_date}")
        else:
            self.logger.warning("Failed to fetch SPY data or SPY data is empty")

        # Fetch SPY FULL intraday data (cached if already fetched)
        spy_intraday_df_full = await self.fetch_intraday_data("SPY", max_date)

        # Calculate SPY 1-hour return (same time window as stock)
        spy_return_1hr = None
        spy_price_at_tweet = None
        spy_price_1hr_after = None

        # ⚠️ CRITICAL: Slice SPY intraday to only include data UP TO 1 hour after tweet
        spy_intraday_df = None
        if spy_intraday_df_full is not None and not spy_intraday_df_full.empty:
            spy_intraday_df = spy_intraday_df_full[spy_intraday_df_full.index <= max_intraday_time].copy()

        if spy_df_full is not None and not spy_df_full.empty:
            # Use SLICED SPY data (already filtered above)
            spy_session = self.get_market_session(timestamp)
            spy_price_at_tweet, _ = self.get_price_at_timestamp(spy_df, spy_intraday_df, timestamp, spy_session)

            # Get SPY price 1hr after
            spy_price_1hr_after, _ = self.get_price_n_hr_after(spy_intraday_df, spy_df, timestamp, hours_after=1.0)

            # Calculate SPY 1hr return
            if spy_price_at_tweet and spy_price_1hr_after:
                spy_return_1hr = (spy_price_1hr_after - spy_price_at_tweet) / spy_price_at_tweet
                self.logger.debug(f"SPY return_1hr: {spy_return_1hr} ({spy_price_at_tweet} -> {spy_price_1hr_after})")
            else:
                self.logger.warning("Could not calculate SPY 1hr return - missing prices")

        # Calculate returns
        return_1hr = None
        return_1hr_adjusted = None

        if price_at_tweet and price_1hr_after:
            return_1hr = (price_1hr_after - price_at_tweet) / price_at_tweet

            if spy_return_1hr is not None:
                # Market-adjusted return (subtract SPY return over SAME timeframe)
                return_1hr_adjusted = return_1hr - spy_return_1hr
                self.logger.debug(f"Adjustment: {return_1hr:.6f} - {spy_return_1hr:.6f} = {return_1hr_adjusted:.6f}")
            else:
                return_1hr_adjusted = return_1hr  # If no SPY data, use raw return
                self.logger.warning("Using unadjusted return (no SPY 1hr data)")

        # Classify into 5 classes based on return_1hr_adjusted
        label_5class = self._classify_return(return_1hr_adjusted)

        # Build result dictionary
        result = {
            "ticker": ticker,
            "timestamp": timestamp,
            "session": session.value if session else None,  # Convert enum to string
            "price_at_tweet": price_at_tweet,
            "price_at_tweet_flag": price_flag,
            "return_1d": indicators["return_1d"],
            "volatility_7d": indicators["volatility_7d"],
            "relative_volume": indicators["relative_volume"],
            "rsi_14": indicators["rsi_14"],
            "distance_from_ma_20": indicators["distance_from_ma_20"],
            "spy_return_1d": spy_return_1d,
            "spy_return_1hr": spy_return_1hr,
            "price_1hr_after": price_1hr_after,
            "price_1hr_after_flag": price_1hr_flag,
            "return_1hr": return_1hr,
            "return_1hr_adjusted": return_1hr_adjusted,
            "label_5class": label_5class,
        }

        return result

    def _classify_return(self, return_value: Optional[float]) -> Optional[str]:
        """
        Classify return into 5 classes.

        Args:
            return_value: Return value to classify

        Returns:
            Class label: 'strong_sell', 'sell', 'hold', 'buy', 'strong_buy'
        """
        if return_value is None:
            return None

        if return_value < -0.02:  # < -2%
            return "strong_sell"
        elif return_value < -0.005:  # -2% to -0.5%
            return "sell"
        elif return_value < 0.005:  # -0.5% to 0.5%
            return "hold"
        elif return_value < 0.02:  # 0.5% to 2%
            return "buy"
        else:  # > 2%
            return "strong_buy"

    def _get_empty_features(self) -> dict:
        """Return empty features dictionary."""
        return {
            "ticker": None,
            "timestamp": None,
            "session": None,
            "price_at_tweet": None,
            "price_at_tweet_flag": "no_data",
            "return_1d": None,
            "volatility_7d": None,
            "relative_volume": None,
            "rsi_14": None,
            "distance_from_ma_20": None,
            "spy_return_1d": None,
            "spy_return_1hr": None,
            "price_1hr_after": None,
            "price_1hr_after_flag": "no_data",
            "return_1hr": None,
            "return_1hr_adjusted": None,
            "label_5class": None,
        }


async def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(__name__)

    # Read tweets CSV
    tweets_file = "output/tweets.csv"
    logger.info(f"Reading tweets from {tweets_file}")

    try:
        tweets_df = pd.read_csv(tweets_file)
        logger.info(f"Loaded {len(tweets_df)} tweets")
    except Exception as e:
        logger.error(f"Error reading tweets file: {e}")
        return

    # For demo, process only first few tweets
    # sample_size = 50
    # tweets_sample = tweets_df.head(sample_size)
    tweets_sample = tweets_df
    logger.info(f"Processing {len(tweets_sample)} tweets as sample...")

    # Calculate max date from dataset (for fetching full date ranges)
    tweets_sample['timestamp'] = pd.to_datetime(tweets_sample['timestamp'])
    max_date = tweets_sample['timestamp'].max() + timedelta(days=2)  # +2 days buffer for "1hr after"

    # Ensure max_date is timezone-aware (localize to ET if naive)
    if max_date.tzinfo is None:
        max_date = ET.localize(max_date.to_pydatetime())
    elif max_date.tzinfo != ET:
        max_date = max_date.tz_convert(ET)

    # Get last market trading day to ensure we don't exceed it
    nyse = mcal.get_calendar('NYSE')
    now_et = datetime.now(ET)
    today = now_et.date()
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    if not schedule.empty:
        last_market_day = schedule.index[-1].date()

        # If today is a trading day but market hasn't closed yet, use previous trading day
        if last_market_day == today:
            current_time_minutes = now_et.hour * 60 + now_et.minute
            if current_time_minutes < MARKET_CLOSE:  # Market hasn't closed yet (before 4:00 PM ET)
                logger.info(f"⏰ Market not closed yet (current time: {now_et.strftime('%H:%M')} ET)")
                # Get previous trading day
                if len(schedule) > 1:
                    last_market_day = schedule.index[-2].date()
                    logger.info(f"📅 Using previous trading day: {last_market_day}")
                else:
                    # Need to extend search range
                    extended_schedule = nyse.schedule(start_date=today - timedelta(days=30), end_date=today)
                    if len(extended_schedule) > 1:
                        last_market_day = extended_schedule.index[-2].date()
                        logger.info(f"📅 Using previous trading day: {last_market_day}")

        last_market_datetime = ET.localize(datetime.combine(last_market_day, datetime.max.time()))
        logger.info(f"🕐 last_market_datetime: {last_market_datetime} (tzinfo: {last_market_datetime.tzinfo})")
        logger.info(f"🕐 max_date before min(): {max_date} (tzinfo: {max_date.tzinfo})")
        max_date = min(max_date, last_market_datetime)
        logger.info(f"📅 Last closed market trading day: {last_market_day}")
        logger.info(f"🕐 max_date after min(): {max_date} (tzinfo: {max_date.tzinfo})")
    else:
        # Fallback to current date if no market data available
        max_date = min(max_date, datetime.now(ET))
        logger.warning("⚠️ Could not determine last market day, using current date")

    min_date = tweets_sample['timestamp'].min()

    logger.info(f"📅 Dataset date range: {min_date.date()} to {max_date.date()}")

    # Extract ALL unique tickers (including SPY)
    unique_tickers_list = tweets_sample['ticker'].unique().tolist()
    unique_tickers_list.append('SPY')  # Add SPY for market adjustment
    unique_tickers_list = list(set(unique_tickers_list))  # Remove duplicates

    # Filter out excluded tickers (problematic symbols that consistently fail)
    excluded_count = len([t for t in unique_tickers_list if t in EXCLUDED_TICKERS])
    if excluded_count > 0:
        logger.info(f"⚠️ Excluding {excluded_count} problematic tickers: {[t for t in unique_tickers_list if t in EXCLUDED_TICKERS]}")
    unique_tickers_list = [t for t in unique_tickers_list if t not in EXCLUDED_TICKERS]

    logger.info("\n" + "=" * 80)
    logger.info("🚀 ULTIMATE OPTIMIZATION: Pre-Fetch ALL Tickers Before Processing!")
    logger.info("=" * 80)
    logger.info(f"📊 Found {len(unique_tickers_list)} unique tickers in dataset")
    logger.info("📊 Using 15-min bars (±7.5 min precision, supports up to 120 days)")
    logger.info("💡 Strategy: Fetch ALL tickers first, then process tweets (all cache hits!)")
    logger.info("")

    # Initialize enricher
    enricher = TweetEnricher()

    # Connect to IB
    connected = await enricher.connect()
    if not connected:
        logger.error("Failed to connect to IB")
        return

    try:
        # ========== PHASE 1: PRE-FETCH ALL TICKER DATA ==========
        logger.info("=" * 80)
        logger.info("PHASE 1: Pre-fetching ALL ticker data (daily + intraday)")
        logger.info("=" * 80)
        logger.info(f"📦 Fetching {len(unique_tickers_list)} tickers (batch size: 50)")
        logger.info("")

        # Pre-fetch daily data for all tickers
        logger.info("📊 Step 1/2: Fetching daily data (3 M, 1 day bars)...")
        await enricher.prefetch_all_daily_data(unique_tickers_list, max_date)

        # Pre-fetch intraday data for all tickers
        logger.info("📊 Step 2/2: Fetching intraday data (60 D, 15 min bars)...")
        await enricher.prefetch_all_intraday_data(unique_tickers_list, max_date)

        logger.info("")
        logger.info(f"✅ Phase 1 complete! Cached data for {len(unique_tickers_list)} tickers")
        logger.info(f"   • Daily cache: {len(enricher.daily_data_cache)} tickers")
        logger.info(f"   • Intraday cache: {len(enricher.intraday_data_cache)} tickers")
        logger.info("=" * 80 + "\n")

        # ========== PHASE 2: PROCESS ALL TWEETS ==========
        logger.info("PHASE 2: Processing tweets (all data cached!)")
        logger.info("=" * 80)
        logger.info(f"📝 Processing {len(tweets_sample)} tweets...")
        logger.info("")

        results = []

        for idx, tweet_row in tweets_sample.iterrows():
            result = await enricher.enrich_tweet(tweet_row, max_date)

            # Add original tweet data to result
            result["author"] = tweet_row["author"]
            result["category"] = tweet_row["category"]
            result["tweet_url"] = tweet_row["tweet_url"]
            result["text"] = tweet_row["text"]

            results.append(result)

            # Print concise summary
            status = "✅" if result["price_at_tweet"] is not None else "❌"
            label = result["label_5class"] or "N/A"
            ret_adj = result["return_1hr_adjusted"]
            return_str = f"{ret_adj:.2%}" if ret_adj is not None else "N/A"
            logger.info(f"{status} [{idx + 1}/{len(tweets_sample)}] {result['ticker']} | {result['session']} | {label} | {return_str}")

        # Save results to CSV
        if results:
            output_df = pd.DataFrame(results)

            # Reorder columns for readability
            column_order = [
                "timestamp",
                "ticker",
                "author",
                "category",
                "session",
                "price_at_tweet",
                "price_at_tweet_flag",
                "price_1hr_after",
                "price_1hr_after_flag",
                "return_1hr",
                "return_1hr_adjusted",
                "label_5class",
                "return_1d",
                "volatility_7d",
                "relative_volume",
                "rsi_14",
                "distance_from_ma_20",
                "spy_return_1d",
                "spy_return_1hr",
                "tweet_url",
                "text",
            ]

            # Only include columns that exist
            column_order = [col for col in column_order if col in output_df.columns]
            output_df = output_df[column_order]

            output_file = "output/enriched_sample.csv"
            output_df.to_csv(output_file, index=False)
            logger.info(f"\n✅ Saved {len(results)} enriched tweets to: {output_file}")

            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total processed: {len(results)}")
            logger.info(f"Successful: {sum(1 for r in results if r['price_at_tweet'] is not None)}")
            logger.info(f"Failed: {sum(1 for r in results if r['price_at_tweet'] is None)}")

            # Label distribution
            labels = [r["label_5class"] for r in results if r["label_5class"] is not None]
            if labels:
                from collections import Counter
                label_counts = Counter(labels)
                logger.info("\nLabel distribution:")
                for label, count in sorted(label_counts.items()):
                    logger.info(f"  {label}: {count}")

    finally:
        await enricher.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
