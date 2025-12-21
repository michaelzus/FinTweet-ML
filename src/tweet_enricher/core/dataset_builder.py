"""Dataset builder for creating training datasets from cached data.

This module processes tweets and cached OHLCV data to create training-ready
datasets. Unlike TweetEnricher, this class has NO external API dependencies
and works entirely offline with cached data.
"""

import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache_reader import CacheReader, ValidationReport
from tweet_enricher.data.stock_metadata import StockMetadataCache
from tweet_enricher.market.regime import get_market_regime
from tweet_enricher.market.session import (
    MarketSession,
    get_market_session,
    normalize_timestamp,
)

logger = logging.getLogger(__name__)

# Required columns for reliable training data
MINIMUM_REQUIRED_COLUMNS = [
    # Core
    "text",
    "label_1d_3class",
    "tweet_hash",
    # Numerical Features (10)
    "volatility_7d",
    "relative_volume",
    "rsi_14",
    "distance_from_ma_20",
    "return_5d",
    "return_20d",
    "above_ma_20",
    "slope_ma_20",
    "gap_open",
    "intraday_range",
    # Categorical Features (6)
    "author",
    "category",
    "session",
    "market_regime",
    "sector",
    "market_cap_bucket",
]

# Output column order for readability
OUTPUT_COLUMN_ORDER = [
    # Traceability
    "timestamp",
    "ticker",
    "tweet_url",
    # Target
    "label_1d_3class",
    # Features - Categorical
    "author",
    "category",
    "session",
    "market_regime",
    "sector",
    "market_cap_bucket",
    # Features - Numerical
    "volatility_7d",
    "relative_volume",
    "rsi_14",
    "distance_from_ma_20",
    "return_5d",
    "return_20d",
    "above_ma_20",
    "slope_ma_20",
    "gap_open",
    "intraday_range",
    # Data quality
    "is_reliable_label",
    "tweet_hash",
    # Debug (prices)
    "entry_price",
    "entry_price_flag",
    "price_next_open",
    "price_next_open_flag",
    "return_to_next_open",
    # Original tweet data
    "text",
]


class DatasetBuilder:
    """
    Build training datasets from cached OHLCV data and tweets.

    This class has NO external API dependencies and works entirely offline.
    All data must be pre-fetched and cached before using this builder.

    Usage:
        cache = CacheReader()
        builder = DatasetBuilder(cache, TechnicalIndicators(), StockMetadataCache())

        # Validate coverage first
        report = builder.validate_coverage(tweets_df)
        if report.missing_tickers:
            print(f"Missing data for: {report.missing_tickers}")

        # Build dataset
        dataset_df = builder.build(tweets_df)
    """

    def __init__(
        self,
        cache: CacheReader,
        indicators: TechnicalIndicators,
        metadata_cache: Optional[StockMetadataCache] = None,
    ):
        """
        Initialize the DatasetBuilder.

        Args:
            cache: CacheReader instance for reading cached OHLCV data
            indicators: TechnicalIndicators instance for calculating indicators
            metadata_cache: StockMetadataCache instance for stock metadata (optional)
        """
        self.cache = cache
        self.indicators = indicators
        self.metadata_cache = metadata_cache or StockMetadataCache()

    def validate_coverage(
        self,
        tweets_df: pd.DataFrame,
        require_intraday: bool = True,
    ) -> ValidationReport:
        """
        Validate that all required OHLCV data is available.

        Args:
            tweets_df: DataFrame with tweets (must have 'ticker' column)
            require_intraday: Whether intraday data is required

        Returns:
            ValidationReport with coverage details and missing tickers
        """
        # Get unique tickers from tweets
        unique_tickers = tweets_df["ticker"].unique().tolist()

        # Always need SPY for market regime calculation
        if "SPY" not in unique_tickers:
            unique_tickers.append("SPY")

        return self.cache.validate_coverage(unique_tickers, require_intraday)

    def build(
        self,
        tweets_df: pd.DataFrame,
        skip_missing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build enriched dataset from tweets and cached OHLCV data.

        Args:
            tweets_df: DataFrame with tweets (required columns: timestamp, ticker, author, category, text)
            skip_missing: If True, skip tweets with missing OHLCV data. If False, raise error.
            verbose: If True, log progress information

        Returns:
            DataFrame with enriched data ready for training
        """
        if verbose:
            logger.info(f"Building dataset from {len(tweets_df)} tweets...")

        # Preload data for all tickers
        unique_tickers = tweets_df["ticker"].unique().tolist()
        if "SPY" not in unique_tickers:
            unique_tickers.append("SPY")

        self.cache.preload_symbols(unique_tickers)

        # Process each tweet
        results = []
        processed = 0
        skipped = 0

        for idx, tweet_row in tweets_df.iterrows():
            result = self._process_tweet(tweet_row)

            if result["entry_price"] is None and skip_missing:
                skipped += 1
                continue

            # Add original tweet data
            result["author"] = tweet_row["author"]
            result["category"] = tweet_row["category"]
            result["tweet_url"] = tweet_row.get("tweet_url", "")
            result["text"] = tweet_row["text"]

            results.append(result)
            processed += 1

            if verbose and processed % 100 == 0:
                logger.info(f"Processed {processed} tweets...")

        if verbose:
            logger.info(f"Dataset built: {processed} successful, {skipped} skipped")

        # Create output DataFrame
        output_df = pd.DataFrame(results)

        # Reorder columns for readability
        column_order = [col for col in OUTPUT_COLUMN_ORDER if col in output_df.columns]
        output_df = output_df[column_order]

        return output_df

    def _process_tweet(self, tweet_row: pd.Series) -> dict:
        """
        Process a single tweet and compute all features.

        Args:
            tweet_row: Row from tweets DataFrame

        Returns:
            Dictionary with all enriched features
        """
        ticker = tweet_row["ticker"]
        timestamp = pd.to_datetime(tweet_row["timestamp"])

        # CRITICAL: Normalize timestamp to ET BEFORE any calculations
        timestamp = normalize_timestamp(timestamp)

        # Get daily data from cache
        daily_df_full = self.cache.get_daily(ticker)

        if daily_df_full is None or daily_df_full.empty:
            logger.debug(f"No daily data for {ticker}")
            return self._get_empty_features()

        # Keep full data for next-open calculation (includes future)
        daily_df_for_next_open = daily_df_full.copy()

        # CRITICAL: Slice to only include data UP TO tweet date (no look-ahead for features!)
        tweet_date = timestamp.date()
        daily_df = daily_df_full[daily_df_full.index.date < tweet_date].copy()

        if daily_df.empty:
            logger.debug(f"No daily data before {tweet_date} for {ticker}")
            return self._get_empty_features()

        # Get intraday data from cache - used for entry price
        intraday_df = self.cache.get_intraday(ticker)

        # Determine market session
        session = get_market_session(timestamp)

        # Get entry price (first available price AFTER tweet - realistic execution)
        entry_price, entry_price_flag = self._get_entry_price(
            intraday_df, daily_df_for_next_open, timestamp, session
        )

        # Find closest daily bar for indicator calculation
        daily_df_dates = daily_df.index.date

        # Get the most recent bar before or on tweet date
        valid_indices = [i for i, d in enumerate(daily_df_dates) if d <= tweet_date]

        if not valid_indices:
            logger.debug(f"No daily data before {tweet_date} for {ticker}")
            return self._get_empty_features()

        current_idx = valid_indices[-1]

        # Calculate technical indicators
        indicators = self.indicators.calculate_all_indicators(daily_df, current_idx)

        # Get SPY data from cache for market regime
        spy_df_full = self.cache.get_daily("SPY")
        market_regime = None

        if spy_df_full is not None and not spy_df_full.empty:
            # CRITICAL: Slice SPY data to only include data BEFORE tweet date (strict)
            spy_df = spy_df_full[spy_df_full.index.date < tweet_date].copy()

            if not spy_df.empty:
                market_regime = get_market_regime(spy_df, timestamp)

        # Compute tweet hash for duplicate detection in train/test splits
        tweet_hash = self._compute_tweet_hash(tweet_row["text"])

        # Get next trading day's open price for 1-day labels
        price_next_open, price_next_open_flag = self._get_price_next_open(
            daily_df_for_next_open, timestamp
        )

        # Calculate return to next open (using entry price as base)
        return_to_next_open = None
        if entry_price and price_next_open:
            return_to_next_open = (price_next_open - entry_price) / entry_price

        # Classify 1-day label
        label_1d_3class = self._classify_return_3class(return_to_next_open)

        # Check if label data is reliable (entry price + next open available)
        price_reliable = self._is_reliable_label(entry_price_flag, price_next_open_flag)

        # Get stock metadata (sector, market cap)
        stock_metadata = self.metadata_cache.get_metadata(ticker)

        # Build result dictionary
        result = {
            "ticker": ticker,
            "timestamp": timestamp,
            "session": session.value if session else None,
            "market_regime": market_regime,
            "sector": stock_metadata.get("sector"),
            "market_cap_bucket": stock_metadata.get("market_cap_bucket"),
            # Original tweet data (needed for feature validation)
            "author": tweet_row["author"],
            "category": tweet_row["category"],
            "text": tweet_row["text"],
            # Entry price (first available AFTER tweet - realistic execution)
            "entry_price": entry_price,
            "entry_price_flag": entry_price_flag,
            # Technical indicators (backward-looking from day before tweet)
            "return_5d": indicators["return_5d"],
            "return_20d": indicators["return_20d"],
            "volatility_7d": indicators["volatility_7d"],
            "relative_volume": indicators["relative_volume"],
            "rsi_14": indicators["rsi_14"],
            "distance_from_ma_20": indicators["distance_from_ma_20"],
            "above_ma_20": indicators["above_ma_20"],
            "slope_ma_20": indicators["slope_ma_20"],
            "gap_open": indicators["gap_open"],
            "intraday_range": indicators["intraday_range"],
            "tweet_hash": tweet_hash,
            # 1-day labels (next trading day open)
            "price_next_open": price_next_open,
            "price_next_open_flag": price_next_open_flag,
            "return_to_next_open": return_to_next_open,
            "label_1d_3class": label_1d_3class,
        }

        # Final reliability check: prices reliable AND all required features present
        result["is_reliable_label"] = price_reliable and self._has_required_features(result)

        return result

    def _get_entry_price(
        self,
        intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        timestamp: datetime,
        session: MarketSession,
    ) -> Tuple[Optional[float], str]:
        """
        Get realistic entry price - the first price available AFTER tweet.

        Uses the OPEN of the first bar that starts AFTER the tweet timestamp.
        This represents the earliest realistic fill price.

        Args:
            intraday_df: Intraday OHLCV data (15-min bars)
            daily_df: Daily OHLCV data
            timestamp: Tweet timestamp
            session: Market session enum value

        Returns:
            Tuple of (price, data_quality_flag)
        """
        timestamp = normalize_timestamp(timestamp)

        # For regular hours and extended hours - use intraday data
        if session in [MarketSession.REGULAR, MarketSession.PREMARKET, MarketSession.AFTERHOURS]:
            if intraday_df is not None and not intraday_df.empty:
                # Find first bar that starts AFTER tweet (realistic entry)
                future_bars = intraday_df[intraday_df.index > timestamp]

                if not future_bars.empty:
                    entry_bar = future_bars.iloc[0]
                    price = entry_bar["open"]
                    return price, f"{session.value}_next_bar_open"

            # Fallback to next day's open
            tweet_date = timestamp.date()
            future_daily = daily_df[daily_df.index.date > tweet_date]
            if not future_daily.empty:
                price = future_daily.iloc[0]["open"]
                return price, f"{session.value}_next_day_open"

        # For closed market (overnight, weekends) - use next trading day open
        elif session == MarketSession.CLOSED:
            tweet_date = timestamp.date()
            future_daily = daily_df[daily_df.index.date > tweet_date]
            if not future_daily.empty:
                price = future_daily.iloc[0]["open"]
                return price, "closed_next_day_open"

        return None, "no_data_available"

    def _get_price_next_open(
        self,
        daily_df: pd.DataFrame,
        timestamp: datetime,
    ) -> Tuple[Optional[float], str]:
        """
        Get next trading day's open price after tweet.

        Args:
            daily_df: Daily OHLCV data (must include data AFTER tweet date)
            timestamp: Tweet timestamp

        Returns:
            Tuple of (price, data_quality_flag)
        """
        timestamp = normalize_timestamp(timestamp)
        tweet_date = timestamp.date()

        # Find first trading day AFTER tweet date
        future_bars = daily_df[daily_df.index.date > tweet_date]

        if not future_bars.empty:
            next_day = future_bars.iloc[0]
            return next_day["open"], "next_open_available"

        return None, "next_open_unavailable"

    def _classify_return_3class(self, return_value: Optional[float]) -> Optional[str]:
        """
        Classify return into 3 classes (SELL/HOLD/BUY).

        Thresholds: SELL < -0.5%, HOLD -0.5% to +0.5%, BUY > +0.5%

        Args:
            return_value: Return value to classify

        Returns:
            Class label: 'SELL', 'HOLD', 'BUY'
        """
        if return_value is None:
            return None

        if return_value < -0.005:  # < -0.5%
            return "SELL"
        elif return_value < 0.005:  # -0.5% to 0.5%
            return "HOLD"
        else:  # > 0.5%
            return "BUY"

    def _is_reliable_label(self, entry_price_flag: str, next_open_flag: str) -> bool:
        """
        Check if 1-day label is based on reliable price data.

        Args:
            entry_price_flag: Data quality flag for entry price
            next_open_flag: Data quality flag for next trading day open

        Returns:
            True if entry price is from intraday data and next open is available
        """
        entry_reliable_patterns = ["next_bar_open", "intraday"]
        entry_unreliable_patterns = ["next_day", "unavailable", "no_data", "closed"]

        for pattern in entry_unreliable_patterns:
            if pattern in entry_price_flag:
                return False

        entry_reliable = any(p in entry_price_flag for p in entry_reliable_patterns)
        next_open_reliable = next_open_flag == "next_open_available"

        return entry_reliable and next_open_reliable

    def _has_required_features(self, result: dict) -> bool:
        """
        Check if all required columns for training are present and non-None.

        Args:
            result: Enriched tweet result dictionary

        Returns:
            True if all required columns have valid values, False otherwise
        """
        for col in MINIMUM_REQUIRED_COLUMNS:
            if col not in result or result[col] is None:
                logger.debug(f"Missing required column: {col}")
                return False
        return True

    def _compute_tweet_hash(self, text: str) -> str:
        """
        Compute hash of tweet text for duplicate detection in train/test splits.

        Args:
            text: Tweet text content

        Returns:
            12-character MD5 hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _get_empty_features(self) -> dict:
        """Return empty features dictionary."""
        return {
            "ticker": None,
            "timestamp": None,
            "session": None,
            "market_regime": None,
            "sector": None,
            "market_cap_bucket": None,
            # Original tweet data
            "author": None,
            "category": None,
            "text": None,
            # Entry price (first available price AFTER tweet)
            "entry_price": None,
            "entry_price_flag": "no_data",
            # Technical indicators
            "return_5d": None,
            "return_20d": None,
            "volatility_7d": None,
            "relative_volume": None,
            "rsi_14": None,
            "distance_from_ma_20": None,
            "above_ma_20": None,
            "slope_ma_20": None,
            "gap_open": None,
            "intraday_range": None,
            "is_reliable_label": False,
            "tweet_hash": None,
            # 1-day labels
            "price_next_open": None,
            "price_next_open_flag": "no_data",
            "return_to_next_open": None,
            "label_1d_3class": None,
        }

    def get_required_tickers(self, tweets_df: pd.DataFrame) -> List[str]:
        """
        Get list of all tickers required for building the dataset.

        Args:
            tweets_df: DataFrame with tweets

        Returns:
            List of unique ticker symbols (including SPY for market regime)
        """
        tickers = tweets_df["ticker"].unique().tolist()
        if "SPY" not in tickers:
            tickers.append("SPY")
        return tickers

