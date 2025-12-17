"""Tweet enrichment with financial indicators and price information."""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache import DataCache
from tweet_enricher.data.ib_fetcher import IBHistoricalDataFetcher
from tweet_enricher.data.stock_metadata import StockMetadataCache
from tweet_enricher.market.regime import get_market_regime
from tweet_enricher.market.session import (
    MarketSession,
    get_market_session,
    normalize_timestamp,
)


class TweetEnricher:
    """
    Enriches tweet data with financial indicators and price information.

    Handles data fetching from IBKR, technical indicator calculation,
    and market session awareness.
    """

    def __init__(
        self,
        ib_fetcher: IBHistoricalDataFetcher,
        cache: DataCache,
        indicators: TechnicalIndicators,
        metadata_cache: Optional[StockMetadataCache] = None,
    ):
        """
        Initialize the TweetEnricher.

        Args:
            ib_fetcher: IBHistoricalDataFetcher instance for fetching data
            cache: DataCache instance for caching data
            indicators: TechnicalIndicators instance for calculating indicators
            metadata_cache: StockMetadataCache instance for stock metadata (optional)
        """
        self.ib_fetcher = ib_fetcher
        self.cache = cache
        self.indicators = indicators
        self.metadata_cache = metadata_cache or StockMetadataCache()
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """Establish connection to Interactive Brokers."""
        return await self.ib_fetcher.connect()

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        await self.ib_fetcher.disconnect()

    def get_entry_price(
        self,
        intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        timestamp: datetime,
        session: MarketSession,
    ) -> Tuple[Optional[float], str]:
        """
        Get realistic entry price - the first price available AFTER tweet.

        Uses the OPEN of the first bar that starts AFTER the tweet timestamp.
        This represents the earliest realistic fill price (you can't buy at
        a price that existed before you saw the tweet).

        Args:
            intraday_df: Intraday OHLCV data (15-min bars, should include future data)
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
                    bar_time = entry_bar.name
                    price = entry_bar["open"]
                    delay = bar_time - timestamp
                    self.logger.debug(
                        f"entry_price: ${price:.2f} at {bar_time} open "
                        f"(tweet: {timestamp}, delay: {delay})"
                    )
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

    def get_price_n_hr_after(
        self,
        intraday_df: Optional[pd.DataFrame],
        daily_df: pd.DataFrame,
        timestamp: datetime,
        hours_after: float = 1.0,
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
        timestamp = normalize_timestamp(timestamp)
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
                    self.logger.debug(f"price_{hours_after}hr_after: ${price:.2f} at {bar_time} (target: {target_time}, diff: {time_diff})")
                    return price, f"{hours_after}hr_after_intraday"

        # Fallback: use next day's open or current close
        if not daily_df.empty:
            return daily_df.iloc[-1]["close"], f"{hours_after}hr_after_unavailable_used_close"

        return None, "no_data_available"

    def get_price_next_open(
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

    def _classify_return_3class(self, return_value: Optional[float]) -> Optional[str]:
        """
        Classify return into 3 classes (SELL/HOLD/BUY).

        Thresholds: SELL < -0.5%, HOLD -0.5% to +0.5%, BUY > +0.5%
        Same thresholds used for both 1hr and 1day labels.

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

    def _is_reliable_label(self, entry_price_flag: str, exit_price_flag: str) -> bool:
        """
        Check if 1hr label is based on real intraday prices (not fallbacks).

        Args:
            entry_price_flag: Data quality flag for entry price
            exit_price_flag: Data quality flag for exit price (1hr after)

        Returns:
            True if both prices are from real intraday data, False otherwise
        """
        # Reliable flags contain "next_bar" or "intraday" (actual 15-min bar data)
        reliable_patterns = ["next_bar_open", "intraday"]
        unreliable_patterns = ["next_day", "unavailable", "no_data", "closed"]

        for pattern in unreliable_patterns:
            if pattern in entry_price_flag or pattern in exit_price_flag:
                return False

        # At least one should have reliable intraday data
        entry_reliable = any(p in entry_price_flag for p in reliable_patterns)
        exit_reliable = any(p in exit_price_flag for p in reliable_patterns)

        return entry_reliable and exit_reliable

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
            # Entry price (first available price AFTER tweet)
            "entry_price": None,
            "entry_price_flag": "no_data",
            # Technical indicators (all backward-looking from day before tweet)
            "return_1d": None,
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
            "spy_return_1d": None,
            # 1-hour exit (for labels)
            "exit_price_1hr": None,
            "exit_price_1hr_flag": "no_data",
            "spy_return_1hr": None,
            "return_1hr": None,
            "return_1hr_adjusted": None,
            "label_5class": None,
            "label_3class": None,
            "is_reliable_label": False,
            "tweet_hash": None,
            # 1-day labels (next trading day open)
            "price_next_open": None,
            "price_next_open_flag": "no_data",
            "return_to_next_open": None,
            "label_1d_5class": None,
            "label_1d_3class": None,
        }

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

        # CRITICAL: Normalize timestamp to ET BEFORE any calculations
        timestamp = normalize_timestamp(timestamp)

        self.logger.info(f"Processing {ticker} at {timestamp}")

        # Get daily data from cache (already prefetched)
        daily_df_full = self.cache.get_daily(ticker)

        if daily_df_full is None or daily_df_full.empty:
            self.logger.warning(f"No data available for {ticker}")
            return self._get_empty_features()

        # Keep full data for next-open calculation (includes future)
        daily_df_for_next_open = daily_df_full.copy()

        # CRITICAL: Slice to only include data UP TO tweet date (no look-ahead for features!)
        tweet_date = timestamp.date()
        daily_df = daily_df_full[daily_df_full.index.date < tweet_date].copy()

        if daily_df.empty:
            self.logger.warning(f"No daily data before/on {tweet_date} for {ticker}")
            return self._get_empty_features()

        # Get intraday data from cache (already prefetched)
        intraday_df_full = self.cache.get_intraday(ticker)

        # CRITICAL: Slice intraday to only include data UP TO 1 hour after tweet
        max_intraday_time = timestamp + timedelta(hours=1, minutes=15)  # +15 min buffer
        intraday_df = None
        if intraday_df_full is not None and not intraday_df_full.empty:
            intraday_df = intraday_df_full[intraday_df_full.index <= max_intraday_time].copy()

        # Determine market session
        session = get_market_session(timestamp)

        # Get entry price (first available price AFTER tweet - realistic execution)
        entry_price, entry_price_flag = self.get_entry_price(
            intraday_df, daily_df_for_next_open, timestamp, session
        )

        # Get exit price 1hr after tweet
        exit_price_1hr, exit_price_1hr_flag = self.get_price_n_hr_after(
            intraday_df, daily_df, timestamp, hours_after=1.0
        )

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
        indicators = self.indicators.calculate_all_indicators(daily_df, current_idx)

        # Get SPY data from cache for market adjustment
        spy_df_full = self.cache.get_daily("SPY")
        spy_return_1d = None
        spy_df = None
        market_regime = None

        if spy_df_full is not None and not spy_df_full.empty:
            # CRITICAL: Slice SPY data to only include data BEFORE tweet date (strict)
            spy_df = spy_df_full[spy_df_full.index.date < tweet_date].copy()

            if not spy_df.empty:
                spy_dates = spy_df.index.date
                spy_valid_indices = [i for i, d in enumerate(spy_dates) if d <= tweet_date]

                if spy_valid_indices:
                    spy_idx = spy_valid_indices[-1]
                    self.logger.debug(f"SPY: Found {len(spy_df)} bars, using index {spy_idx} (date: {spy_dates[spy_idx]})")
                    spy_return_1d = self.indicators.calculate_return(spy_df, spy_idx, periods=1)
                    self.logger.debug(f"SPY return_1d: {spy_return_1d}")

                    # Calculate market regime
                    market_regime = get_market_regime(spy_df, timestamp)
                    self.logger.debug(f"Market regime: {market_regime}")
                else:
                    self.logger.warning(f"No SPY data found for date <= {tweet_date}")
            else:
                self.logger.warning(f"No SPY data before/on {tweet_date}")
        else:
            self.logger.warning("Failed to fetch SPY data or SPY data is empty")

        # Get SPY intraday data for 1hr return calculation
        spy_intraday_df_full = self.cache.get_intraday("SPY")

        # Calculate SPY 1-hour return (same time window as stock for market adjustment)
        spy_return_1hr = None

        # CRITICAL: Slice SPY intraday to only include data UP TO 1 hour after tweet
        spy_intraday_df = None
        if spy_intraday_df_full is not None and not spy_intraday_df_full.empty:
            spy_intraday_df = spy_intraday_df_full[spy_intraday_df_full.index <= max_intraday_time].copy()

        if spy_df_full is not None and not spy_df_full.empty:
            # Use full SPY data for entry price (needs future data for next bar open)
            spy_session = get_market_session(timestamp)
            spy_entry_price, _ = self.get_entry_price(
                spy_intraday_df, spy_df_full, timestamp, spy_session
            )

            # Get SPY exit price 1hr after
            spy_exit_price_1hr, _ = self.get_price_n_hr_after(
                spy_intraday_df, spy_df, timestamp, hours_after=1.0
            )

            # Calculate SPY 1hr return (for market adjustment)
            if spy_entry_price and spy_exit_price_1hr:
                spy_return_1hr = (spy_exit_price_1hr - spy_entry_price) / spy_entry_price
                self.logger.debug(f"SPY return_1hr: {spy_return_1hr} ({spy_entry_price} -> {spy_exit_price_1hr})")
            else:
                self.logger.warning("Could not calculate SPY 1hr return - missing prices")

        # Calculate returns (entry to exit)
        return_1hr = None
        return_1hr_adjusted = None

        if entry_price and exit_price_1hr:
            return_1hr = (exit_price_1hr - entry_price) / entry_price

            if spy_return_1hr is not None:
                # Market-adjusted return (subtract SPY return over SAME timeframe)
                return_1hr_adjusted = return_1hr - spy_return_1hr
                self.logger.debug(f"Adjustment: {return_1hr:.6f} - {spy_return_1hr:.6f} = {return_1hr_adjusted:.6f}")
            else:
                return_1hr_adjusted = return_1hr  # If no SPY data, use raw return
                self.logger.warning("Using unadjusted return (no SPY 1hr data)")

        # Classify into 5 classes based on return_1hr_adjusted
        label_5class = self._classify_return(return_1hr_adjusted)

        # Classify into 3 classes
        label_3class = self._classify_return_3class(return_1hr_adjusted)

        # Check if 1hr label is reliable (based on real intraday prices)
        is_reliable_label = self._is_reliable_label(entry_price_flag, exit_price_1hr_flag)

        # Compute tweet hash for duplicate detection in train/test splits
        tweet_hash = self._compute_tweet_hash(tweet_row["text"])

        # Get next trading day's open price for 1-day labels
        price_next_open, price_next_open_flag = self.get_price_next_open(
            daily_df_for_next_open, timestamp
        )

        # Calculate return to next open (using entry price as base)
        return_to_next_open = None
        if entry_price and price_next_open:
            return_to_next_open = (price_next_open - entry_price) / entry_price

        # Classify 1-day labels (same thresholds as 1hr)
        label_1d_5class = self._classify_return(return_to_next_open)
        label_1d_3class = self._classify_return_3class(return_to_next_open)

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
            # Entry price (first available AFTER tweet - realistic execution)
            "entry_price": entry_price,
            "entry_price_flag": entry_price_flag,
            # Technical indicators (backward-looking from day before tweet)
            "return_1d": indicators["return_1d"],
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
            "spy_return_1d": spy_return_1d,
            # 1-hour exit (for labels)
            "exit_price_1hr": exit_price_1hr,
            "exit_price_1hr_flag": exit_price_1hr_flag,
            "spy_return_1hr": spy_return_1hr,
            "return_1hr": return_1hr,
            "return_1hr_adjusted": return_1hr_adjusted,
            "label_5class": label_5class,
            "label_3class": label_3class,
            "is_reliable_label": is_reliable_label,
            "tweet_hash": tweet_hash,
            # 1-day labels (next trading day open)
            "price_next_open": price_next_open,
            "price_next_open_flag": price_next_open_flag,
            "return_to_next_open": return_to_next_open,
            "label_1d_5class": label_1d_5class,
            "label_1d_3class": label_1d_3class,
        }

        return result

    async def enrich_dataframe(self, tweets_df: pd.DataFrame, max_date: datetime) -> pd.DataFrame:
        """
        Enrich all tweets in a DataFrame.

        Args:
            tweets_df: DataFrame with tweets
            max_date: Maximum date for data fetching

        Returns:
            DataFrame with enriched data
        """
        results = []

        for idx, tweet_row in tweets_df.iterrows():
            result = await self.enrich_tweet(tweet_row, max_date)

            # Add original tweet data to result
            result["author"] = tweet_row["author"]
            result["category"] = tweet_row["category"]
            result["tweet_url"] = tweet_row["tweet_url"]
            result["text"] = tweet_row["text"]

            results.append(result)

            # Print concise summary
            status = "OK" if result["entry_price"] is not None else "FAIL"
            label = result["label_5class"] or "N/A"
            ret_adj = result["return_1hr_adjusted"]
            return_str = f"{ret_adj:.2%}" if ret_adj is not None else "N/A"
            self.logger.info(f"[{status}] [{idx + 1}/{len(tweets_df)}] {result['ticker']} | {result['session']} | {label} | {return_str}")

        # Create output DataFrame
        output_df = pd.DataFrame(results)

        # Reorder columns for readability
        column_order = [
            "timestamp",
            "ticker",
            "author",
            "category",
            "session",
            "market_regime",
            "sector",
            "market_cap_bucket",
            # Entry/exit prices (for labels)
            "entry_price",
            "entry_price_flag",
            "exit_price_1hr",
            "exit_price_1hr_flag",
            "return_1hr",
            "return_1hr_adjusted",
            "label_5class",
            "label_3class",
            # 1-day labels
            "price_next_open",
            "price_next_open_flag",
            "return_to_next_open",
            "label_1d_5class",
            "label_1d_3class",
            # Data quality
            "is_reliable_label",
            "tweet_hash",
            # Technical indicators (backward-looking features)
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_7d",
            "relative_volume",
            "rsi_14",
            "distance_from_ma_20",
            "above_ma_20",
            "slope_ma_20",
            "gap_open",
            "intraday_range",
            "spy_return_1d",
            "spy_return_1hr",
            # Original tweet data
            "tweet_url",
            "text",
        ]

        # Only include columns that exist
        column_order = [col for col in column_order if col in output_df.columns]
        output_df = output_df[column_order]

        return output_df
