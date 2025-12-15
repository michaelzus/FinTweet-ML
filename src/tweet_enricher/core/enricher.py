"""Tweet enrichment with financial indicators and price information."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache import DataCache
from tweet_enricher.data.ib_fetcher import IBHistoricalDataFetcher
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
    ):
        """
        Initialize the TweetEnricher.

        Args:
            ib_fetcher: IBHistoricalDataFetcher instance for fetching data
            cache: DataCache instance for caching data
            indicators: TechnicalIndicators instance for calculating indicators
        """
        self.ib_fetcher = ib_fetcher
        self.cache = cache
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """Establish connection to Interactive Brokers."""
        return await self.ib_fetcher.connect()

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        await self.ib_fetcher.disconnect()

    def get_price_at_timestamp(
        self,
        daily_df: pd.DataFrame,
        intraday_df: Optional[pd.DataFrame],
        timestamp: datetime,
        session: MarketSession,
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
        timestamp = normalize_timestamp(timestamp)

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

        # CRITICAL: Slice to only include data UP TO tweet date (no look-ahead!)
        tweet_date = timestamp.date()
        daily_df = daily_df_full[daily_df_full.index.date <= tweet_date].copy()

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
        indicators = self.indicators.calculate_all_indicators(daily_df, current_idx)

        # Get SPY data from cache for market adjustment
        spy_df_full = self.cache.get_daily("SPY")
        spy_return_1d = None

        if spy_df_full is not None and not spy_df_full.empty:
            # CRITICAL: Slice SPY data to only include data UP TO tweet date
            spy_df = spy_df_full[spy_df_full.index.date <= tweet_date].copy()

            if not spy_df.empty:
                spy_dates = spy_df.index.date
                spy_valid_indices = [i for i, d in enumerate(spy_dates) if d <= tweet_date]

                if spy_valid_indices:
                    spy_idx = spy_valid_indices[-1]
                    self.logger.debug(f"SPY: Found {len(spy_df)} bars, using index {spy_idx} (date: {spy_dates[spy_idx]})")
                    spy_return_1d = self.indicators.calculate_return(spy_df, spy_idx, periods=1)
                    self.logger.debug(f"SPY return_1d: {spy_return_1d}")
                else:
                    self.logger.warning(f"No SPY data found for date <= {tweet_date}")
            else:
                self.logger.warning(f"No SPY data before/on {tweet_date}")
        else:
            self.logger.warning("Failed to fetch SPY data or SPY data is empty")

        # Get SPY intraday data for 1hr return calculation
        spy_intraday_df_full = self.cache.get_intraday("SPY")

        # Calculate SPY 1-hour return (same time window as stock)
        spy_return_1hr = None
        spy_price_at_tweet = None
        spy_price_1hr_after = None

        # CRITICAL: Slice SPY intraday to only include data UP TO 1 hour after tweet
        spy_intraday_df = None
        if spy_intraday_df_full is not None and not spy_intraday_df_full.empty:
            spy_intraday_df = spy_intraday_df_full[spy_intraday_df_full.index <= max_intraday_time].copy()

        if spy_df_full is not None and not spy_df_full.empty:
            # Use SLICED SPY data (already filtered above)
            spy_session = get_market_session(timestamp)
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
            "session": session.value if session else None,
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
            status = "OK" if result["price_at_tweet"] is not None else "FAIL"
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

        return output_df

