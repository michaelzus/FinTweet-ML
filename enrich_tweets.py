"""
Tweet enrichment script.

This script enriches tweet data with financial indicators and price information
from Interactive Brokers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple

import pandas as pd
import pytz

from ib_fetcher import IBHistoricalDataFetcher
from technical_indicators import TechnicalIndicators

# US Eastern timezone
ET = pytz.timezone("US/Eastern")

# Market hours definition (all in ET)
MARKET_OPEN = 9 * 60 + 30  # 9:30 AM in minutes
MARKET_CLOSE = 16 * 60  # 4:00 PM in minutes
PREMARKET_START = 4 * 60  # 4:00 AM in minutes
AFTERHOURS_END = 20 * 60  # 8:00 PM in minutes


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
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def connect(self) -> bool:
        """Establish connection to Interactive Brokers."""
        return await self.ib_fetcher.connect()

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        await self.ib_fetcher.disconnect()

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

    async def fetch_daily_data(self, symbol: str, end_date: datetime, duration: str = "2 M") -> Optional[pd.DataFrame]:
        """
        Fetch daily historical data for a symbol up to a specific date.

        Args:
            symbol: Stock ticker symbol
            end_date: End date for historical data (typically tweet date)
            duration: Duration string (default: 2 M for 2 months)

        Returns:
            DataFrame with daily OHLCV data (timezone-aware)
        """
        # Cache key includes date to avoid stale data across different tweet dates
        cache_key = f"{symbol}_{end_date.date()}"

        if cache_key in self.daily_data_cache:
            return self.daily_data_cache[cache_key]

        try:
            # Format end_datetime for IBKR
            end_datetime_str = end_date.strftime("%Y%m%d %H:%M:%S") if end_date else ""

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

            self.daily_data_cache[cache_key] = df
            self.logger.debug(f"Fetched {len(df)} daily bars for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None

    async def fetch_intraday_data(self, symbol: str, end_date: datetime, duration: str = "2 D") -> Optional[pd.DataFrame]:
        """
        Fetch intraday historical data (1-min bars) with extended hours.

        Args:
            symbol: Stock ticker symbol
            end_date: End date for historical data
            duration: Duration string (default: 2 D)

        Returns:
            DataFrame with 1-minute OHLCV data (timezone-aware)
        """
        cache_key = f"{symbol}_{end_date.date()}"

        if cache_key in self.intraday_data_cache:
            return self.intraday_data_cache[cache_key]

        try:
            # Format end_datetime for IBKR
            end_datetime_str = end_date.strftime("%Y%m%d %H:%M:%S") if end_date else ""

            df = await self.ib_fetcher.fetch_historical_data(
                symbol=symbol,
                exchange="SMART",
                currency="USD",
                duration=duration,
                bar_size="1 min",
                use_rth=False,  # Include extended hours (pre-market, after-hours)
                end_datetime=end_datetime_str,
            )

            if df is None or df.empty:
                self.logger.warning(f"No intraday data for {symbol}")
                return None

            # Ensure timezone-aware datetime
            df = self._normalize_dataframe_timezone(df)

            self.intraday_data_cache[cache_key] = df
            self.logger.debug(f"Fetched {len(df)} intraday bars for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def get_price_at_timestamp(
        self, daily_df: pd.DataFrame, intraday_df: Optional[pd.DataFrame], timestamp: datetime, session: MarketSession
    ) -> Tuple[Optional[float], str]:
        """
        Get price at specific timestamp based on market session.

        Args:
            daily_df: Daily OHLCV data
            intraday_df: Intraday OHLCV data (1-min bars)
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
                # Find closest bar (within 5 minutes)
                time_diff = abs(intraday_df.index - timestamp)
                min_diff = time_diff.min()

                if min_diff <= pd.Timedelta(minutes=5):
                    closest_idx = time_diff.argmin()
                    price = intraday_df.iloc[closest_idx]["close"]
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
            # Find closest bar after target time (within 10 minutes)
            future_bars = intraday_df[intraday_df.index >= target_time]

            if not future_bars.empty:
                closest_bar = future_bars.iloc[0]
                time_diff = abs(closest_bar.name - target_time)

                if time_diff <= pd.Timedelta(minutes=10):
                    return closest_bar["close"], f"{hours_after}hr_after_intraday"

        # Fallback: use next day's open or current close
        if not daily_df.empty:
            return daily_df.iloc[-1]["close"], f"{hours_after}hr_after_unavailable_used_close"

        return None, "no_data_available"

    async def enrich_tweet(self, tweet_row: pd.Series) -> dict:
        """
        Enrich a single tweet with all required features.

        Args:
            tweet_row: Row from tweets DataFrame

        Returns:
            Dictionary with all enriched features
        """
        ticker = tweet_row["ticker"]
        timestamp = pd.to_datetime(tweet_row["timestamp"])

        self.logger.info(f"Processing {ticker} at {timestamp}")

        # Fetch data (up to tweet date to avoid look-ahead bias)
        daily_df = await self.fetch_daily_data(ticker, timestamp)

        if daily_df is None or daily_df.empty:
            self.logger.warning(f"No data available for {ticker}")
            return self._get_empty_features()

        # Fetch intraday data (2 days around tweet)
        intraday_df = await self.fetch_intraday_data(ticker, timestamp + timedelta(days=1))

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

        # Fetch SPY data for market-adjusted returns (same date as ticker)
        spy_df = await self.fetch_daily_data("SPY", timestamp)
        spy_return_1d = None

        if spy_df is not None and not spy_df.empty:
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
            self.logger.warning("Failed to fetch SPY data or SPY data is empty")

        # Fetch SPY intraday data for 1-hour return calculation
        spy_intraday_df = await self.fetch_intraday_data("SPY", timestamp)

        # Calculate SPY 1-hour return (same time window as stock)
        spy_return_1hr = None
        spy_price_at_tweet = None
        spy_price_1hr_after = None

        if spy_df is not None and not spy_df.empty:
            # Get SPY price at tweet time
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
    sample_size = 30
    tweets_sample = tweets_df.head(sample_size)

    logger.info(f"Processing {len(tweets_sample)} tweets as sample...")

    # Initialize enricher
    enricher = TweetEnricher()

    # Connect to IB
    connected = await enricher.connect()
    if not connected:
        logger.error("Failed to connect to IB")
        return

    try:
        # Process each tweet
        results = []

        for idx, tweet_row in tweets_sample.iterrows():
            result = await enricher.enrich_tweet(tweet_row)

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
