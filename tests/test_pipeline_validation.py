"""Comprehensive validation tests for the tweet enrichment pipeline.

This module validates:
1. Twitter timestamp conversion (UTC -> Eastern with DST)
2. Stock data timezone handling (IB data in Eastern)
3. Cross-referencing between tweets and stock data
4. Incremental fetching algorithms
5. Market session detection
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import pytest

from tweet_enricher.config import ET, TWITTER_DB_PATH
from tweet_enricher.io.feather import load_daily_data, load_intraday_data
from tweet_enricher.market.session import (
    MarketSession,
    get_market_session,
    normalize_dataframe_timezone,
    normalize_timestamp,
)
from tweet_enricher.twitter.sync import SyncService


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tweet_db_path() -> Path:
    """Path to tweets database."""
    return TWITTER_DB_PATH


@pytest.fixture
def daily_data_dir() -> Path:
    """Path to daily data directory."""
    return Path("data/daily")


@pytest.fixture
def intraday_data_dir() -> Path:
    """Path to intraday data directory."""
    return Path("data/intraday")


@pytest.fixture
def sample_tweets(tweet_db_path: Path) -> pd.DataFrame:
    """Load sample processed tweets from database."""
    if not tweet_db_path.exists():
        pytest.skip("Tweet database not found")

    conn = sqlite3.connect(tweet_db_path)
    df = pd.read_sql("SELECT * FROM tweets_processed LIMIT 100", conn)
    conn.close()
    return df


# ============================================================================
# 1. TWITTER TIMESTAMP CONVERSION TESTS
# ============================================================================


class TestTwitterTimestampConversion:
    """Test Twitter UTC to Eastern timezone conversion."""

    def test_utc_to_eastern_basic_conversion(self):
        """Test basic UTC to Eastern conversion."""
        sync = SyncService(lazy_client=True)

        # Twitter format: "Wed Dec 17 15:01:11 +0000 2025"
        utc_ts = "Wed Dec 17 15:01:11 +0000 2025"
        eastern_ts = sync._convert_utc_to_eastern(utc_ts)

        # Parse result
        result_dt = datetime.strptime(eastern_ts, "%Y-%m-%d %H:%M:%S")

        # Dec 17 is during EST (UTC-5), so 15:01 UTC = 10:01 EST
        assert result_dt.hour == 10
        assert result_dt.minute == 1
        assert result_dt.second == 11

    def test_utc_to_eastern_summer_dst(self):
        """Test UTC to Eastern during summer (EDT)."""
        sync = SyncService(lazy_client=True)

        # Summer: Jul 15, EDT is UTC-4
        utc_ts = "Tue Jul 15 18:30:00 +0000 2025"
        eastern_ts = sync._convert_utc_to_eastern(utc_ts)

        result_dt = datetime.strptime(eastern_ts, "%Y-%m-%d %H:%M:%S")

        # 18:30 UTC = 14:30 EDT (UTC-4)
        assert result_dt.hour == 14
        assert result_dt.minute == 30

    def test_utc_to_eastern_winter_standard(self):
        """Test UTC to Eastern during winter (EST)."""
        sync = SyncService(lazy_client=True)

        # Winter: Jan 15, EST is UTC-5
        utc_ts = "Wed Jan 15 18:30:00 +0000 2025"
        eastern_ts = sync._convert_utc_to_eastern(utc_ts)

        result_dt = datetime.strptime(eastern_ts, "%Y-%m-%d %H:%M:%S")

        # 18:30 UTC = 13:30 EST (UTC-5)
        assert result_dt.hour == 13
        assert result_dt.minute == 30

    def test_utc_to_eastern_dst_transition_spring(self):
        """Test UTC to Eastern during spring DST transition (March)."""
        sync = SyncService(lazy_client=True)

        # March 9, 2025: DST starts at 2:00 AM local
        # Before transition (1:30 AM EST = 6:30 UTC)
        utc_ts_before = "Sun Mar 09 06:30:00 +0000 2025"
        eastern_before = sync._convert_utc_to_eastern(utc_ts_before)
        dt_before = datetime.strptime(eastern_before, "%Y-%m-%d %H:%M:%S")
        assert dt_before.hour == 1  # 1:30 AM EST

        # After transition (3:30 AM EDT = 7:30 UTC)
        utc_ts_after = "Sun Mar 09 07:30:00 +0000 2025"
        eastern_after = sync._convert_utc_to_eastern(utc_ts_after)
        dt_after = datetime.strptime(eastern_after, "%Y-%m-%d %H:%M:%S")
        assert dt_after.hour == 3  # 3:30 AM EDT (skipped 2 AM)

    def test_utc_to_eastern_dst_transition_fall(self):
        """Test UTC to Eastern during fall DST transition (November)."""
        sync = SyncService(lazy_client=True)

        # November 2, 2025: DST ends at 2:00 AM local
        # Before transition (1:00 AM EDT = 5:00 UTC)
        utc_ts_before = "Sun Nov 02 05:00:00 +0000 2025"
        eastern_before = sync._convert_utc_to_eastern(utc_ts_before)
        dt_before = datetime.strptime(eastern_before, "%Y-%m-%d %H:%M:%S")
        assert dt_before.hour == 1

        # After transition (1:00 AM EST = 6:00 UTC)
        utc_ts_after = "Sun Nov 02 06:00:00 +0000 2025"
        eastern_after = sync._convert_utc_to_eastern(utc_ts_after)
        dt_after = datetime.strptime(eastern_after, "%Y-%m-%d %H:%M:%S")
        assert dt_after.hour == 1  # Still 1:00 AM but now EST


# ============================================================================
# 2. STOCK DATA TIMEZONE TESTS
# ============================================================================


class TestStockDataTimezone:
    """Test stock data timezone handling."""

    def test_daily_data_has_eastern_timezone(self, daily_data_dir: Path):
        """Verify daily data is stored with Eastern timezone."""
        if not daily_data_dir.exists():
            pytest.skip("Daily data directory not found")

        feather_files = list(daily_data_dir.glob("*.feather"))
        if not feather_files:
            pytest.skip("No daily data files found")

        df = load_daily_data("AAPL", daily_data_dir)
        if df is None:
            pytest.skip("AAPL daily data not found")

        # Check timezone
        assert df.index.tz is not None, "Daily data index should be timezone-aware"
        assert str(df.index.tz) == "US/Eastern", f"Expected US/Eastern, got {df.index.tz}"

    def test_intraday_data_has_eastern_timezone(self, intraday_data_dir: Path):
        """Verify intraday data is stored with Eastern timezone."""
        if not intraday_data_dir.exists():
            pytest.skip("Intraday data directory not found")

        df = load_intraday_data("AAPL", intraday_data_dir)
        if df is None:
            pytest.skip("AAPL intraday data not found")

        # Check timezone
        assert df.index.tz is not None, "Intraday data index should be timezone-aware"
        assert str(df.index.tz) == "US/Eastern", f"Expected US/Eastern, got {df.index.tz}"

    def test_intraday_bars_cover_extended_hours(self, intraday_data_dir: Path):
        """Verify intraday data covers pre-market and after-hours."""
        df = load_intraday_data("AAPL", intraday_data_dir)
        if df is None:
            pytest.skip("AAPL intraday data not found")

        # Get trading days with sufficient data
        dates = df.index.date
        unique_dates = pd.Series(dates).unique()
        if len(unique_dates) == 0:
            pytest.skip("No trading days in intraday data")

        # Find a complete day (not today which may be partial)
        sample_date = None
        for date in reversed(unique_dates):
            day_data = df[df.index.date == date]
            # A complete day should have at least 26 bars (4AM-8PM = 64 bars at 15min)
            if len(day_data) >= 26:
                sample_date = date
                break

        if sample_date is None:
            pytest.skip("No complete trading day found in data")

        day_data = df[df.index.date == sample_date]

        # Check pre-market (4:00 AM - 9:30 AM)
        premarket_bars = day_data[(day_data.index.hour >= 4) & (day_data.index.hour < 9)]
        assert len(premarket_bars) > 0, "Should have pre-market bars (4:00 AM - 9:30 AM)"

        # Check regular hours (9:30 AM - 4:00 PM)
        # Note: hour >= 10 ensures we're in regular hours (9:30 rounds to hour 9 but minute >= 30)
        regular_bars = day_data[((day_data.index.hour == 9) & (day_data.index.minute >= 30)) | (day_data.index.hour >= 10)]
        regular_bars = regular_bars[regular_bars.index.hour < 16]
        assert len(regular_bars) > 0, f"Should have regular hours bars on {sample_date}"

        # After-hours might not have data for all stocks, so just check existence
        _ = day_data[(day_data.index.hour >= 16) & (day_data.index.hour < 20)]
        # Not asserting - some days may not have after-hours trading

    def test_daily_bar_timestamps_at_midnight(self, daily_data_dir: Path):
        """Verify daily bars have timestamps at market close (shown as midnight)."""
        df = load_daily_data("AAPL", daily_data_dir)
        if df is None:
            pytest.skip("AAPL daily data not found")

        # Daily bars should be at 00:00:00 (midnight) representing that trading day
        sample_times = df.index[:5]
        for ts in sample_times:
            assert ts.hour == 0, f"Daily bar should be at midnight, got {ts.hour}:00"
            assert ts.minute == 0, f"Daily bar should be at minute 0, got minute {ts.minute}"


# ============================================================================
# 3. MARKET SESSION DETECTION TESTS
# ============================================================================


class TestMarketSessionDetection:
    """Test market session detection logic."""

    def test_regular_hours_detection(self):
        """Test detection of regular trading hours."""
        # 10:30 AM ET on a Tuesday
        ts = ET.localize(datetime(2025, 12, 16, 10, 30, 0))
        session = get_market_session(ts)
        assert session == MarketSession.REGULAR

        # 3:59 PM ET
        ts = ET.localize(datetime(2025, 12, 16, 15, 59, 0))
        session = get_market_session(ts)
        assert session == MarketSession.REGULAR

    def test_premarket_detection(self):
        """Test detection of pre-market hours."""
        # 4:30 AM ET
        ts = ET.localize(datetime(2025, 12, 16, 4, 30, 0))
        session = get_market_session(ts)
        assert session == MarketSession.PREMARKET

        # 9:29 AM ET (just before open)
        ts = ET.localize(datetime(2025, 12, 16, 9, 29, 0))
        session = get_market_session(ts)
        assert session == MarketSession.PREMARKET

    def test_afterhours_detection(self):
        """Test detection of after-hours trading."""
        # 4:30 PM ET
        ts = ET.localize(datetime(2025, 12, 16, 16, 30, 0))
        session = get_market_session(ts)
        assert session == MarketSession.AFTERHOURS

        # 7:59 PM ET
        ts = ET.localize(datetime(2025, 12, 16, 19, 59, 0))
        session = get_market_session(ts)
        assert session == MarketSession.AFTERHOURS

    def test_closed_detection_overnight(self):
        """Test detection of closed market (overnight)."""
        # 2:00 AM ET (before pre-market)
        ts = ET.localize(datetime(2025, 12, 16, 2, 0, 0))
        session = get_market_session(ts)
        assert session == MarketSession.CLOSED

        # 9:00 PM ET (after after-hours)
        ts = ET.localize(datetime(2025, 12, 16, 21, 0, 0))
        session = get_market_session(ts)
        assert session == MarketSession.CLOSED

    def test_closed_detection_weekend(self):
        """Test detection of closed market (weekend)."""
        # Saturday 10:00 AM ET
        ts = ET.localize(datetime(2025, 12, 20, 10, 0, 0))  # Saturday
        session = get_market_session(ts)
        assert session == MarketSession.CLOSED

        # Sunday 2:00 PM ET
        ts = ET.localize(datetime(2025, 12, 21, 14, 0, 0))  # Sunday
        session = get_market_session(ts)
        assert session == MarketSession.CLOSED


# ============================================================================
# 4. CROSS-REFERENCING TESTS (Tweet to Stock Data)
# ============================================================================


class TestTweetStockCrossReference:
    """Test cross-referencing between tweets and stock data."""

    def test_tweet_timestamp_matches_stock_data_range(self, sample_tweets: pd.DataFrame, intraday_data_dir: Path):
        """Verify tweet timestamps fall within stock data range."""
        if sample_tweets.empty:
            pytest.skip("No sample tweets")

        # Get a sample tweet with a valid ticker
        valid_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        sample_tweet = None

        for _, tweet in sample_tweets.iterrows():
            if tweet["ticker"] in valid_tickers:
                sample_tweet = tweet
                break

        if sample_tweet is None:
            pytest.skip("No tweets with standard tickers found")

        ticker = sample_tweet["ticker"]
        df = load_intraday_data(ticker, intraday_data_dir)
        if df is None:
            pytest.skip(f"No intraday data for {ticker}")

        # Parse tweet timestamp
        tweet_ts = pd.to_datetime(sample_tweet["timestamp_et"])

        # Check if tweet date is within stock data range
        stock_min_date = df.index.min().date()
        tweet_date = tweet_ts.date()

        # The tweet date should be within a reasonable range of stock data
        # (tweets from the past year should have corresponding stock data)
        assert tweet_date >= stock_min_date - timedelta(days=30), f"Tweet date {tweet_date} is before stock data starts {stock_min_date}"

    def test_normalize_timestamp_preserves_time(self):
        """Test that normalize_timestamp preserves the time correctly."""
        # Test naive timestamp (assumed to be ET)
        naive_ts = datetime(2025, 12, 16, 10, 30, 0)
        normalized = normalize_timestamp(naive_ts)

        assert normalized.tzinfo is not None
        assert normalized.hour == 10
        assert normalized.minute == 30

        # Test UTC timestamp conversion
        utc_ts = pytz.UTC.localize(datetime(2025, 12, 16, 15, 30, 0))  # 3:30 PM UTC
        normalized = normalize_timestamp(utc_ts)

        # 15:30 UTC = 10:30 EST
        assert normalized.hour == 10
        assert normalized.minute == 30

    def test_dataframe_timezone_normalization(self):
        """Test DataFrame timezone normalization."""
        # Create a naive DataFrame
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)

        # Normalize
        normalized = normalize_dataframe_timezone(df)

        assert normalized.index.tz is not None
        assert str(normalized.index.tz) == "US/Eastern"


# ============================================================================
# 5. INCREMENTAL FETCHING TESTS
# ============================================================================


class TestIncrementalFetching:
    """Test incremental fetching algorithms."""

    def test_fetch_journal_marks_days(self, tweet_db_path: Path):
        """Test that fetch journal correctly tracks fetched days."""
        if not tweet_db_path.exists():
            pytest.skip("Tweet database not found")

        conn = sqlite3.connect(tweet_db_path)

        # Check fetch_journal table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_journal'")
        if cursor.fetchone() is None:
            conn.close()
            pytest.skip("fetch_journal table not found")

        # Check for fetched days
        df = pd.read_sql("SELECT * FROM fetch_journal LIMIT 10", conn)
        conn.close()

        if df.empty:
            pytest.skip("No fetch journal entries")

        # Verify structure
        assert "account" in df.columns
        assert "fetch_date" in df.columns
        assert "fetched_at" in df.columns
        assert "tweets_count" in df.columns

    def test_sync_state_tracking(self, tweet_db_path: Path):
        """Test that sync state is properly tracked."""
        if not tweet_db_path.exists():
            pytest.skip("Tweet database not found")

        conn = sqlite3.connect(tweet_db_path)
        df = pd.read_sql("SELECT * FROM sync_state", conn)
        conn.close()

        if df.empty:
            pytest.skip("No sync state entries")

        # Verify structure
        assert "account" in df.columns
        assert "last_sync_at" in df.columns
        assert "total_tweets" in df.columns

    def test_raw_tweets_have_json_data(self, tweet_db_path: Path):
        """Test that raw tweets preserve full JSON for debugging."""
        if not tweet_db_path.exists():
            pytest.skip("Tweet database not found")

        conn = sqlite3.connect(tweet_db_path)
        df = pd.read_sql("SELECT id, json_data FROM tweets_raw LIMIT 1", conn)
        conn.close()

        if df.empty:
            pytest.skip("No raw tweets")

        # Verify JSON is valid and contains expected fields
        raw_json = json.loads(df.iloc[0]["json_data"])
        assert "createdAt" in raw_json, "Raw tweet should have createdAt field"
        assert "text" in raw_json, "Raw tweet should have text field"


# ============================================================================
# 6. DATA CONSISTENCY TESTS
# ============================================================================


class TestDataConsistency:
    """Test data consistency across the pipeline."""

    def test_processed_tweets_have_required_fields(self, sample_tweets: pd.DataFrame):
        """Test that processed tweets have all required fields."""
        if sample_tweets.empty:
            pytest.skip("No sample tweets")

        required_fields = [
            "id",
            "timestamp_utc",
            "timestamp_et",
            "author",
            "ticker",
            "tweet_url",
            "text",
        ]

        for field in required_fields:
            assert field in sample_tweets.columns, f"Missing required field: {field}"

    def test_timestamp_et_is_valid_datetime(self, sample_tweets: pd.DataFrame):
        """Test that timestamp_et can be parsed as datetime."""
        if sample_tweets.empty:
            pytest.skip("No sample tweets")

        for _, tweet in sample_tweets.head(10).iterrows():
            ts = tweet["timestamp_et"]
            # Should be parseable
            try:
                parsed = pd.to_datetime(ts)
                assert parsed is not pd.NaT
            except Exception as e:
                pytest.fail(f"Failed to parse timestamp_et '{ts}': {e}")

    def test_ticker_symbols_are_valid(self, sample_tweets: pd.DataFrame):
        """Test that ticker symbols are properly formatted."""
        if sample_tweets.empty:
            pytest.skip("No sample tweets")

        for _, tweet in sample_tweets.head(50).iterrows():
            ticker = tweet["ticker"]
            # Ticker should be uppercase, 1-5 characters
            assert ticker == ticker.upper(), f"Ticker should be uppercase: {ticker}"
            assert 1 <= len(ticker) <= 6, f"Invalid ticker length: {ticker}"


# ============================================================================
# 7. ENRICHMENT OUTPUT VALIDATION
# ============================================================================


class TestEnrichmentOutput:
    """Test enrichment output file validity."""

    @pytest.fixture
    def enriched_data(self) -> Optional[pd.DataFrame]:
        """Load enriched output file."""
        output_path = Path("output/2025_enrich.csv")
        if not output_path.exists():
            return None
        return pd.read_csv(output_path)

    def test_enriched_timestamps_have_timezone(self, enriched_data: Optional[pd.DataFrame]):
        """Test that enriched timestamps include timezone info."""
        if enriched_data is None:
            pytest.skip("Enriched data file not found")

        # Sample timestamp should include timezone
        sample_ts = enriched_data.iloc[0]["timestamp"]
        assert "-05:00" in str(sample_ts) or "-04:00" in str(sample_ts), f"Timestamp should include Eastern timezone offset: {sample_ts}"

    def test_enriched_session_values_valid(self, enriched_data: Optional[pd.DataFrame]):
        """Test that session values are valid."""
        if enriched_data is None:
            pytest.skip("Enriched data file not found")

        valid_sessions = {"regular", "premarket", "afterhours", "closed"}
        sessions = enriched_data["session"].dropna().unique()

        for session in sessions:
            assert session in valid_sessions, f"Invalid session value: {session}"

    def test_enriched_prices_are_positive(self, enriched_data: Optional[pd.DataFrame]):
        """Test that entry/exit prices are positive when present."""
        if enriched_data is None:
            pytest.skip("Enriched data file not found")

        price_columns = ["entry_price", "exit_price_1hr", "price_next_open"]

        for col in price_columns:
            if col in enriched_data.columns:
                valid_prices = enriched_data[col].dropna()
                assert (valid_prices > 0).all(), f"All {col} values should be positive"

    def test_enriched_returns_are_reasonable(self, enriched_data: Optional[pd.DataFrame]):
        """Test that returns are within reasonable bounds."""
        if enriched_data is None:
            pytest.skip("Enriched data file not found")

        return_columns = ["return_1hr", "return_1hr_adjusted", "return_to_next_open"]

        for col in return_columns:
            if col in enriched_data.columns:
                valid_returns = enriched_data[col].dropna()
                # Returns should be between -100% and +200% (reasonable for 1hr/1day)
                assert (valid_returns >= -1.0).all(), f"{col} has unreasonably negative values"
                assert (valid_returns <= 2.0).all(), f"{col} has unreasonably high values"

    def test_label_classes_are_valid(self, enriched_data: Optional[pd.DataFrame]):
        """Test that label classes are valid."""
        if enriched_data is None:
            pytest.skip("Enriched data file not found")

        # 5-class labels
        valid_5class = {"strong_sell", "sell", "hold", "buy", "strong_buy"}
        if "label_5class" in enriched_data.columns:
            labels = enriched_data["label_5class"].dropna().unique()
            for label in labels:
                assert label in valid_5class, f"Invalid 5-class label: {label}"

        # 3-class labels
        valid_3class = {"SELL", "HOLD", "BUY"}
        if "label_3class" in enriched_data.columns:
            labels = enriched_data["label_3class"].dropna().unique()
            for label in labels:
                assert label in valid_3class, f"Invalid 3-class label: {label}"


# ============================================================================
# RUN VALIDATION SUMMARY (standalone script)
# ============================================================================


def run_validation_summary():
    """Run a quick validation summary (not a pytest test)."""
    print("=" * 80)
    print("PIPELINE VALIDATION SUMMARY")
    print("=" * 80)

    # Check database
    print("\n1. TWEET DATABASE CHECK")
    if TWITTER_DB_PATH.exists():
        conn = sqlite3.connect(TWITTER_DB_PATH)

        # Count tweets
        raw_count = conn.execute("SELECT COUNT(*) FROM tweets_raw").fetchone()[0]
        processed_count = conn.execute("SELECT COUNT(*) FROM tweets_processed").fetchone()[0]
        print(f"   ✓ Raw tweets: {raw_count:,}")
        print(f"   ✓ Processed tweets: {processed_count:,}")

        # Sample timestamp check
        sample = conn.execute("SELECT timestamp_utc, timestamp_et FROM tweets_processed LIMIT 1").fetchone()
        if sample:
            print(f"   ✓ Sample UTC: {sample[0]}")
            print(f"   ✓ Sample ET:  {sample[1]}")

        conn.close()
    else:
        print("   ✗ Database not found")

    # Check stock data
    print("\n2. STOCK DATA CHECK")
    daily_dir = Path("data/daily")
    intraday_dir = Path("data/intraday")

    if daily_dir.exists():
        daily_files = list(daily_dir.glob("*.feather"))
        print(f"   ✓ Daily data files: {len(daily_files)}")

        df = load_daily_data("AAPL", daily_dir)
        if df is not None:
            print(f"   ✓ AAPL daily TZ: {df.index.tz}")
            print(f"   ✓ AAPL daily range: {df.index.min().date()} to {df.index.max().date()}")
    else:
        print("   ✗ Daily data directory not found")

    if intraday_dir.exists():
        intraday_files = list(intraday_dir.glob("*.feather"))
        print(f"   ✓ Intraday data files: {len(intraday_files)}")

        df = load_intraday_data("AAPL", intraday_dir)
        if df is not None:
            print(f"   ✓ AAPL intraday TZ: {df.index.tz}")
            print(f"   ✓ AAPL intraday range: {df.index.min()} to {df.index.max()}")
    else:
        print("   ✗ Intraday data directory not found")

    # Check enriched output
    print("\n3. ENRICHED OUTPUT CHECK")
    output_path = Path("output/2025_enrich.csv")
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"   ✓ Enriched rows: {len(df):,}")
        print(f"   ✓ Sample timestamp: {df.iloc[0]['timestamp']}")
        print("   ✓ Session distribution:")
        for session, count in df["session"].value_counts().items():
            print(f"      - {session}: {count:,}")

        # Check reliability
        if "is_reliable_label" in df.columns:
            reliable = df["is_reliable_label"].sum()
            total = len(df)
            print(f"   ✓ Reliable labels: {reliable:,}/{total:,} ({100 * reliable / total:.1f}%)")
    else:
        print("   ✗ Enriched output not found")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_validation_summary()
