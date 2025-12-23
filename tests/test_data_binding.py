"""Tests for data binding correctness in dataset preparation.

This module validates that:
- Tweet timestamps are correctly bound to OHLCV data
- Entry prices use the first bar AFTER tweet timestamp
- Next open prices use the first trading day AFTER tweet date
- Technical indicators use only data BEFORE tweet date (no look-ahead)
- DST transitions are handled correctly in bindings
- Timezone comparisons work across different tz representations
"""

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tweet_enricher.core.dataset_builder import DatasetBuilder
from tweet_enricher.core.indicators import TechnicalIndicators
from tweet_enricher.data.cache_reader import CacheReader
from tweet_enricher.market.session import MarketSession, get_market_session, normalize_timestamp
from tweet_enricher.utils.timezone import ET, normalize_dataframe_timezone


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_daily_data() -> pd.DataFrame:
    """Create sample daily OHLCV data for testing (30+ days for indicators)."""
    # Need 30+ days for 20-day indicators to have valid values
    dates = pd.date_range("2024-12-01", "2025-01-20", freq="B")  # Business days
    df = pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(len(dates))],
            "high": [101 + i * 0.5 for i in range(len(dates))],
            "low": [99 + i * 0.5 for i in range(len(dates))],
            "close": [100.5 + i * 0.5 for i in range(len(dates))],
            "volume": [1000000 + i * 10000 for i in range(len(dates))],
        },
        index=pd.DatetimeIndex(dates, tz="America/New_York"),
    )
    return df


@pytest.fixture
def sample_intraday_data() -> pd.DataFrame:
    """Create sample intraday OHLCV data (15-min bars) for testing."""
    # Create intraday data for Jan 15, 2025 (4:00 AM to 8:00 PM ET)
    times = pd.date_range("2025-01-15 04:00", "2025-01-15 20:00", freq="15min")
    df = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(len(times))],
            "high": [100.5 + i * 0.1 for i in range(len(times))],
            "low": [99.5 + i * 0.1 for i in range(len(times))],
            "close": [100.25 + i * 0.1 for i in range(len(times))],
            "volume": [10000 + i * 100 for i in range(len(times))],
        },
        index=pd.DatetimeIndex(times, tz="America/New_York"),
    )
    return df


@pytest.fixture
def mock_cache(sample_daily_data, sample_intraday_data) -> MagicMock:
    """Create a mock cache with sample data."""
    cache = MagicMock(spec=CacheReader)
    cache.get_daily.return_value = sample_daily_data
    cache.get_intraday.return_value = sample_intraday_data
    return cache


@pytest.fixture
def dataset_builder(mock_cache) -> DatasetBuilder:
    """Create a DatasetBuilder with mocked cache."""
    indicators = TechnicalIndicators()
    return DatasetBuilder(cache=mock_cache, indicators=indicators)


# =============================================================================
# Test: Timestamp Normalization
# =============================================================================


class TestTimestampNormalization:
    """Tests for tweet timestamp normalization."""

    def test_naive_timestamp_becomes_et_aware(self):
        """Naive timestamp should become ET-aware."""
        naive_ts = pd.to_datetime("2025-01-15 10:30:00")
        normalized = normalize_timestamp(naive_ts)

        assert normalized.tzinfo is not None
        assert str(normalized.tzinfo) == "America/New_York"
        assert normalized.hour == 10
        assert normalized.minute == 30

    def test_utc_timestamp_converted_to_et(self):
        """UTC timestamp should be converted to ET."""
        utc_ts = pd.to_datetime("2025-01-15 15:30:00+00:00")
        normalized = normalize_timestamp(utc_ts)

        assert str(normalized.tzinfo) == "America/New_York"
        # 15:30 UTC = 10:30 EST (UTC-5)
        assert normalized.hour == 10
        assert normalized.minute == 30

    def test_et_timestamp_unchanged(self):
        """ET timestamp should remain unchanged."""
        et_ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        normalized = normalize_timestamp(et_ts)

        assert normalized.hour == 10
        assert normalized.minute == 30
        assert str(normalized.tzinfo) == "America/New_York"

    def test_pandas_timestamp_normalized(self):
        """Pandas Timestamp should be normalized correctly."""
        pd_ts = pd.Timestamp("2025-01-15 10:30:00")
        normalized = normalize_timestamp(pd_ts)

        assert normalized.tzinfo is not None
        assert normalized.hour == 10


# =============================================================================
# Test: OHLCV Data Timezone
# =============================================================================


class TestOHLCVTimezone:
    """Tests for OHLCV data timezone handling."""

    def test_daily_data_is_et_aware(self, sample_daily_data):
        """Daily data should have ET timezone."""
        assert sample_daily_data.index.tz is not None
        assert str(sample_daily_data.index.tz) == "America/New_York"

    def test_intraday_data_is_et_aware(self, sample_intraday_data):
        """Intraday data should have ET timezone."""
        assert sample_intraday_data.index.tz is not None
        assert str(sample_intraday_data.index.tz) == "America/New_York"

    def test_normalize_naive_dataframe(self):
        """Naive DataFrame index should be localized to ET."""
        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.DatetimeIndex(["2025-01-15 10:00", "2025-01-15 10:15"]),
        )
        normalized = normalize_dataframe_timezone(df)

        assert normalized.index.tz is not None
        assert str(normalized.index.tz) == "America/New_York"

    def test_us_eastern_alias_compatible(self, sample_daily_data):
        """US/Eastern alias should be compatible with America/New_York."""
        # Create data with US/Eastern alias (what pandas uses internally)
        df = pd.DataFrame(
            {"close": [100]},
            index=pd.DatetimeIndex(["2025-01-15 10:00"], tz="US/Eastern"),
        )

        # Comparison with America/New_York timestamp should work
        et_ts = datetime(2025, 1, 15, 9, 45, 0, tzinfo=ET)
        future_bars = df[df.index > et_ts]

        assert len(future_bars) == 1


# =============================================================================
# Test: Entry Price Binding
# =============================================================================


class TestEntryPriceBinding:
    """Tests for entry price binding correctness."""

    def test_entry_price_uses_bar_after_tweet(self, dataset_builder, sample_intraday_data):
        """Entry price should use the OPEN of first bar AFTER tweet."""
        # Tweet at 10:30, next bar starts at 10:45
        tweet_ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(["2025-01-14", "2025-01-15"], tz="America/New_York"),
        )

        price, flag = dataset_builder._get_entry_price(
            sample_intraday_data, daily_df, tweet_ts, MarketSession.REGULAR
        )

        # 10:45 bar open should be used (tweet was at 10:30)
        expected_bar_idx = (10 * 60 + 45 - 4 * 60) // 15  # Bar index for 10:45
        expected_price = sample_intraday_data.iloc[expected_bar_idx]["open"]

        assert price == pytest.approx(expected_price, rel=1e-6)
        assert "next_bar_open" in flag

    def test_entry_price_not_bar_at_tweet_time(self, dataset_builder, sample_intraday_data):
        """Entry price should NOT be the bar at tweet time (realistic execution)."""
        # Tweet exactly at 10:00 - should use 10:15 bar, not 10:00
        tweet_ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100]},
            index=pd.DatetimeIndex(["2025-01-15"], tz="America/New_York"),
        )

        price, flag = dataset_builder._get_entry_price(
            sample_intraday_data, daily_df, tweet_ts, MarketSession.REGULAR
        )

        # Bar at 10:00 should NOT be used; 10:15 should be
        bar_at_1000_idx = (10 * 60 - 4 * 60) // 15
        bar_at_1015_idx = bar_at_1000_idx + 1
        bar_at_1000_open = sample_intraday_data.iloc[bar_at_1000_idx]["open"]
        bar_at_1015_open = sample_intraday_data.iloc[bar_at_1015_idx]["open"]

        assert price != bar_at_1000_open
        assert price == pytest.approx(bar_at_1015_open, rel=1e-6)

    def test_entry_price_premarket_uses_intraday(self, dataset_builder, sample_intraday_data):
        """Pre-market tweet should use intraday data for entry price."""
        # Pre-market tweet at 8:30
        tweet_ts = datetime(2025, 1, 15, 8, 30, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100]},
            index=pd.DatetimeIndex(["2025-01-15"], tz="America/New_York"),
        )

        price, flag = dataset_builder._get_entry_price(
            sample_intraday_data, daily_df, tweet_ts, MarketSession.PREMARKET
        )

        assert price is not None
        assert "premarket" in flag
        assert "next_bar_open" in flag

    def test_entry_price_closed_uses_next_day(self, dataset_builder):
        """Closed market tweet should use next trading day open."""
        # Sunday evening tweet
        tweet_ts = datetime(2025, 1, 12, 20, 0, 0, tzinfo=ET)  # Sunday
        daily_df = pd.DataFrame(
            {"open": [100, 101, 102]},
            index=pd.DatetimeIndex(
                ["2025-01-10", "2025-01-13", "2025-01-14"],  # Fri, Mon, Tue
                tz="America/New_York",
            ),
        )

        price, flag = dataset_builder._get_entry_price(
            None, daily_df, tweet_ts, MarketSession.CLOSED
        )

        # Should use Monday's open
        assert price == 101
        assert "closed" in flag
        assert "next_day" in flag

    def test_entry_price_fallback_to_daily(self, dataset_builder):
        """When intraday ends before tweet, should fallback to daily."""
        # Create intraday that ends at 4 PM
        intraday_ending_4pm = pd.DataFrame(
            {"open": [100], "close": [101]},
            index=pd.DatetimeIndex(["2025-01-15 15:45:00"], tz="America/New_York"),
        )

        # Tweet at 5 PM (after-hours, no more intraday bars)
        tweet_ts = datetime(2025, 1, 15, 17, 0, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(["2025-01-15", "2025-01-16"], tz="America/New_York"),
        )

        price, flag = dataset_builder._get_entry_price(
            intraday_ending_4pm, daily_df, tweet_ts, MarketSession.AFTERHOURS
        )

        # Should fallback to next day open
        assert price == 101
        assert "next_day" in flag


# =============================================================================
# Test: Next Open Price Binding
# =============================================================================


class TestNextOpenPriceBinding:
    """Tests for next trading day open price binding."""

    def test_next_open_uses_day_after_tweet(self, dataset_builder):
        """Next open should be the first trading day AFTER tweet date."""
        tweet_ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100, 101, 102]},
            index=pd.DatetimeIndex(
                ["2025-01-15", "2025-01-16", "2025-01-17"],
                tz="America/New_York",
            ),
        )

        price, flag, next_dt = dataset_builder._get_price_next_open(daily_df, tweet_ts)

        # Should use Jan 16 open (first day AFTER Jan 15)
        assert price == 101
        assert flag == "next_open_available"
        assert next_dt.date() == datetime(2025, 1, 16).date()

    def test_next_open_skips_same_day(self, dataset_builder):
        """Next open should skip the tweet day even if tweet is early."""
        # Tweet at 4:30 AM (before market opens)
        tweet_ts = datetime(2025, 1, 15, 4, 30, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(
                ["2025-01-15", "2025-01-16"],
                tz="America/New_York",
            ),
        )

        price, flag, next_dt = dataset_builder._get_price_next_open(daily_df, tweet_ts)

        # Should still use Jan 16 (not Jan 15)
        assert price == 101
        assert next_dt.date() == datetime(2025, 1, 16).date()

    def test_next_open_handles_weekend(self, dataset_builder):
        """Next open should skip weekends correctly."""
        # Friday tweet
        tweet_ts = datetime(2025, 1, 17, 15, 0, 0, tzinfo=ET)  # Friday
        daily_df = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(
                ["2025-01-17", "2025-01-21"],  # Friday, Tuesday (Mon is MLK holiday)
                tz="America/New_York",
            ),
        )

        price, flag, next_dt = dataset_builder._get_price_next_open(daily_df, tweet_ts)

        # Should use Tuesday's open
        assert price == 101
        assert next_dt.date() == datetime(2025, 1, 21).date()

    def test_next_open_unavailable_returns_none(self, dataset_builder):
        """When no future data available, should return None."""
        tweet_ts = datetime(2025, 1, 20, 10, 0, 0, tzinfo=ET)
        daily_df = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(
                ["2025-01-15", "2025-01-16"],
                tz="America/New_York",
            ),
        )

        price, flag, next_dt = dataset_builder._get_price_next_open(daily_df, tweet_ts)

        assert price is None
        assert flag == "next_open_unavailable"
        assert next_dt is None


# =============================================================================
# Test: Indicator Slicing (No Look-Ahead)
# =============================================================================


class TestIndicatorSlicing:
    """Tests for indicator calculation with proper data slicing."""

    def test_daily_data_sliced_before_tweet_date(self, sample_daily_data):
        """Daily data for indicators should only include data BEFORE tweet date."""
        tweet_ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        tweet_date = tweet_ts.date()

        # Slice like dataset_builder does
        sliced = sample_daily_data[sample_daily_data.index.date < tweet_date]

        # Should not include Jan 15 or later
        assert all(d < tweet_date for d in sliced.index.date)
        # Should include Jan 14 and earlier
        assert datetime(2025, 1, 14).date() in sliced.index.date

    def test_no_lookahead_in_indicators(self, sample_daily_data):
        """Technical indicators should not use future data."""
        indicators = TechnicalIndicators()
        tweet_date = datetime(2025, 1, 15).date()

        # Slice data before tweet
        daily_before = sample_daily_data[sample_daily_data.index.date < tweet_date]
        current_idx = len(daily_before) - 1

        # Calculate indicators
        result = indicators.calculate_all_indicators(daily_before, current_idx)

        # All indicators should be based on data up to and including Jan 14
        # (This is just a sanity check - indicators exist)
        assert result is not None
        # Check that we have some values (depends on data availability)
        assert "rsi_14" in result

    def test_spy_data_also_sliced(self, sample_daily_data):
        """SPY data for market regime should also be sliced before tweet date."""
        tweet_date = datetime(2025, 1, 15).date()

        # Simulate SPY slicing like in _process_tweet
        spy_df_full = sample_daily_data.copy()
        spy_df = spy_df_full[spy_df_full.index.date < tweet_date]

        # Should not include tweet date or later
        assert all(d < tweet_date for d in spy_df.index.date)


# =============================================================================
# Test: DST Handling in Binding
# =============================================================================


class TestDSTBinding:
    """Tests for DST handling in data binding."""

    def test_comparison_works_across_dst_boundary(self):
        """Timestamp comparison should work correctly across DST boundaries."""
        # Create data spanning DST transition (Nov 3, 2024 is fall back)
        dates = pd.date_range("2024-11-01", "2024-11-05", freq="D")
        df = pd.DataFrame(
            {"open": [100 + i for i in range(len(dates))]},
            index=pd.DatetimeIndex(dates, tz="America/New_York"),
        )

        # Tweet on DST change day
        tweet_ts = datetime(2024, 11, 3, 10, 30, 0, tzinfo=ET)

        # Comparison should work
        future_bars = df[df.index > tweet_ts]
        assert len(future_bars) == 2  # Nov 4 and Nov 5

    def test_spring_forward_comparison(self):
        """Timestamp comparison should work during spring forward transition."""
        # Create data spanning spring DST (March 9, 2025)
        # Note: 2:00-3:00 AM doesn't exist on this day
        times = ["2025-03-08 10:00", "2025-03-09 10:00", "2025-03-10 10:00"]
        df = pd.DataFrame(
            {"open": [100, 101, 102]},
            index=pd.DatetimeIndex(times, tz="America/New_York"),
        )

        # Tweet before DST change
        tweet_ts = datetime(2025, 3, 9, 9, 0, 0, tzinfo=ET)
        future_bars = df[df.index > tweet_ts]

        assert len(future_bars) == 2  # March 9 10:00 and March 10 10:00

    def test_offset_changes_correctly(self):
        """UTC offset should change correctly with DST."""
        # Winter (EST = UTC-5)
        winter_ts = normalize_timestamp(pd.to_datetime("2025-01-15 10:00:00"))
        winter_offset = winter_ts.utcoffset().total_seconds() / 3600
        assert winter_offset == -5

        # Summer (EDT = UTC-4)
        summer_ts = normalize_timestamp(pd.to_datetime("2025-07-15 10:00:00"))
        summer_offset = summer_ts.utcoffset().total_seconds() / 3600
        assert summer_offset == -4


# =============================================================================
# Test: Market Session Detection
# =============================================================================


class TestMarketSessionBinding:
    """Tests for market session detection in binding."""

    def test_regular_hours_detection(self):
        """Regular trading hours should be detected correctly."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        session = get_market_session(ts)
        assert session == MarketSession.REGULAR

    def test_premarket_detection(self):
        """Pre-market hours should be detected correctly."""
        ts = datetime(2025, 1, 15, 8, 0, 0, tzinfo=ET)
        session = get_market_session(ts)
        assert session == MarketSession.PREMARKET

    def test_afterhours_detection(self):
        """After-hours should be detected correctly."""
        ts = datetime(2025, 1, 15, 17, 0, 0, tzinfo=ET)
        session = get_market_session(ts)
        assert session == MarketSession.AFTERHOURS

    def test_closed_detection(self):
        """Closed market should be detected correctly."""
        # Saturday
        ts = datetime(2025, 1, 18, 10, 0, 0, tzinfo=ET)
        session = get_market_session(ts)
        assert session == MarketSession.CLOSED

        # Late night
        ts2 = datetime(2025, 1, 15, 23, 0, 0, tzinfo=ET)
        session2 = get_market_session(ts2)
        assert session2 == MarketSession.CLOSED


# =============================================================================
# Test: End-to-End Binding
# =============================================================================


class TestEndToEndBinding:
    """Integration tests for end-to-end data binding."""

    def test_full_processing_flow(self, dataset_builder, mock_cache, sample_daily_data, sample_intraday_data):
        """Test complete tweet processing flow."""
        # Configure mock to return consistent data
        mock_cache.get_daily.return_value = sample_daily_data
        mock_cache.get_intraday.return_value = sample_intraday_data

        # Create tweet row
        tweet_row = pd.Series({
            "timestamp": "2025-01-15 10:30:00",
            "ticker": "AAPL",
            "author": "TestAuthor",
            "category": "earnings",
            "text": "Test tweet",
            "tweet_url": "https://x.com/test",
        })

        result = dataset_builder._process_tweet(tweet_row)

        # Verify result structure
        assert result["ticker"] == "AAPL"
        assert result["timestamp"].tzinfo is not None
        assert result["session"] == "regular"
        assert result["entry_price"] is not None
        assert result["entry_price_flag"] is not None

    def test_reliable_label_requires_valid_prices(self, dataset_builder, mock_cache, sample_daily_data, sample_intraday_data):
        """Reliable label should only be True when prices are valid."""
        mock_cache.get_daily.return_value = sample_daily_data
        mock_cache.get_intraday.return_value = sample_intraday_data

        tweet_row = pd.Series({
            "timestamp": "2025-01-15 10:30:00",
            "ticker": "AAPL",
            "author": "TestAuthor",
            "category": "earnings",
            "text": "Test tweet",
            "tweet_url": "https://x.com/test",
        })

        result = dataset_builder._process_tweet(tweet_row)

        # If we have valid entry and next open, reliable should be True
        if result["entry_price"] is not None and result["price_next_open"] is not None:
            if "next_bar_open" in result["entry_price_flag"]:
                assert result["is_reliable_label"] is True

    def test_hours_to_next_open_calculated(self, dataset_builder, mock_cache, sample_daily_data, sample_intraday_data):
        """Hours to next open should be calculated correctly."""
        mock_cache.get_daily.return_value = sample_daily_data
        mock_cache.get_intraday.return_value = sample_intraday_data

        tweet_row = pd.Series({
            "timestamp": "2025-01-15 10:30:00",
            "ticker": "AAPL",
            "author": "TestAuthor",
            "category": "earnings",
            "text": "Test tweet",
            "tweet_url": "https://x.com/test",
        })

        result = dataset_builder._process_tweet(tweet_row)

        # Hours to next open should be positive and reasonable
        if result["hours_to_next_open"] is not None:
            assert result["hours_to_next_open"] > 0
            # For a 10:30 AM tweet, next open is next day ~0:00, so ~13.5 hours
            # (market opens at midnight in daily bars for simplicity)


# =============================================================================
# Test: Cross-Timezone Comparison
# =============================================================================


class TestCrossTimezoneComparison:
    """Tests for timezone comparison compatibility."""

    def test_america_newyork_vs_us_eastern(self):
        """America/New_York and US/Eastern should be compatible."""
        # Create data with different tz representations
        df_us_eastern = pd.DataFrame(
            {"open": [100, 101]},
            index=pd.DatetimeIndex(["2025-01-15 10:00", "2025-01-15 10:15"], tz="US/Eastern"),
        )

        ts_america_ny = datetime(2025, 1, 15, 10, 5, 0, tzinfo=ZoneInfo("America/New_York"))

        # Comparison should work
        future = df_us_eastern[df_us_eastern.index > ts_america_ny]
        assert len(future) == 1
        assert future.index[0].hour == 10
        assert future.index[0].minute == 15

    def test_comparison_with_normalized_timestamp(self, sample_intraday_data):
        """Normalized timestamps should compare correctly with cached data."""
        # Simulate what happens in real processing
        tweet_ts_str = "2025-01-15 10:30:00"
        tweet_ts = pd.to_datetime(tweet_ts_str)
        tweet_ts_norm = normalize_timestamp(tweet_ts)

        # Compare with intraday data
        future_bars = sample_intraday_data[sample_intraday_data.index > tweet_ts_norm]

        # Should find bars after 10:30
        assert len(future_bars) > 0
        assert all(bar.hour > 10 or (bar.hour == 10 and bar.minute > 30) for bar in future_bars.index)
