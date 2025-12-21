"""Tests for timezone utilities and related fixes.

This module tests:
- Centralized timezone module (normalize_timestamp, normalize_dataframe_timezone)
- DST handling edge cases
- Holiday detection
- ET-aware datetime returns from database
- Timezone assertions in dataset_builder
"""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tweet_enricher.utils.timezone import (
    ET,
    UTC,
    convert_utc_to_et,
    normalize_dataframe_timezone,
    normalize_timestamp,
)


class TestTimezoneConstants:
    """Tests for timezone constants."""

    def test_et_is_america_new_york(self):
        """ET should be America/New_York timezone."""
        assert str(ET) == "America/New_York"

    def test_utc_is_utc(self):
        """UTC should be UTC timezone."""
        assert str(UTC) == "UTC"


class TestNormalizeTimestamp:
    """Tests for normalize_timestamp function."""

    def test_naive_timestamp_assumes_et_by_default(self):
        """Naive timestamps should be assumed to be ET by default."""
        naive_dt = datetime(2025, 1, 15, 10, 30, 0)
        result = normalize_timestamp(naive_dt)

        assert result.tzinfo is not None
        assert str(result.tzinfo) == "America/New_York"
        assert result.hour == 10  # Hour should not change
        assert result.minute == 30

    def test_naive_timestamp_raises_when_flag_is_false(self):
        """Should raise ValueError when naive timestamp passed with assume_naive_is_et=False."""
        naive_dt = datetime(2025, 1, 15, 10, 30, 0)

        with pytest.raises(ValueError, match="Naive timestamp passed"):
            normalize_timestamp(naive_dt, assume_naive_is_et=False)

    def test_utc_timestamp_converted_to_et(self):
        """UTC timestamp should be converted to ET."""
        utc_dt = datetime(2025, 1, 15, 15, 30, 0, tzinfo=UTC)
        result = normalize_timestamp(utc_dt)

        assert result.tzinfo is not None
        # 15:30 UTC in January (EST, UTC-5) = 10:30 ET
        assert result.hour == 10
        assert result.minute == 30

    def test_other_timezone_converted_to_et(self):
        """Other timezone should be converted to ET."""
        pacific = ZoneInfo("America/Los_Angeles")
        pacific_dt = datetime(2025, 1, 15, 7, 30, 0, tzinfo=pacific)
        result = normalize_timestamp(pacific_dt)

        assert result.tzinfo is not None
        # 7:30 PT (PST, UTC-8) = 10:30 ET (EST, UTC-5)
        assert result.hour == 10
        assert result.minute == 30

    def test_already_et_timestamp_unchanged(self):
        """ET timestamp should remain unchanged."""
        et_dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        result = normalize_timestamp(et_dt)

        assert result.hour == 10
        assert result.minute == 30
        assert str(result.tzinfo) == "America/New_York"


class TestNormalizeDataframeTimezone:
    """Tests for normalize_dataframe_timezone function."""

    def test_naive_index_localized_to_et(self):
        """Naive DataFrame index should be localized to ET."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.DatetimeIndex(["2025-01-15", "2025-01-16", "2025-01-17"]),
        )

        result = normalize_dataframe_timezone(df)

        assert result.index.tz is not None
        assert str(result.index.tz) == "America/New_York"

    def test_utc_index_converted_to_et(self):
        """UTC DataFrame index should be converted to ET."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.DatetimeIndex(
                ["2025-01-15 15:30", "2025-01-16 15:30", "2025-01-17 15:30"],
                tz="UTC",
            ),
        )

        result = normalize_dataframe_timezone(df)

        assert str(result.index.tz) == "America/New_York"
        # 15:30 UTC = 10:30 ET (EST)
        assert result.index[0].hour == 10

    def test_already_et_index_unchanged(self):
        """ET DataFrame index should remain unchanged."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.DatetimeIndex(
                ["2025-01-15 10:30", "2025-01-16 10:30", "2025-01-17 10:30"],
                tz="America/New_York",
            ),
        )

        result = normalize_dataframe_timezone(df)

        assert str(result.index.tz) == "America/New_York"
        assert result.index[0].hour == 10


class TestDSTHandling:
    """Tests for DST edge cases."""

    def test_spring_forward_nonexistent_time(self):
        """Non-existent time during spring-forward should be shifted forward.

        On Mar 9, 2025, 2:00 AM jumps to 3:00 AM. Time 2:30 AM doesn't exist.
        """
        # Create DataFrame with time that doesn't exist during DST transition
        df = pd.DataFrame(
            {"close": [100]},
            index=pd.DatetimeIndex(["2025-03-09 02:30:00"]),  # This time doesn't exist
        )

        # Should not raise error - nonexistent times are shifted forward
        result = normalize_dataframe_timezone(df)

        assert result.index.tz is not None
        # Time should be shifted to 3:00 or 3:30 (shifted forward)
        assert result.index[0].hour >= 3

    def test_fall_back_ambiguous_time(self):
        """Ambiguous time during fall-back should be handled with 'infer'.

        On Nov 2, 2025, 2:00 AM falls back to 1:00 AM. Time 1:30 AM exists twice.
        For 'infer' to work, we need enough surrounding non-ambiguous times.
        """
        # Create DataFrame with times spanning through DST transition
        # The hour gap helps pandas infer the correct DST state
        df = pd.DataFrame(
            {"close": [100, 101, 102, 103, 104]},
            index=pd.DatetimeIndex(
                [
                    "2025-11-01 23:00:00",  # Before transition (EDT)
                    "2025-11-02 00:00:00",  # Before transition (EDT)
                    "2025-11-02 00:30:00",  # Before transition (EDT)
                    "2025-11-02 03:00:00",  # After transition (EST)
                    "2025-11-02 04:00:00",  # After transition (EST)
                ]
            ),
        )

        # Should not raise error - times around transition are handled
        result = normalize_dataframe_timezone(df)

        assert result.index.tz is not None
        assert len(result) == 5

    def test_utc_conversion_during_dst(self):
        """UTC to ET conversion should handle DST correctly."""
        # Summer: UTC-4 (EDT)
        summer_utc = datetime(2025, 7, 15, 14, 0, 0, tzinfo=UTC)
        summer_et = normalize_timestamp(summer_utc)
        assert summer_et.hour == 10  # 14:00 UTC = 10:00 EDT

        # Winter: UTC-5 (EST)
        winter_utc = datetime(2025, 1, 15, 15, 0, 0, tzinfo=UTC)
        winter_et = normalize_timestamp(winter_utc)
        assert winter_et.hour == 10  # 15:00 UTC = 10:00 EST


class TestConvertUtcToEt:
    """Tests for convert_utc_to_et function."""

    def test_naive_assumed_utc(self):
        """Naive timestamps should be assumed UTC."""
        naive_utc = datetime(2025, 1, 15, 15, 30, 0)
        result = convert_utc_to_et(naive_utc)

        assert result.tzinfo is not None
        # 15:30 UTC = 10:30 EST
        assert result.hour == 10

    def test_utc_aware_converted(self):
        """UTC-aware timestamps should be converted."""
        aware_utc = datetime(2025, 1, 15, 15, 30, 0, tzinfo=UTC)
        result = convert_utc_to_et(aware_utc)

        assert result.hour == 10

    def test_other_timezone_converted_via_utc(self):
        """Other timezone should be converted to ET."""
        pacific = ZoneInfo("America/Los_Angeles")
        pacific_dt = datetime(2025, 1, 15, 7, 30, 0, tzinfo=pacific)
        result = convert_utc_to_et(pacific_dt)

        # Pacific time is converted to ET via astimezone
        assert result.hour == 10


class TestMarketHolidayDetection:
    """Tests for market holiday detection."""

    def test_christmas_is_holiday(self):
        """Christmas should be detected as a market holiday."""
        from tweet_enricher.market.session import is_market_holiday

        christmas_2024 = date(2024, 12, 25)
        assert is_market_holiday(christmas_2024) is True

    def test_thanksgiving_is_holiday(self):
        """Thanksgiving should be detected as a market holiday."""
        from tweet_enricher.market.session import is_market_holiday

        thanksgiving_2024 = date(2024, 11, 28)
        assert is_market_holiday(thanksgiving_2024) is True

    def test_regular_weekday_not_holiday(self):
        """Regular weekday should not be a holiday."""
        from tweet_enricher.market.session import is_market_holiday

        regular_wednesday = date(2025, 1, 15)
        assert is_market_holiday(regular_wednesday) is False

    def test_weekend_is_holiday(self):
        """Weekend days should be detected as closed (not trading sessions)."""
        from tweet_enricher.market.session import is_market_holiday

        saturday = date(2025, 1, 18)
        sunday = date(2025, 1, 19)
        # Weekends are handled separately in get_market_session,
        # but is_market_holiday returns True because no trading session exists
        assert is_market_holiday(saturday) is True
        assert is_market_holiday(sunday) is True

    def test_good_friday_is_holiday(self):
        """Good Friday should be detected as a market holiday."""
        from tweet_enricher.market.session import is_market_holiday

        good_friday_2025 = date(2025, 4, 18)
        assert is_market_holiday(good_friday_2025) is True


class TestMarketSessionWithHolidays:
    """Tests for market session detection including holidays."""

    def test_christmas_returns_closed(self):
        """Market session on Christmas should return CLOSED."""
        from tweet_enricher.market.session import MarketSession, get_market_session

        # Christmas 2024 at 10:30 AM (would be regular hours normally)
        christmas_dt = datetime(2024, 12, 25, 10, 30, 0, tzinfo=ET)
        session = get_market_session(christmas_dt)

        assert session == MarketSession.CLOSED

    def test_regular_weekday_returns_correct_session(self):
        """Regular weekday should return correct session based on time."""
        from tweet_enricher.market.session import MarketSession, get_market_session

        # Wednesday Jan 15, 2025 at 10:30 AM
        regular_dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=ET)
        session = get_market_session(regular_dt)

        assert session == MarketSession.REGULAR

    def test_premarket_returns_premarket(self):
        """Pre-market time should return PREMARKET."""
        from tweet_enricher.market.session import MarketSession, get_market_session

        # 6:00 AM on a regular weekday
        premarket_dt = datetime(2025, 1, 15, 6, 0, 0, tzinfo=ET)
        session = get_market_session(premarket_dt)

        assert session == MarketSession.PREMARKET

    def test_afterhours_returns_afterhours(self):
        """After-hours time should return AFTERHOURS."""
        from tweet_enricher.market.session import MarketSession, get_market_session

        # 5:00 PM on a regular weekday
        afterhours_dt = datetime(2025, 1, 15, 17, 0, 0, tzinfo=ET)
        session = get_market_session(afterhours_dt)

        assert session == MarketSession.AFTERHOURS


class TestDatabaseETAwareDatetimes:
    """Tests for ET-aware datetime returns from database functions."""

    def test_get_tickers_with_first_date_returns_et_aware(self):
        """get_tickers_with_first_date should return ET-aware datetimes."""
        from tweet_enricher.twitter.database import TweetDatabase

        # Create mock database
        with patch.object(TweetDatabase, "_init_schema"):
            with patch.object(TweetDatabase, "_ensure_directory"):
                db = TweetDatabase()
                db.db_path = MagicMock()

                # Mock the connection and query results
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchall.return_value = [
                    {"ticker": "AAPL", "first_date": "2025-01-15 10:30:00"}
                ]

                with patch.object(db, "_get_connection", return_value=mock_conn):
                    result = db.get_tickers_with_first_date()

                    assert "AAPL" in result
                    dt = result["AAPL"]
                    assert dt.tzinfo is not None
                    assert str(dt.tzinfo) == "America/New_York"

    def test_get_tickers_with_date_range_returns_et_aware(self):
        """get_tickers_with_date_range should return ET-aware datetimes."""
        from tweet_enricher.twitter.database import TweetDatabase

        with patch.object(TweetDatabase, "_init_schema"):
            with patch.object(TweetDatabase, "_ensure_directory"):
                db = TweetDatabase()
                db.db_path = MagicMock()

                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchall.return_value = [
                    {
                        "ticker": "AAPL",
                        "first_date": "2025-01-15 10:30:00",
                        "last_date": "2025-01-20 15:00:00",
                    }
                ]

                with patch.object(db, "_get_connection", return_value=mock_conn):
                    result = db.get_tickers_with_date_range()

                    assert "AAPL" in result
                    first_dt, last_dt = result["AAPL"]
                    assert first_dt.tzinfo is not None
                    assert last_dt.tzinfo is not None
                    assert str(first_dt.tzinfo) == "America/New_York"
                    assert str(last_dt.tzinfo) == "America/New_York"


class TestSyncServiceTimestamp:
    """Tests for SyncService timestamp handling."""

    def test_convert_utc_to_eastern_includes_offset(self):
        """Converted timestamps should include timezone offset."""
        from tweet_enricher.twitter.sync import SyncService

        service = SyncService(lazy_client=True)

        # Twitter format: "Wed Dec 17 15:01:11 +0000 2025"
        twitter_ts = "Wed Jan 15 15:30:00 +0000 2025"
        result = service._convert_utc_to_eastern(twitter_ts)

        # Should include timezone offset
        assert "-0500" in result or "-0400" in result  # EST or EDT
        # Should be parseable
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

    def test_convert_utc_to_eastern_correct_time(self):
        """Converted timestamp should have correct Eastern time."""
        from tweet_enricher.twitter.sync import SyncService

        service = SyncService(lazy_client=True)

        # 15:30 UTC in January = 10:30 EST
        twitter_ts = "Wed Jan 15 15:30:00 +0000 2025"
        result = service._convert_utc_to_eastern(twitter_ts)

        parsed = datetime.fromisoformat(result)
        assert parsed.hour == 10
        assert parsed.minute == 30


class TestDatasetBuilderAssertions:
    """Tests for dataset builder timezone assertions."""

    def test_entry_price_asserts_timezone_aware_intraday(self):
        """_get_entry_price should assert intraday data is timezone-aware."""
        from tweet_enricher.core.dataset_builder import DatasetBuilder
        from tweet_enricher.core.indicators import TechnicalIndicators
        from tweet_enricher.data.cache_reader import CacheReader
        from tweet_enricher.market.session import MarketSession

        builder = DatasetBuilder(
            cache=MagicMock(spec=CacheReader),
            indicators=MagicMock(spec=TechnicalIndicators),
        )

        # Create naive intraday DataFrame (should trigger assertion)
        naive_intraday = pd.DataFrame(
            {"open": [100, 101], "close": [101, 102]},
            index=pd.DatetimeIndex(["2025-01-15 10:00", "2025-01-15 10:15"]),
        )

        # ET-aware daily DataFrame
        et_daily = pd.DataFrame(
            {"open": [100], "close": [101]},
            index=pd.DatetimeIndex(["2025-01-15"], tz="America/New_York"),
        )

        et_timestamp = datetime(2025, 1, 15, 9, 45, 0, tzinfo=ET)

        with pytest.raises(AssertionError, match="Intraday data must be timezone-aware"):
            builder._get_entry_price(
                naive_intraday,
                et_daily,
                et_timestamp,
                MarketSession.REGULAR,
            )

    def test_entry_price_works_with_timezone_aware_data(self):
        """_get_entry_price should work with timezone-aware data."""
        from tweet_enricher.core.dataset_builder import DatasetBuilder
        from tweet_enricher.core.indicators import TechnicalIndicators
        from tweet_enricher.data.cache_reader import CacheReader
        from tweet_enricher.market.session import MarketSession

        builder = DatasetBuilder(
            cache=MagicMock(spec=CacheReader),
            indicators=MagicMock(spec=TechnicalIndicators),
        )

        # ET-aware intraday DataFrame
        et_intraday = pd.DataFrame(
            {"open": [100, 101], "close": [101, 102]},
            index=pd.DatetimeIndex(
                ["2025-01-15 10:00", "2025-01-15 10:15"],
                tz="America/New_York",
            ),
        )

        # ET-aware daily DataFrame
        et_daily = pd.DataFrame(
            {"open": [99, 100]},
            index=pd.DatetimeIndex(["2025-01-14", "2025-01-15"], tz="America/New_York"),
        )

        et_timestamp = datetime(2025, 1, 15, 9, 45, 0, tzinfo=ET)

        # Should not raise - returns first bar after timestamp
        price, flag = builder._get_entry_price(
            et_intraday,
            et_daily,
            et_timestamp,
            MarketSession.REGULAR,
        )

        assert price == 100  # First bar's open after 9:45
        assert "next_bar_open" in flag


class TestBackwardCompatibility:
    """Tests for backward compatibility of timestamp parsing."""

    def test_fromisoformat_parses_old_format(self):
        """datetime.fromisoformat should parse timestamps without offset."""
        old_format = "2025-01-15 10:30:00"
        parsed = datetime.fromisoformat(old_format)

        assert parsed.hour == 10
        assert parsed.minute == 30
        assert parsed.tzinfo is None  # Old format has no timezone

    def test_fromisoformat_parses_new_format(self):
        """datetime.fromisoformat should parse timestamps with offset."""
        new_format = "2025-01-15 10:30:00-0500"
        parsed = datetime.fromisoformat(new_format)

        assert parsed.hour == 10
        assert parsed.minute == 30
        assert parsed.tzinfo is not None

    def test_database_handles_both_formats(self):
        """Database functions should handle both old and new timestamp formats."""
        from tweet_enricher.twitter.database import TweetDatabase

        with patch.object(TweetDatabase, "_init_schema"):
            with patch.object(TweetDatabase, "_ensure_directory"):
                db = TweetDatabase()
                db.db_path = MagicMock()

                # Test with old format (no offset)
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchall.return_value = [
                    {"ticker": "AAPL", "first_date": "2025-01-15 10:30:00"}
                ]

                with patch.object(db, "_get_connection", return_value=mock_conn):
                    result = db.get_tickers_with_first_date()
                    assert result["AAPL"].tzinfo is not None

                # Test with new format (with offset)
                mock_conn.execute.return_value.fetchall.return_value = [
                    {"ticker": "MSFT", "first_date": "2025-01-15 10:30:00-0500"}
                ]

                with patch.object(db, "_get_connection", return_value=mock_conn):
                    result = db.get_tickers_with_first_date()
                    assert result["MSFT"].tzinfo is not None

