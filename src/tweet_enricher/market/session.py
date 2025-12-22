"""Market session detection and timestamp utilities."""

from datetime import date, datetime
from enum import Enum
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from tweet_enricher.config import (
    AFTERHOURS_END,
    PREMARKET_START,
)
from tweet_enricher.utils.timezone import (
    normalize_dataframe_timezone,
    normalize_timestamp,
)

# Re-export for backward compatibility
__all__ = [
    "MarketSession",
    "get_market_session",
    "is_market_holiday",
    "normalize_timestamp",
    "normalize_dataframe_timezone",
]


class MarketSession(Enum):
    """Market session types."""

    REGULAR = "regular"
    PREMARKET = "premarket"
    AFTERHOURS = "afterhours"
    CLOSED = "closed"


@lru_cache(maxsize=1)
def _get_nyse_calendar() -> mcal.MarketCalendar:
    """Get cached NYSE calendar instance."""
    return mcal.get_calendar("NYSE")


def is_market_holiday(check_date: date) -> bool:
    """
    Check if a given date is a US market holiday.

    Args:
        check_date: Date to check (datetime.date object)

    Returns:
        True if the date is a market holiday, False otherwise

    Note:
        This checks against the NYSE holiday schedule, which includes:
        New Year's Day, MLK Day, Presidents' Day, Good Friday,
        Memorial Day, Juneteenth, Independence Day, Labor Day,
        Thanksgiving, Christmas, and any special closures.
    """
    nyse = _get_nyse_calendar()

    # Get valid trading days for a range around the date
    # We check if the date is NOT in the list of valid trading days
    start = pd.Timestamp(check_date)
    end = pd.Timestamp(check_date)

    schedule = nyse.schedule(start_date=start, end_date=end)

    # If schedule is empty, the date is not a trading day (holiday or weekend)
    return schedule.empty


@lru_cache(maxsize=128)
def _get_market_schedule(check_date: date) -> tuple[int, int] | None:
    """
    Get actual market open/close times for a date (handles half-days).

    Args:
        check_date: Date to get schedule for

    Returns:
        Tuple of (open_minutes, close_minutes) since midnight, or None if market closed
    """
    nyse = _get_nyse_calendar()
    schedule = nyse.schedule(start_date=pd.Timestamp(check_date), end_date=pd.Timestamp(check_date))

    if schedule.empty:
        return None

    # Extract actual times and convert to ET
    market_open_utc = schedule.iloc[0]["market_open"]
    market_close_utc = schedule.iloc[0]["market_close"]

    # Convert to ET
    market_open_et = market_open_utc.tz_convert("America/New_York")
    market_close_et = market_close_utc.tz_convert("America/New_York")

    # Convert to minutes since midnight
    open_minutes = market_open_et.hour * 60 + market_open_et.minute
    close_minutes = market_close_et.hour * 60 + market_close_et.minute

    return (open_minutes, close_minutes)


def get_market_session(timestamp: datetime) -> MarketSession:
    """
    Determine market session for a given timestamp.

    Handles:
    - Regular trading hours (actual times from NYSE calendar, handles half-days)
    - Pre-market (4:00 AM - market open ET)
    - After-hours (market close - 8:00 PM ET)
    - Closed (weekends, holidays, overnight)

    Note:
        On half-trading days (e.g., Thanksgiving eve), market closes at 1:00 PM.
        This function correctly classifies tweets after early close as AFTERHOURS.

    Args:
        timestamp: Timestamp to check

    Returns:
        MarketSession enum value
    """
    # Ensure timestamp is in ET
    timestamp = normalize_timestamp(timestamp)

    # Check if it's a weekend
    if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return MarketSession.CLOSED

    # Get actual market schedule for this date (handles half-days)
    schedule = _get_market_schedule(timestamp.date())

    if schedule is None:
        # Market holiday
        return MarketSession.CLOSED

    actual_open, actual_close = schedule

    # Convert timestamp to minutes since midnight
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute

    if actual_open <= minutes_since_midnight < actual_close:
        return MarketSession.REGULAR
    elif PREMARKET_START <= minutes_since_midnight < actual_open:
        return MarketSession.PREMARKET
    elif actual_close <= minutes_since_midnight < AFTERHOURS_END:
        return MarketSession.AFTERHOURS
    else:
        return MarketSession.CLOSED
