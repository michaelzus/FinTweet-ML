"""Market session detection and timestamp utilities."""

from datetime import date, datetime
from enum import Enum
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from tweet_enricher.config import (
    AFTERHOURS_END,
    MARKET_CLOSE,
    MARKET_OPEN,
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


def get_market_session(timestamp: datetime) -> MarketSession:
    """
    Determine market session for a given timestamp.

    Handles:
    - Regular trading hours (9:30 AM - 4:00 PM ET)
    - Pre-market (4:00 AM - 9:30 AM ET)
    - After-hours (4:00 PM - 8:00 PM ET)
    - Closed (weekends, holidays, overnight)

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

    # Check if it's a market holiday
    if is_market_holiday(timestamp.date()):
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
