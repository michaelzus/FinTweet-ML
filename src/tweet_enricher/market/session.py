"""Market session detection and timestamp utilities."""

from datetime import datetime
from enum import Enum

import pandas as pd

from tweet_enricher.config import (
    AFTERHOURS_END,
    ET,
    MARKET_CLOSE,
    MARKET_OPEN,
    PREMARKET_START,
)


class MarketSession(Enum):
    """Market session types."""

    REGULAR = "regular"
    PREMARKET = "premarket"
    AFTERHOURS = "afterhours"
    CLOSED = "closed"


def normalize_timestamp(timestamp: datetime) -> datetime:
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


def normalize_dataframe_timezone(df: pd.DataFrame) -> pd.DataFrame:
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


def get_market_session(timestamp: datetime) -> MarketSession:
    """
    Determine market session for a given timestamp.

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
