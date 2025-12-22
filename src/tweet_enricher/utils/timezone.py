"""Centralized timezone utilities for the tweet_enricher package.

This module provides DST-safe timezone handling functions using Python's
modern zoneinfo module (3.9+) instead of pytz for better DST handling.

All timestamps in this pipeline are normalized to US Eastern Time (ET)
for consistency with US market hours.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# Standard timezone constants
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def normalize_timestamp(timestamp: datetime, assume_naive_is_et: bool = True) -> datetime:
    """
    Normalize a timestamp to timezone-aware ET datetime.

    Args:
        timestamp: Input datetime (may be naive or timezone-aware)
        assume_naive_is_et: If True, assume naive timestamps are already in ET.
                           If False, raise ValueError for naive timestamps.

    Returns:
        Timezone-aware datetime in US Eastern Time

    Raises:
        ValueError: If timestamp is naive and assume_naive_is_et is False

    Note:
        When assume_naive_is_et=True, naive timestamps are interpreted as
        already being in ET (not converted from UTC). This is the correct
        behavior for timestamps loaded from our database, which stores
        ET strings without timezone info.
    """
    if timestamp.tzinfo is None:
        if assume_naive_is_et:
            # Use replace() instead of localize() - works correctly with zoneinfo
            return timestamp.replace(tzinfo=ET)
        raise ValueError(
            "Naive timestamp passed without explicit assumption. "
            "Either pass a timezone-aware datetime or set assume_naive_is_et=True."
        )
    return timestamp.astimezone(ET)


def normalize_dataframe_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame index to timezone-aware ET with DST-safe handling.

    This function handles Daylight Saving Time transitions safely:
    - Fall-back (Nov): Marks ambiguous times as NaT and filters them out
    - Spring-forward (Mar): Shifts non-existent times forward (e.g., 2:30 AM -> 3:00 AM)

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with timezone-aware index in US Eastern Time.
        Rows with ambiguous DST times are filtered out with a warning.

    Note:
        If the index is already timezone-aware, it will be converted to ET.
        If the index is naive, it will be localized as ET (assumed to already
        be in ET, not converted from UTC).

        Using ambiguous="NaT" is safer than "infer" because "infer" can raise
        AmbiguousTimeError when pandas cannot determine the correct offset
        from the data sequence (e.g., gaps during DST transition).
    """
    if df.index.tz is None:
        df.index = df.index.tz_localize(
            "America/New_York",
            ambiguous="NaT",  # Mark ambiguous times as NaT (safer than "infer")
            nonexistent="shift_forward",  # Handle spring-forward DST (Mar)
        )
        # Filter out rows with ambiguous DST times (now NaT)
        nat_count = df.index.isna().sum()
        if nat_count > 0:
            logger.warning(f"Dropped {nat_count} rows with ambiguous DST times")
            df = df[df.index.notna()].copy()
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


def convert_utc_to_et(timestamp: datetime) -> datetime:
    """
    Explicitly convert a UTC timestamp to ET.

    Use this when you know the input is in UTC and needs conversion.

    Args:
        timestamp: UTC datetime (naive assumed UTC, or timezone-aware)

    Returns:
        Timezone-aware datetime in US Eastern Time
    """
    if timestamp.tzinfo is None:
        # Assume naive is UTC
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(ET)
