"""Technical indicators calculator using pandas-ta.

This module provides an OOP interface for calculating various technical indicators
needed for tweet enrichment.
"""

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta


class TechnicalIndicators:
    """
    Calculates technical indicators for stock data.

    Uses pandas-ta library for efficient indicator calculations.
    """

    def __init__(self):
        """Initialize the TechnicalIndicators calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_return(self, df: pd.DataFrame, current_idx: int, periods: int = 1) -> Optional[float]:
        """
        Calculate return over specified periods.

        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            current_idx: Current index position
            periods: Number of periods to look back (default: 1)

        Returns:
            Return as decimal (e.g., 0.05 = 5% gain) or None if not enough data
        """
        if current_idx < periods or current_idx >= len(df):
            return None

        current_close = df.iloc[current_idx]["close"]
        prev_close = df.iloc[current_idx - periods]["close"]

        if prev_close == 0:
            return None

        return (current_close - prev_close) / prev_close

    def calculate_volatility(self, df: pd.DataFrame, current_idx: int, window: int = 7) -> Optional[float]:
        """
        Calculate historical volatility (standard deviation of returns).

        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            current_idx: Current index position
            window: Lookback window in periods (default: 7)

        Returns:
            Historical volatility for the specified window or None if not enough data
        """
        if current_idx < window:
            return None

        # Get closing prices for the window
        closes = df.iloc[current_idx - window : current_idx + 1]["close"].values

        if len(closes) < window + 1:
            return None

        # Calculate returns and their standard deviation
        returns = pd.Series(closes).pct_change().dropna()

        return returns.std()

    def calculate_relative_volume(self, df: pd.DataFrame, current_idx: int, window: int = 20) -> Optional[float]:
        """
        Calculate relative volume (current volume / average volume).

        Args:
            df: DataFrame with OHLCV data (must have 'volume' column)
            current_idx: Current index position
            window: Lookback window for average calculation (default: 20 days)

        Returns:
            Relative volume ratio or None if not enough data
        """
        if current_idx < window or current_idx >= len(df):
            return None

        current_volume = df.iloc[current_idx]["volume"]
        avg_volume = df.iloc[current_idx - window : current_idx]["volume"].mean()

        if avg_volume == 0:
            return None

        return current_volume / avg_volume

    def calculate_rsi(self, df: pd.DataFrame, current_idx: int, period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            current_idx: Current index position
            period: RSI period (default: 14)

        Returns:
            RSI value (0-100) or None if not enough data
        """
        if current_idx < period:
            return None

        # Need at least period+1 data points for RSI
        subset = df.iloc[: current_idx + 1].copy()

        if len(subset) < period + 1:
            return None

        # Calculate RSI using pandas-ta
        rsi_series = ta.rsi(subset["close"], length=period)

        if rsi_series is None or rsi_series.empty:
            return None

        # Get the RSI value at current index
        rsi_value = rsi_series.iloc[-1]

        return rsi_value if pd.notna(rsi_value) else None

    def calculate_distance_from_ma(self, df: pd.DataFrame, current_idx: int, period: int = 20) -> Optional[float]:
        """
        Calculate distance from moving average as percentage.

        Args:
            df: DataFrame with OHLCV data (must have 'close' column)
            current_idx: Current index position
            period: MA period (default: 20)

        Returns:
            Distance from MA as decimal (e.g., 0.05 = 5% above MA) or None if not enough data
        """
        if current_idx < period - 1:
            return None

        # Calculate SMA
        subset = df.iloc[: current_idx + 1].copy()
        sma = ta.sma(subset["close"], length=period)

        if sma is None or sma.empty:
            return None

        current_close = df.iloc[current_idx]["close"]
        current_ma = sma.iloc[-1]

        if pd.isna(current_ma) or current_ma == 0:
            return None

        return (current_close - current_ma) / current_ma

    def calculate_above_ma(self, df: pd.DataFrame, current_idx: int, period: int = 20) -> Optional[int]:
        """
        Check if price is above moving average (binary feature).

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position
            period: MA period (default: 20)

        Returns:
            1 if above MA, 0 if below/at, None if not enough data
        """
        if current_idx < period - 1:
            return None

        subset = df.iloc[: current_idx + 1].copy()
        ma = ta.sma(subset["close"], length=period)

        if ma is None or ma.empty or pd.isna(ma.iloc[-1]):
            return None

        current_close = df.iloc[current_idx]["close"]
        return 1 if current_close > ma.iloc[-1] else 0

    def calculate_ma_slope(self, df: pd.DataFrame, current_idx: int, period: int = 20, lookback: int = 5) -> Optional[float]:
        """
        Calculate slope of moving average (trend direction).

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position
            period: MA period (default: 20)
            lookback: How many periods to measure slope over (default: 5)

        Returns:
            MA slope as percentage change or None if not enough data
        """
        if current_idx < period + lookback - 1:
            return None

        subset = df.iloc[: current_idx + 1].copy()
        ma = ta.sma(subset["close"], length=period)

        if ma is None or len(ma) < lookback + 1:
            return None

        ma_current = ma.iloc[-1]
        ma_past = ma.iloc[-1 - lookback]

        if pd.isna(ma_current) or pd.isna(ma_past) or ma_past == 0:
            return None

        return (ma_current - ma_past) / ma_past

    def calculate_gap_open(self, df: pd.DataFrame, current_idx: int) -> Optional[float]:
        """
        Calculate overnight gap (open vs previous close).

        Args:
            df: DataFrame with OHLCV data (must have 'open' and 'close')
            current_idx: Current index position

        Returns:
            Gap as percentage or None if not enough data
        """
        if current_idx < 1 or current_idx >= len(df):
            return None

        open_today = df.iloc[current_idx]["open"]
        close_yesterday = df.iloc[current_idx - 1]["close"]

        if close_yesterday == 0:
            return None

        return (open_today - close_yesterday) / close_yesterday

    def calculate_intraday_range(self, df: pd.DataFrame, current_idx: int) -> Optional[float]:
        """
        Calculate intraday volatility (high-low range).

        Args:
            df: DataFrame with OHLCV data (must have 'high', 'low', 'close')
            current_idx: Current index position

        Returns:
            Intraday range as percentage of previous close or None
        """
        if current_idx < 1 or current_idx >= len(df):
            return None

        high_today = df.iloc[current_idx]["high"]
        low_today = df.iloc[current_idx]["low"]
        close_yesterday = df.iloc[current_idx - 1]["close"]

        if close_yesterday == 0:
            return None

        return (high_today - low_today) / close_yesterday

    def calculate_all_indicators(self, df: pd.DataFrame, current_idx: int) -> dict:
        """
        Calculate all technical indicators at once.

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position

        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {
            # Original core indicators
            "return_1d": self.calculate_return(df, current_idx, periods=1),
            "volatility_7d": self.calculate_volatility(df, current_idx, window=7),
            "relative_volume": self.calculate_relative_volume(df, current_idx),
            "rsi_14": self.calculate_rsi(df, current_idx),
            "distance_from_ma_20": self.calculate_distance_from_ma(df, current_idx),
            # NEW: Multi-period momentum
            "return_5d": self.calculate_return(df, current_idx, periods=5),
            "return_20d": self.calculate_return(df, current_idx, periods=20),
            # NEW: Trend confirmation
            "above_ma_20": self.calculate_above_ma(df, current_idx, period=20),
            "slope_ma_20": self.calculate_ma_slope(df, current_idx, period=20, lookback=5),
            # NEW: Shock/Gap features
            "gap_open": self.calculate_gap_open(df, current_idx),
            "intraday_range": self.calculate_intraday_range(df, current_idx),
        }

        return indicators
