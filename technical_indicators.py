"""
Technical indicators calculator using pandas-ta.

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
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

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
        closes = df.iloc[current_idx - window:current_idx + 1]["close"].values

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
        avg_volume = df.iloc[current_idx - window:current_idx]["volume"].mean()

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

    def calculate_all_indicators(
        self, df: pd.DataFrame, current_idx: int
    ) -> dict:
        """
        Calculate all technical indicators at once.

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position

        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {
            "return_1d": self.calculate_return(df, current_idx, periods=1),
            "volatility_7d": self.calculate_volatility(df, current_idx, window=7),
            "relative_volume": self.calculate_relative_volume(df, current_idx),
            "rsi_14": self.calculate_rsi(df, current_idx),
            "distance_from_ma_20": self.calculate_distance_from_ma(df, current_idx),
        }

        return indicators
