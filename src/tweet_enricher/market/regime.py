"""Market regime classification based on SPY behavior.

This module classifies market conditions into regimes (trending_up, trending_down,
volatile, calm) to provide context for tweet enrichment.
"""

import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

from tweet_enricher.config import (
    REGIME_LOOKBACK_RETURN,
    REGIME_LOOKBACK_VOL,
    REGIME_TRENDING_THRESHOLD,
    REGIME_VOLATILE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Classifies market regime based on SPY behavior.

    Uses rolling returns and volatility to determine if the market is
    trending up, trending down, volatile, or calm.
    """

    def __init__(
        self,
        trending_threshold: float = REGIME_TRENDING_THRESHOLD,
        volatile_threshold: float = REGIME_VOLATILE_THRESHOLD,
        lookback_return: int = REGIME_LOOKBACK_RETURN,
        lookback_vol: int = REGIME_LOOKBACK_VOL,
    ):
        """
        Initialize the regime classifier.

        Args:
            trending_threshold: Return threshold for trending regimes (default: 2%)
            volatile_threshold: Annualized volatility threshold for volatile regime (default: 18%)
            lookback_return: Days to calculate return over (default: 5)
            lookback_vol: Days to calculate volatility over (default: 5)
        """
        self.trending_threshold = trending_threshold
        self.volatile_threshold = volatile_threshold
        self.lookback_return = lookback_return
        self.lookback_vol = lookback_vol
        self._regime_cache = {}

    def _calculate_return(self, df: pd.DataFrame, current_idx: int, periods: int) -> Optional[float]:
        """
        Calculate return over specified periods.

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position
            periods: Number of periods to look back

        Returns:
            Return as decimal or None if not enough data
        """
        if current_idx < periods or current_idx >= len(df):
            return None

        current_close = df.iloc[current_idx]["close"]
        prev_close = df.iloc[current_idx - periods]["close"]

        if prev_close == 0:
            return None

        return (current_close - prev_close) / prev_close

    def _calculate_volatility(self, df: pd.DataFrame, current_idx: int, window: int) -> Optional[float]:
        """
        Calculate annualized historical volatility (standard deviation of returns).

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index position
            window: Lookback window in periods

        Returns:
            Annualized historical volatility or None if not enough data
        """
        # Need at least window data points: indices [current_idx - window + 1, current_idx]
        # This requires current_idx - window + 1 >= 0, i.e., current_idx >= window - 1
        if current_idx < window - 1:
            return None

        closes = df.iloc[current_idx - window + 1 : current_idx + 1]["close"].values

        if len(closes) < window:
            return None

        returns = pd.Series(closes).pct_change().dropna()
        daily_vol = returns.std()

        # Annualize: multiply by sqrt(252 trading days)
        annualized_vol = daily_vol * math.sqrt(252)
        return annualized_vol

    def classify(self, spy_df: pd.DataFrame, date: datetime) -> str:
        """
        Classify market regime for a given date.

        Args:
            spy_df: SPY daily OHLCV DataFrame (must include data up to date)
            date: Date to classify regime for

        Returns:
            Regime classification: 'trending_up', 'trending_down', 'volatile', 'calm'
        """
        # Check cache first
        date_key = date.date() if hasattr(date, "date") else date
        if date_key in self._regime_cache:
            return self._regime_cache[date_key]

        # Find index for the given date
        if spy_df.empty:
            logger.warning(f"Empty SPY DataFrame for date {date_key}")
            regime = "calm"  # Default
            self._regime_cache[date_key] = regime
            return regime

        spy_dates = spy_df.index.date if hasattr(spy_df.index[0], "date") else spy_df.index
        valid_indices = [i for i, d in enumerate(spy_dates) if d < date_key]

        if not valid_indices:
            logger.warning(f"No SPY data available for date {date_key}")
            regime = "calm"  # Default
            self._regime_cache[date_key] = regime
            return regime

        current_idx = valid_indices[-1]

        # Calculate metrics
        return_nd = self._calculate_return(spy_df, current_idx, self.lookback_return)
        volatility = self._calculate_volatility(spy_df, current_idx, self.lookback_vol)

        # Classify regime
        regime = "calm"  # Default

        if volatility is not None and volatility > self.volatile_threshold:
            regime = "volatile"
        elif return_nd is not None:
            if return_nd > self.trending_threshold:
                regime = "trending_up"
            elif return_nd < -self.trending_threshold:
                regime = "trending_down"

        return_str = f"{return_nd:.2%}" if return_nd is not None else "N/A"
        vol_str = f"{volatility:.2%}" if volatility is not None else "N/A"
        logger.debug(
            f"Regime for {date_key}: {regime} "
            f"(return_{self.lookback_return}d={return_str}, vol_{self.lookback_vol}d={vol_str})"
        )

        # Cache result
        self._regime_cache[date_key] = regime
        return regime


# Module-level singleton for convenience
_classifier = None


def get_market_regime(spy_df: pd.DataFrame, date: datetime) -> str:
    """
    Get market regime for a given date (module-level convenience function).

    Args:
        spy_df: SPY daily OHLCV DataFrame
        date: Date to classify regime for

    Returns:
        Regime classification: 'trending_up', 'trending_down', 'volatile', 'calm'
    """
    global _classifier
    if _classifier is None:
        _classifier = MarketRegimeClassifier()
    return _classifier.classify(spy_df, date)

