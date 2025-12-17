"""Stock metadata fetching and caching for sector and market cap classification.

This module provides utilities to fetch and cache stock metadata (sector, market cap)
from yfinance API to enrich tweet data with stock context.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import yfinance as yf

from tweet_enricher.config import METADATA_CACHE_FILE

logger = logging.getLogger(__name__)


class StockMetadataCache:
    """
    Persistent cache for stock metadata (sector, market cap bucket).

    Fetches data from yfinance and caches locally to avoid repeated API calls.
    """

    # Simplified sector mapping from GICS sectors
    SECTOR_MAPPING = {
        "Technology": "Technology",
        "Information Technology": "Technology",
        "Communication Services": "Communications",
        "Consumer Cyclical": "Consumer",
        "Consumer Defensive": "Consumer",
        "Consumer Discretionary": "Consumer",
        "Consumer Staples": "Consumer",
        "Financial Services": "Financials",
        "Financials": "Financials",
        "Healthcare": "Healthcare",
        "Health Care": "Healthcare",
        "Industrials": "Industrials",
        "Energy": "Energy",
        "Basic Materials": "Materials",
        "Materials": "Materials",
        "Utilities": "Utilities",
        "Real Estate": "Real Estate",
    }

    # Market cap bucket thresholds (in billions)
    MEGA_CAP_THRESHOLD = 200_000_000_000  # $200B
    LARGE_CAP_THRESHOLD = 10_000_000_000  # $10B
    MID_CAP_THRESHOLD = 2_000_000_000  # $2B

    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize the metadata cache.

        Args:
            cache_file: Path to cache file (default: from config)
        """
        self.cache_file = cache_file or METADATA_CACHE_FILE
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached stock metadata entries")
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
                self._cache = {}
        else:
            logger.info("No existing metadata cache found, starting fresh")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved metadata cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {e}")

    def _classify_market_cap(self, market_cap: Optional[float]) -> str:
        """
        Classify market cap into buckets.

        Args:
            market_cap: Market capitalization in USD

        Returns:
            Bucket classification: 'mega_cap', 'large_cap', 'mid_cap', 'small_cap', or 'unknown'
        """
        if market_cap is None:
            return "unknown"

        if market_cap >= self.MEGA_CAP_THRESHOLD:
            return "mega_cap"
        elif market_cap >= self.LARGE_CAP_THRESHOLD:
            return "large_cap"
        elif market_cap >= self.MID_CAP_THRESHOLD:
            return "mid_cap"
        else:
            return "small_cap"

    def _simplify_sector(self, sector: Optional[str]) -> str:
        """
        Simplify sector name to standard categories.

        Args:
            sector: Raw sector name from yfinance

        Returns:
            Simplified sector name or 'Other'
        """
        if not sector:
            return "Other"

        return self.SECTOR_MAPPING.get(sector, "Other")

    def _fetch_from_yfinance(self, ticker: str) -> Dict:
        """
        Fetch metadata from yfinance API.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with 'sector' and 'market_cap_bucket' keys
        """
        try:
            logger.debug(f"Fetching metadata for {ticker} from yfinance...")
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract sector
            raw_sector = info.get("sector")
            sector = self._simplify_sector(raw_sector)

            # Extract market cap
            market_cap = info.get("marketCap")
            market_cap_bucket = self._classify_market_cap(market_cap)

            cap_str = f"(${market_cap / 1e9:.1f}B)" if market_cap else "(market_cap=None)"
            logger.info(
                f"{ticker}: sector={sector} (raw: {raw_sector}), "
                f"market_cap_bucket={market_cap_bucket} {cap_str}"
            )

            return {"sector": sector, "market_cap_bucket": market_cap_bucket}

        except Exception as e:
            logger.error(f"Failed to fetch metadata for {ticker}: {e}")
            return {"sector": "Other", "market_cap_bucket": "unknown"}

    def get_metadata(self, ticker: str) -> Dict:
        """
        Get stock metadata (sector, market cap bucket).

        Checks cache first, fetches from yfinance if not cached.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with 'sector' and 'market_cap_bucket' keys
        """
        # Check cache first
        if ticker in self._cache:
            logger.debug(f"Using cached metadata for {ticker}")
            return self._cache[ticker]

        # Fetch from API
        metadata = self._fetch_from_yfinance(ticker)

        # Cache result
        self._cache[ticker] = metadata
        self._save_cache()

        return metadata

    def prefetch_metadata(self, tickers: list[str]) -> None:
        """
        Prefetch metadata for multiple tickers.

        Args:
            tickers: List of ticker symbols
        """
        uncached = [t for t in tickers if t not in self._cache]

        if not uncached:
            logger.info("All tickers already cached")
            return

        logger.info(f"Prefetching metadata for {len(uncached)} tickers...")

        for i, ticker in enumerate(uncached, 1):
            logger.info(f"[{i}/{len(uncached)}] Fetching {ticker}...")
            self.get_metadata(ticker)

        logger.info(f"Prefetch complete. Total cached: {len(self._cache)}")

