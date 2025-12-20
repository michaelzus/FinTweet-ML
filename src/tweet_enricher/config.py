"""Configuration constants for tweet_enricher package."""

import os
from pathlib import Path

import pytz

# Timezone
ET = pytz.timezone("US/Eastern")

# Data directories
DAILY_DATA_DIR = Path("data/daily")
INTRADAY_CACHE_DIR = Path("data/intraday")

# Market hours definition (all in ET, minutes since midnight)
MARKET_OPEN = 9 * 60 + 30  # 9:30 AM
MARKET_CLOSE = 16 * 60  # 4:00 PM
PREMARKET_START = 4 * 60  # 4:00 AM
AFTERHOURS_END = 20 * 60  # 8:00 PM

# Tickers that consistently fail to fetch from IBKR API
# These include: Class A/B shares, tickers with naming issues, and others that timeout
EXCLUDED_TICKERS = {
    # Class A/B shares and special share classes
    "BRK-B",
    "BF-B",
    "BF-A",
    "CWEN-A",
    "HEI-A",
    "LEN-B",
    "UHAL-B",
    # Tickers with naming/symbol issues
    "LNW",
    "DNB",
    "CCCS",
    "COOP",
    "IPG",
    "K",
    "INFA",
    # Tickers that consistently timeout (added from recent errors)
    "WDAY",
    "NDAQ",
    "FHN",
    "LKQ",
    "ABT",
    "ED",
    "COO",
    "UGI",
    "SYF",
}

# Default IB connection settings
DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 7497
DEFAULT_CLIENT_ID = 1

# Default fetch settings
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_DELAY = 2.0

# Intraday fetch settings
INTRADAY_TOTAL_DAYS = 200       # Total days of intraday history to fetch
INTRADAY_FETCH_DELAY = 2.0      # Seconds between symbol requests

# Market regime classification thresholds
REGIME_TRENDING_THRESHOLD = 0.02  # ±2% weekly return = trending
REGIME_VOLATILE_THRESHOLD = 0.18  # 18% annualized volatility = volatile
REGIME_LOOKBACK_RETURN = 5  # 5 trading days (1 week) for trend detection
REGIME_LOOKBACK_VOL = 5  # 5 trading days for volatility calculation

# Stock metadata cache
METADATA_CACHE_FILE = Path("data/stock_metadata.json")

# TwitterAPI.io settings
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
TWITTER_DB_PATH = Path("data/tweets.db")
TWITTER_RATE_LIMIT_DELAY = 0.2  # seconds between requests (paid tier - very fast!)
TWITTER_TWEETS_PER_REQUEST = 100  # max tweets per API call (paid tier supports more)
TWITTER_ACCOUNTS = [
    "StockMKTNewz",
    "wallstengine",
    "amitisinvesting",
    "AIStockSavvy",
    "fiscal_ai",
    "EconomyApp",
]
