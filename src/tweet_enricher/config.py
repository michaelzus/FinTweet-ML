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

# Market regime classification thresholds
REGIME_TRENDING_THRESHOLD = 0.05  # ±5%
REGIME_VOLATILE_THRESHOLD = 0.20  # 20% volatility
REGIME_LOOKBACK_RETURN = 20  # days
REGIME_LOOKBACK_VOL = 7  # days

# Stock metadata cache
METADATA_CACHE_FILE = Path("data/stock_metadata.json")

# TwitterAPI.io settings
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
TWITTER_DB_PATH = Path("data/tweets.db")
TWITTER_RATE_LIMIT_DELAY = 5.0  # seconds between requests (free tier limit)
TWITTER_ACCOUNTS = [
    "StockMKTNewz",
    "wallstengine",
    "amitisinvesting",
    "AIStockSavvy",
    "fiscal_ai",
    "EconomyApp",
]

