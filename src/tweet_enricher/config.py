"""Configuration constants for tweet_enricher package."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# Timezone - using modern zoneinfo (Python 3.9+) for better DST handling
# Note: For timezone utilities, prefer importing from tweet_enricher.utils.timezone
ET = ZoneInfo("America/New_York")

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
    # Indices, crypto, and non-stock symbols (not available as stocks in IB)
    "DJI",  # Dow Jones Industrial Average index
    "DXY",  # US Dollar Index
    "DOGE",  # Dogecoin crypto
    "BTC",  # Bitcoin
    "ETH",  # Ethereum
    "XRP",  # Ripple
    "SOL",  # Solana
    "DSYNC",  # Not a valid stock
    "SPX",  # S&P 500 index (use SPY instead)
    "NDX",  # Nasdaq 100 index (use QQQ instead)
    "VIX",  # Volatility index
    "TNX",  # 10-Year Treasury yield
    # Wrong symbol format (period instead of dash)
    "BRK",  # Should be BRK-B
    "BRK.B",  # Should be BRK-B
    "BF",  # Should be BF-B
    "BF.B",  # Should be BF-B
    # Crypto/AI tokens (not stocks)
    "AITECH",
    "BLENDR",
    "DEAI",
    "HASHAI",
    "OPSEC",
    "WIF",
    "SAI",
    "GPU",  # Crypto token
    # Delisted/acquired/invalid
    "FFIE",  # Faraday Future - delisted
    "GBLE",  # Invalid
    "PARA",  # Paramount - acquired
    "SQ",  # Block Inc - changed to XYZ
    "WBA",  # Walgreens - went private
    "ZK",  # Chinese company - BEST queries not supported
    # Invalid/unrecognized symbols from IB validation (batch 2024-12-22)
    # Typos and wrong symbols
    "A285B", "AMAZ", "AMZ", "AMNZ", "ANZN", "APPL", "FORD", "GOOGLE", "LENOVO", "LVMH",
    "NIKE", "ORACLE", "TSMC", "ZOOM", "HERMES", "MSFTON", "NLFX",
    # Crypto/tokens (not stocks)
    "ADA", "AI16Z", "AIABSC", "AIBSTRACTA", "AIC", "AIW", "AIX", "BASE", "BILLIONS",
    "BLUB", "BSKY", "BTCM", "CUDIS", "DOLLARS", "ECONOMY", "FARTCOIN", "GOAT", "GREED",
    "HAEDAL", "HTTPS", "KMNO", "LINEA", "LOFI", "OMEGA", "SLEEP", "SOLQ", "SOLQ.U",
    "SQUIRT", "STARTUP", "STETH", "SWARMS", "TRUMP", "VIRTUAL",
    # European/foreign tickers (not on US exchanges)
    "ADYEN", "AENA", "ASMI", "ATZ.TO", "BEIJB", "BRBY", "CARLB", "DGE", "DHER", "DSV",
    "HAYPP", "HEXAB", "KER", "LDO", "LDO.MI", "LIFCO", "MBG", "MC.PA", "MDA.CA", "NOVO",
    "RHMG", "RHMG.DE", "RMS", "RMS.PA", "UMG",
    # Delisted/invalid US tickers
    "ADVM", "AGT", "AKRO", "ALCC", "ALCN", "ALTR", "AMI", "AMRK", "ANGUS", "ANSS", "AOF",
    "ARAMCO", "ARCH", "ASTR", "ATD", "ATMV", "ATNF", "ATUS", "ATZ", "AUR.US", "AVDX",
    "AVGP", "AXO", "AZEK", "AZPN", "AZUL", "BATS", "BECN", "BERY", "BFB", "BIGC", "BLDE",
    "BLSY", "BLUE", "BOLSA", "BPMC", "BRDG", "BREA", "BRK.A", "BRPHF", "BRY", "BSGM",
    "BVLOS", "BYDD", "BYON", "CBOW", "CCCM", "CEIX", "CENC", "CEO", "CFL", "CGLW",
    "CHKEL", "CHKEZ", "CHX", "CJET", "CLFT", "CRGX", "CRWW", "CSGO", "CSU", "CSWI",
    "CWRV", "DALN", "DCXM", "DFS", "DNUE", "DOOO", "DSK", "DTC", "DULO", "DYK", "DYNX",
    "EDR", "ER", "ERJ", "ES_F", "ETNB", "ETWO", "EXAI", "FAAS", "FI", "FL", "FLSR",
    "FUBP", "GA", "GB", "GB.WS", "GCI", "GMRAIRPORT", "GMS", "GOEV", "GOGL", "GOTO",
    "GPS", "GSRT", "HBI", "HCP", "HEES", "HEI.A", "HEIA", "HES", "HIM", "HMST", "HO",
    "HPW", "HUSA", "IAA", "IBRK", "ICLK", "INMS", "IPA", "IRBN", "IRBT", "ITCI", "JDD",
    "JNPR", "JNVR", "JWN", "KDKRW", "KDLY", "KIND", "KLA", "KLG", "KLK", "KNW", "LANC",
    "LIFX", "LPTX", "LURN", "MCG", "MCSA", "MDA", "MDEP", "ME", "MEIP", "MES", "MIU",
    "MKX", "MLNK", "MNQ", "MOSTAFA", "MTAL", "MTSR", "MULN", "MWT", "MYM", "NARI", "NBI",
    "NBUS", "NEP", "NEW", "NEWS", "NKLA", "NOVA", "NPWR.WS", "NQ", "NQ_F", "NUH", "NWTN",
    "NYDOY", "OBKR", "OCTO", "ODP", "OLO", "ONVO", "OTRK", "PASS", "PBPB", "PCSO",
    "PEAK", "PKSY", "PLYA", "PME", "PONY.AI", "PPONY", "PROR", "PRPX", "PRTG", "PSY",
    "PTLR", "PXYS", "QBTI", "QLGN", "QSG", "QTBS", "RDFN", "RDT", "RGIT", "RGLS", "RI",
    "RIOR", "RKBL", "RKLV", "RPI", "RRDW", "SAGE", "SATX", "SAVE", "SCS", "SHLL", "SKH",
    "SKX", "SLHX", "SNPSP", "SOF", "SOI", "SOND", "SOX", "SPLG", "SPNS", "SPR", "SPTN",
    "SRTAW", "SRX", "STR", "SUPM", "SWI", "SWTX", "TALA", "TBLX", "TEMP", "TEV", "TGI",
    "THRD", "THTX", "TMS", "TPIC", "TRML", "TTW", "TVK", "UBM", "UBRT", "ULMN", "UMB",
    "UMDY", "UNHO", "USM", "VBTX", "VDA", "VERV", "VI", "VITB", "VMEO", "VRNA", "VRNT",
    "VRV", "VSET", "WAV", "WEL", "WHIN", "WHRL", "WHSP", "WMY", "WNS", "WULD", "X",
    "XIACF", "XM", "XOF", "YLDS", "YM", "YNDX", "YY", "ZI",
}

# Default IB connection settings
DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 7497
DEFAULT_CLIENT_ID = 1

# Default fetch settings
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_DELAY = 2.0

# Intraday fetch settings
INTRADAY_TOTAL_DAYS = 200  # Total days of intraday history to fetch
INTRADAY_FETCH_DELAY = 2.0  # Seconds between symbol requests

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
