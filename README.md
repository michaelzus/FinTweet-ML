# FinTweet-ML

ML pipeline for enriching financial tweets with market data and sentiment classification.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FinBERT](https://img.shields.io/badge/Model-FinBERT-green.svg)](https://huggingface.co/ProsusAI/finbert)
[![IBKR](https://img.shields.io/badge/IBKR-TWS%2FGateway-orange.svg)](https://www.interactivebrokers.com/)

---

## 🎯 What This Project Does

**FinTweet-ML** is a comprehensive ML data pipeline that:

1. 📊 **Fetches historical OHLCV data** from Interactive Brokers for S&P 500 / Russell 1000
2. 🔍 **Filters tickers by volume** to identify liquid, tradeable stocks  
3. 💬 **Processes Discord financial messages** into structured, categorized CSV datasets
4. 🕐 **Converts timezones** (Jerusalem → US Eastern) for market alignment
5. 📈 **Outputs analysis-ready data** for ML, trading, or research

**Use cases:** Event studies, sentiment analysis, trading strategies, market research, ML training data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Interactive Brokers TWS or Gateway (for historical data fetching)

### Installation

```bash
# Clone or navigate to project
cd FinTweet-ML

# Install dependencies
pip install -e .
```

### Complete Pipeline Example

```bash
# 1. Fetch historical data for S&P 500
python fetch_historical_data.py --sp500 --duration "1 Y" --output-dir data

# 2. Filter tickers by average volume (1M+)
python filter_by_volume.py --data-dir data --min-volume 1000000 --output high_volume.csv

# 3. Convert Discord messages to structured CSV
python discord_to_csv.py \
    -i discrod_data/AI_INVEST_ISRAEL.txt \
    -o output/discord_messages.csv \
    -f high_volume.csv
```

**Result:** Clean, filtered, categorized financial messages with market data integration!

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Installation](#-installation)
- [Pipeline Components](#-pipeline-components)
  - [1. Historical Data Fetcher](#1-historical-data-fetcher)
  - [2. Volume Filter](#2-volume-filter)
  - [3. Discord Converter](#3-discord-converter)
- [Discord Converter Details](#-discord-converter-details)
  - [Categories](#-message-categories)
  - [Output Format](#-output-format)
  - [Features](#-features)
  - [Architecture](#-architecture)
  - [Extending](#-extending)
- [Complete Workflows](#-complete-workflows)
- [Use Cases](#-use-cases)
- [Timezone Conversion](#-timezone-conversion)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)

---

## 🌟 Project Overview

### The Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    FINANCIAL DATA PIPELINE                   │
└─────────────────────────────────────────────────────────────┘

STEP 1: Fetch Historical Data
┌──────────────────────────┐
│ Interactive Brokers API  │
│  • S&P 500 tickers       │
│  • Russell 1000 tickers  │
│  • Custom ticker lists   │
└────────────┬─────────────┘
             │
             ↓ (fetch_historical_data.py)
┌────────────────────────────┐
│  Historical OHLCV Data     │
│  • data/AAPL.csv           │
│  • data/MSFT.csv           │
│  • data/GOOGL.csv          │
│  • ... (1000+ files)       │
└────────────┬───────────────┘
             │
             ↓ STEP 2: Filter by Volume
             │ (filter_by_volume.py)
┌────────────────────────────┐
│  Filtered Ticker List      │
│  high_volume.csv           │
│  • Only liquid stocks      │
│  • 1M+ avg daily volume    │
└────────────┬───────────────┘
             │
             ↓ STEP 3: Process Messages
             │ (discord_to_csv.py)
┌─────────────────────────────────────┐
│  Discord Channel Export             │
│  • News, tweets, alerts             │
│  • Ticker mentions                  │
│  • Timestamps (Jerusalem time)      │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  FINAL OUTPUT                       │
│  discord_messages.csv               │
│  • Categorized messages             │
│  • Filtered tickers                 │
│  • US Eastern timestamps            │
│  • Clean, analysis-ready            │
└─────────────────────────────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.8+
- Interactive Brokers TWS or Gateway (running)
- API enabled in TWS/Gateway settings

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas` - Data processing
- `ib_async` - Interactive Brokers async API
- `pytz` - Timezone handling
- `lxml` - HTML/XML parsing (for Wikipedia ticker lists)

---

## 🔧 Pipeline Components

### 1. Historical Data Fetcher

**Script:** `fetch_historical_data.py`

**Purpose:** Fetch OHLCV historical data from Interactive Brokers for any ticker list.

#### Features
- ✅ Fetch S&P 500 tickers automatically (from Wikipedia)
- ✅ Fetch Russell 1000 tickers automatically
- ✅ Custom ticker lists
- ✅ Configurable duration (1 Y, 6 M, 30 D, etc.)
- ✅ Configurable bar size (1 day, 1 hour, 5 mins, etc.)
- ✅ Batch processing with rate limiting
- ✅ Async I/O for performance
- ✅ Automatic retry and error handling

#### Usage

```bash
# Fetch S&P 500 (last 1 year, daily bars)
python fetch_historical_data.py --sp500

# Fetch Russell 1000 (last 6 months)
python fetch_historical_data.py --russell1000 --duration "6 M"

# Fetch both S&P 500 + Russell 1000 combined
python fetch_historical_data.py --all

# Custom ticker list
python fetch_historical_data.py --symbols AAPL MSFT GOOGL TSLA

# Custom configuration
python fetch_historical_data.py --sp500 \
    --duration "2 Y" \
    --bar-size "1 hour" \
    --output-dir my_data \
    --batch-size 100 \
    --batch-delay 3.0
```

#### CLI Options

```
Required (one of):
  --symbols [TICKER ...]    Custom list of tickers (e.g., AAPL MSFT GOOGL)
  --sp500                   Fetch all S&P 500 stocks
  --russell1000            Fetch all Russell 1000 stocks
  --all                    Fetch both S&P 500 + Russell 1000 (combined, deduplicated)

Optional:
  --duration DURATION      Duration string (default: "1 Y")
                          Examples: "1 Y", "6 M", "30 D", "1 W"
  
  --bar-size SIZE         Bar size (default: "1 day")
                          Examples: "1 day", "1 hour", "30 mins", "5 mins", "1 min"
  
  --output-dir DIR        Output directory (default: "data")
  --batch-size N          Symbols per batch (default: 200)
  --batch-delay N         Delay between batches in seconds (default: 2.0)
  
  --host HOST            TWS/Gateway host (default: "127.0.0.1")
  --port PORT            TWS/Gateway port (default: 7497)
                         Use 4002 for Gateway
  --client-id N          Client ID (default: 1)
  
  --exchange EXCHANGE    Exchange (default: "SMART")
  --currency CURRENCY    Currency (default: "USD")
```

#### Output

Creates individual CSV files in output directory:

```
data/
├── AAPL.csv
├── MSFT.csv
├── GOOGL.csv
└── ... (one file per ticker)
```

Each CSV contains:
- `date` - Date/timestamp
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume
- `average` - VWAP (if available)
- `barCount` - Number of trades (if available)

---

### 2. Volume Filter

**Script:** `filter_by_volume.py`

**Purpose:** Filter tickers by average daily trading volume to identify liquid stocks.

#### Features
- ✅ Analyzes all CSV files in data directory
- ✅ Calculates average daily volume
- ✅ Filters by minimum threshold
- ✅ Outputs ticker list to CSV
- ✅ Useful for ensuring tradeable stocks only

#### Usage

```bash
# Filter with 1M minimum volume (default)
python filter_by_volume.py --data-dir data --output high_volume.csv

# Custom minimum volume (5M)
python filter_by_volume.py \
    --data-dir data \
    --min-volume 5000000 \
    --output very_high_volume.csv

# Just analyze, don't save
python filter_by_volume.py --data-dir data --min-volume 1000000
```

#### CLI Options

```
--data-dir DIR        Directory with CSV files (default: "data")
--min-volume N        Minimum avg daily volume (default: 1,000,000)
--output FILE         Output CSV file (optional)
```

#### Output

Creates CSV with filtered tickers:

```csv
symbol
AAPL
MSFT
GOOGL
TSLA
NVDA
...
```

**This file is then used by `discord_to_csv.py` to filter messages!**

---

### 3. Discord Converter

**Script:** `discord_to_csv.py`

**Purpose:** Convert Discord channel exports to structured, categorized CSV datasets.

#### Features
- ✅ **Ticker Extraction** - Finds all `$TICKER` symbols
- ✅ **12 Categories** - Earnings, M&A, Breaking News, etc.
- ✅ **Text Cleaning** - Removes noise, URLs, escape characters
- ✅ **Deduplication** - Removes exact duplicates
- ✅ **Volume Filtering** - Uses ticker whitelist from filter_by_volume.py
- ✅ **Timezone Conversion** - Jerusalem → US Eastern Time (handles DST)
- ✅ **Quality Controls** - Minimum text length, validation
- ✅ **Full CLI** - Flexible command-line interface

#### Usage

```bash
# Basic conversion
python discord_to_csv.py -i discord_data.txt -o output.csv

# With ticker filter (recommended)
python discord_to_csv.py \
    -i discrod_data/AI_INVEST_ISRAEL.txt \
    -o output/discord_messages.csv \
    -f high_volume.csv

# Custom options
python discord_to_csv.py \
    -i discord_data.txt \
    -o output.csv \
    --min-length 100 \
    --no-dedup \
    -q
```

#### CLI Options

```
Required:
  -i, --input INPUT      Input Discord export file
  -o, --output OUTPUT    Output CSV file

Optional:
  -f, --filter FILTER    Ticker filter CSV (from filter_by_volume.py)
  --no-dedup            Disable duplicate removal
  --min-length N        Minimum text length (default: 60)
  -q, --quiet           Suppress progress output
  --version             Show version
  -h, --help            Show help
```

---

## 📊 Discord Converter Details

### 📑 Message Categories

The Discord converter classifies messages into **12 financial categories** based on keyword matching:

#### 1. **Earnings** 📈
Financial reports, quarterly results, EPS announcements

**Keywords:** `earnings`, `quarter`, `Q1-Q4`, `EPS`, `fiscal`, `beats/misses estimate`, `earnings report`, `earnings call`, `net income`, `gross margin`, `operating income`, `ebitda`

**Example:** *"AAPL reports Q3 earnings beat expectations with EPS of $1.52"*

---

#### 2. **Breaking News** 🚨
Major, unexpected, market-moving events

**Keywords:** `breaking`, `alert`, `urgent`, `crisis`, `halt`, `unprecedented`, `just in`, `developing`, `shutdown`, `surge`, `plunge`

**Example:** *"Breaking: Fed announces emergency 50bp rate cut"*

---

#### 3. **Mergers & Acquisitions** 🤝
Buyouts, mergers, strategic deals

**Keywords:** `merger`, `acquisition`, `acquire`, `buyout`, `takeover`, `consolidation`, `joint venture`, `spinoff`, `purchase`

**Example:** *"MSFT to acquire ATVI for $69B in all-cash deal"*

---

#### 4. **Guidance & Forecasts** 🎯
Forward-looking statements, analyst ratings, price targets

**Keywords:** `guidance`, `outlook`, `forecast`, `price target`, `upgrade`, `downgrade`, `analyst`, `rating`, `projection`, `expects`, `initiates coverage`

**Example:** *"Morgan Stanley upgrades MSFT to Overweight with $450 price target"*

---

#### 5. **Regulatory & Legal** ⚖️
Lawsuits, government actions, policy changes

**Keywords:** `lawsuit`, `SEC`, `FDA`, `regulation`, `investigation`, `fine`, `compliance`, `antitrust`, `probe`, `settlement`, `ruling`

**Example:** *"DOJ opens antitrust investigation into GOOGL search practices"*

---

#### 6. **Product Launch** 🚀
New products, features, services, delays, cancellations

**Keywords:** `launches`, `unveils`, `new product`, `delayed`, `cancelled`, `released`, `rolls out`, `introducing`, `debut`, `coming soon`

**Example:** *"AAPL unveils iPhone 16 with AI-powered features, ships Sept 20"*

---

#### 7. **Partnerships & Deals** 🤝
Strategic partnerships, collaborations, contracts

**Keywords:** `partnership`, `collaboration`, `deal with`, `signs contract`, `agreement with`, `teams up`, `alliance`, `works with`

**Example:** *"AMZN partners with WMT to expand delivery network in rural areas"*

---

#### 8. **Market Data** 📊
Stock prices, market cap, valuations, historical data

**Keywords:** `market cap`, `stock price`, `valuation`, `trading at`, `trillion`, `shares`, `all-time high`, `historical`

**Example:** *"NVDA hits $4.5T market cap, surpassing AAPL as world's largest company"*

---

#### 9. **Company Strategy** 🎯
Business operations, expansion, automation, restructuring

**Keywords:** `plans to`, `expansion`, `automation`, `restructuring`, `robots`, `workforce`, `transformation`, `initiative`, `investing in`

**Example:** *"AMZN plans warehouse automation to replace 600K jobs by 2027"*

---

#### 10. **Company Metrics** 📈
Revenue, users, growth metrics, KPIs

**Keywords:** `million users`, `subscribers`, `revenue of`, `grew by`, `customers`, `active users`, `monthly active`, `user growth`

**Example:** *"META reaches 3 billion monthly active users, up 8% YoY"*

---

#### 11. **Personnel Changes** 👔
Executive moves, hirings, layoffs

**Keywords:** `CEO`, `hired`, `resigns`, `layoffs`, `appointed`, `chief executive`, `steps down`, `departing`, `replaces`

**Example:** *"TSLA CFO Zachary Kirkhorn steps down after 13 years"*

---

#### 12. **Other** 📝
Everything else that doesn't fit above categories
- Default category for miscellaneous content

---

### 📤 Output Format

#### CSV Structure

```csv
timestamp,author,ticker,tweet_url,category,text
2024-10-21 02:00:00,User,NVDA,https://twitter.com/...,Earnings,"Nvidia reports Q3 earnings..."
2024-10-21 09:30:00,User,AAPL,https://twitter.com/...,Product Launch,"Apple unveils new iPhone..."
```

#### Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | US Eastern Time (YYYY-MM-DD HH:MM:SS) |
| `author` | string | Discord username |
| `ticker` | string | Stock ticker symbol (AAPL, MSFT, etc.) |
| `tweet_url` | string | Extracted Twitter/X URL (if present) |
| `category` | string | One of 12 categories |
| `text` | string | Cleaned message text (single line) |

#### Features
- **One row per ticker** - Messages with multiple tickers create multiple rows
- **Clean text** - No newlines, URLs removed, special chars handled
- **US Eastern timestamps** - Aligned with market hours
- **No duplicates** - Same (timestamp + ticker + text) only appears once

---

### ✨ Features

#### 1. Text Cleaning

Automatically removes:
- `{Embed}` and `{Attachments}` markers
- URLs (extracted separately to `tweet_url` field)
- `TweetShift` watermarks
- Emoji indicators (`📷1`, etc.)
- Escape characters (`\-`, `\(`, `\)`, etc.)
- Newlines (converted to spaces for CSV compatibility)

#### 2. Quality Filtering

- **Minimum length**: Filters out texts <60 characters (configurable)
- **Ticker required**: Only includes messages mentioning valid tickers
- **Duplicate removal**: Removes exact duplicates (timestamp + ticker + text)

#### 3. Ticker Extraction

- Finds all `$TICKER` mentions in text
- Creates separate row for each ticker
- Uppercase normalization
- Validates against filter list if provided

#### 4. Smart Categorization

- Keyword-based classification
- 12 predefined categories
- Priority matching (checks categories in order)
- Falls back to "Other" if no match

---

### 🏗️ Architecture

#### Object-Oriented Design

**Classes:**
```
MessageCategorizer      # Categorizes messages by keywords
MessageProcessor        # Cleans text, extracts tickers
DiscordParser          # Parses Discord export format
CSVWriter              # Writes CSV with filtering/dedup
TickerFilter           # Loads ticker whitelist
DiscordToCSVConverter  # Main orchestrator
```

**Design Principles:**
- **Single Responsibility** - Each class has one job
- **Dependency Injection** - Components are injected
- **Testability** - Easy to unit test
- **Extensibility** - Easy to add features
- **Type Hints** - Full type annotations

#### Data Flow

```
Discord Export (.txt)
        ↓
[DiscordParser]
  • Parse timestamp, author, text
  • Split into messages
        ↓
[MessageProcessor]
  • Extract tickers ($AAPL)
  • Extract tweet URLs
  • Clean text (remove noise)
  • Validate length
        ↓
[MessageCategorizer]
  • Match keywords
  • Assign category
        ↓
[TickerFilter] (optional)
  • Check against whitelist
        ↓
[CSVWriter]
  • Remove duplicates
  • Convert timestamp (Jerusalem → US Eastern)
  • Write to CSV
        ↓
Structured CSV Output
```

---

### 🔨 Extending

#### Add New Category

```python
# In MessageCategorizer class (discord_to_csv.py)
CATEGORIES = {
    # ... existing categories ...
    'Your Category': [
        'keyword1', 'keyword2', 'keyword3'
    ]
}
```

#### Custom Text Cleaning

```python
class CustomProcessor(MessageProcessor):
    def clean_text(self, text: str) -> str:
        text = super().clean_text(text)
        # Add your custom cleaning
        text = text.replace('noise', '')
        return text
```

#### Add Output Column

```python
# In CSVWriter class
FIELDNAMES = ['timestamp', 'author', 'ticker', 'tweet_url', 
              'category', 'text', 'your_new_field']

# Modify writer.writerow() to include new field
```

#### Use as Library

```python
from discord_to_csv import DiscordToCSVConverter
from pathlib import Path

# Create converter
converter = DiscordToCSVConverter(
    min_text_length=60,
    deduplicate=True
)

# Convert
stats = converter.convert(
    input_file=Path('input.txt'),
    output_file=Path('output.csv'),
    ticker_filter_file=Path('filter.csv'),
    verbose=True
)

print(f"Wrote {stats['written']} messages")
```

---

## 📁 Project Structure

```
FinTweet-ML/
├── src/
│   ├── tweet_enricher/           # Core enrichment pipeline
│   │   ├── core/                 # Business logic (enricher, indicators)
│   │   ├── data/                 # Data fetching (IBKR, cache)
│   │   ├── parsers/              # Input parsing (Discord)
│   │   ├── io/                   # File I/O (CSV, Feather)
│   │   ├── market/               # Market session utilities
│   │   └── twitter/              # Twitter API client
│   └── tweet_classifier/         # FinBERT sentiment classifier
│
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md           # System architecture
│   ├── ENRICHMENT_DEMO.md        # Enrichment examples
│   └── TWITTER_FETCH.md          # Twitter API docs
│
├── data/                         # Data cache (gitignored)
│   ├── daily/                    # Daily OHLCV cache
│   └── intraday/                 # Intraday cache
│
├── pyproject.toml                # Project configuration
├── ROADMAP.md                    # Future improvements
└── README.md                     # This file
```

---

## 🔄 Complete Workflows

### Workflow 1: Build Ticker Universe (First Time Setup)

```bash
# Step 1: Start TWS/Gateway
# Make sure Interactive Brokers TWS or Gateway is running

# Step 2: Fetch historical data for S&P 500
python fetch_historical_data.py --sp500 --duration "1 Y"

# Output: ~500 CSV files in data/ directory
# Time: ~30-60 minutes (depending on rate limits)

# Step 3: Filter by volume to get liquid stocks
python filter_by_volume.py \
    --data-dir data \
    --min-volume 1000000 \
    --output high_volume.csv

# Output: high_volume.csv with ~288 tickers
```

### Workflow 2: Process Discord Messages

```bash
# Use filtered ticker list to process Discord data
python discord_to_csv.py \
    -i discrod_data/AI_INVEST_ISRAEL.txt \
    -o output/discord_messages.csv \
    -f high_volume.csv

# Output: Structured CSV with ~2,836 categorized messages
```

### Workflow 3: Update Historical Data (Periodic)

```bash
# Update data for existing tickers
python fetch_historical_data.py --sp500 --duration "1 M"

# Re-filter if needed
python filter_by_volume.py --data-dir data --min-volume 1000000 --output high_volume.csv
```

### Workflow 4: Custom Ticker Set

```bash
# Fetch specific tickers
python fetch_historical_data.py --symbols AAPL MSFT GOOGL TSLA NVDA AMD

# Create custom_tickers.csv
echo "symbol\nAAPL\nMSFT\nGOOGL\nTSLA\nNVDA\nAMD" > custom_tickers.csv

# Process Discord with custom list
python discord_to_csv.py \
    -i discord_data.txt \
    -o output.csv \
    -f custom_tickers.csv
```

---

## 🎯 Use Cases

### 1. Event Study Analysis

**Goal:** Analyze stock price reactions to news events

```bash
# 1. Get historical data
python fetch_historical_data.py --sp500

# 2. Filter liquid stocks
python filter_by_volume.py --data-dir data --output liquid.csv

# 3. Get categorized news events
python discord_to_csv.py -i discord.txt -o events.csv -f liquid.csv

# 4. Analyze in Python
python
>>> import pandas as pd
>>> events = pd.read_csv('events.csv')
>>> # Match event timestamps to price movements
>>> # Calculate returns (1h, 1d, 1w after event)
>>> # Analyze which categories have most impact
```

### 2. Sentiment Analysis Training

**Goal:** Build ML model for financial sentiment

```bash
# Get labeled dataset
python discord_to_csv.py -i discord.txt -o training_data.csv -f high_volume.csv

# Output: ~2,800 samples ready for:
# - FinBERT fine-tuning
# - Custom sentiment model
# - Expected accuracy: 70-80%
```

### 3. Trading Strategy Backtesting

**Goal:** Test sentiment-based trading strategies

```bash
# 1. Get data
python fetch_historical_data.py --sp500 --duration "2 Y"
python filter_by_volume.py --data-dir data --output tradeable.csv
python discord_to_csv.py -i discord.txt -o signals.csv -f tradeable.csv

# 2. Backtest
# - Filter to specific categories (e.g., "Earnings")
# - Use as entry signals
# - Calculate returns and Sharpe ratio
```

### 4. Market Intelligence Dashboard

**Goal:** Track mentions and trends

```bash
# Regular updates
python discord_to_csv.py -i new_messages.txt -o latest.csv -f high_volume.csv

# Analyze:
# - Ticker mention frequency
# - Category trends over time
# - Author influence
# - Sector-wide themes
```

---

## 🕐 Timezone Conversion

**Important:** Discord timestamps are automatically converted from **Jerusalem time** to **US Eastern Time**.

### Why This Matters
- 📊 Market data APIs use US Eastern Time
- 🔗 Easy correlation with price movements
- ⏰ DST handled automatically
- 📈 Timestamps align with US trading hours

### Time Difference
- **7 hours** - Most of the year (Jerusalem ahead)
- **6 hours** - Brief transition period (March/April)

### Market Hours in Jerusalem Time

| Period | US Eastern | Jerusalem |
|--------|------------|-----------|
| Pre-market | 7:00-9:30 AM | 14:00-16:30 |
| Regular | 9:30 AM-4:00 PM | 16:30-23:00 |
| After-hours | 4:00-8:00 PM | 23:00-02:00 (next day) |

### Timezone Details

**Jerusalem Time:**
- Standard (IST): UTC+2 (winter)
- Daylight (IDT): UTC+3 (summer)
- DST Period: Late March to late October

**US Eastern Time:**
- Standard (EST): UTC-5 (winter)
- Daylight (EDT): UTC-4 (summer)
- DST Period: Early March to early November

### Examples

```
Winter (Both Standard Time):
Jerusalem:  15/01/2024 10:00 IST  →  US Eastern: 2024-01-15 03:00:00 EST

Summer (Both Daylight Time):
Jerusalem:  15/07/2024 10:00 IDT  →  US Eastern: 2024-07-15 03:00:00 EDT

Market Open:
Jerusalem:  15/11/2024 16:30 IST  →  US Eastern: 2024-11-15 09:30:00 EST ✅
```

### Test It

```bash
python test_timezone.py
```

---

## 🔧 Troubleshooting

### IBKR Connection Issues

**Error:** "Connection refused" or "Could not connect to IB"

**Solution:**
1. Ensure TWS or Gateway is running
2. Enable API in settings:
   - TWS: Configure → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Uncheck "Read-Only API"
3. Check port:
   - TWS default: 7497
   - Gateway default: 4002
4. Try: `--port 4002` if using Gateway

### No Data Returned

**Error:** "No data fetched for any symbols"

**Solutions:**
```bash
# Check connection
python -c "from ib_async import IB; import asyncio; asyncio.run(IB().connectAsync('127.0.0.1', 7497))"

# Try single ticker first
python fetch_historical_data.py --symbols AAPL

# Reduce batch size
python fetch_historical_data.py --sp500 --batch-size 50 --batch-delay 5.0
```

### Volume Filter Returns Nothing

**Error:** "No tickers found with volume >= X"

**Solutions:**
```bash
# Check data directory
ls -la data/*.csv | wc -l

# Lower threshold
python filter_by_volume.py --min-volume 100000

# Check a specific file
python -c "import pandas as pd; df = pd.read_csv('data/AAPL.csv'); print(df['volume'].mean())"
```

### Discord Converter: No Messages

**Solutions:**
```bash
# Try without filter
python discord_to_csv.py -i input.txt -o output.csv

# Lower minimum length
python discord_to_csv.py -i input.txt -o output.csv --min-length 30

# Check Discord file format
head -20 discrod_data/AI_INVEST_ISRAEL.txt
```

### Encoding Issues

```bash
# Check file encoding
file discord_export.txt

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
python discord_to_csv.py -i input.txt -o output.csv
```

---

## 📊 Performance & Statistics

### Fetch Historical Data
- **Speed:** ~5-10 tickers/second (IBKR rate limits)
- **S&P 500:** ~30-60 minutes for 1 year daily data
- **Russell 1000:** ~60-120 minutes
- **Data Size:** ~1-50 KB per ticker (depends on duration/bar size)

### Volume Filter
- **Speed:** ~100-200 files/second
- **1000 tickers:** < 30 seconds

### Discord Converter
- **Speed:** ~10,000 messages/second
- **7,800 messages:** < 1 second
- **Output:** ~500 bytes/message average

### Example Output Statistics

```
Reading Discord data from: discrod_data/AI_INVEST_ISRAEL.txt
Found 7834 messages
Loaded 288 tickers from filter list: high_volume.csv
Writing to CSV: output/discord_messages.csv

Conversion complete!
Filtered out 4531 messages (tickers not in filter list)
Removed 467 duplicate messages
Written 2836 unique messages to CSV
```

**Result:**
- 2,836 clean messages
- 236 unique tickers
- 12 categories
- US Eastern timestamps
- Ready for analysis!

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources
- More categories or better keyword matching
- Real-time streaming
- Web interface
- ML integration
- Sentiment analysis features
- Performance optimizations

---

## 📄 License

MIT License - Free to use and modify

---

## 🙏 Acknowledgments

- Interactive Brokers API (`ib_async`)
- Wikipedia (S&P 500 and Russell 1000 ticker lists)
- `pytz` (timezone handling)
- `pandas` (data processing)

---

## 📞 Support

For questions or issues:
1. Check this README
2. Review `--help` for each script
3. Check troubleshooting section
4. Test with small datasets first
5. Review individual module documentation in code

---

## 🎉 Summary

**What This Pipeline Provides:**

✅ **Complete Data Collection**
- Historical OHLCV from IBKR (S&P 500, Russell 1000)
- Discord financial messages with news and tweets

✅ **Data Quality**
- Volume-filtered tickers (liquid stocks only)
- Categorized messages (12 intelligent categories)
- Timezone-corrected timestamps (market-aligned)
- Cleaned, deduplicated, validated data

✅ **Production Ready**
- CLI tools for all steps
- Async I/O for performance
- Error handling and retries
- Comprehensive logging
- Type hints and documentation

✅ **Analysis Ready**
- ~1,000+ historical price datasets
- ~2,800+ categorized messages
- Perfect for ML training
- Ready for backtesting
- Event study analysis-ready

**Total Setup Time:** 1-2 hours for complete pipeline → **Saves years of manual work!**

---

**Get Started Now:**

```bash
# 1. Fetch historical data
python fetch_historical_data.py --sp500

# 2. Filter by volume
python filter_by_volume.py --data-dir data --output high_volume.csv

# 3. Process Discord messages
python discord_to_csv.py -i discord.txt -o output.csv -f high_volume.csv
```

**You now have a complete financial data pipeline!** 📈

Happy analyzing! 🚀
