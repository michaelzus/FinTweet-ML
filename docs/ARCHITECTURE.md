# Architecture Documentation

## System Overview

FinTweet-ML is an end-to-end ML pipeline for financial sentiment analysis, consisting of two main components:

1. **tweet_enricher** - Data collection and feature engineering pipeline
2. **tweet_classifier** - FinBERT-based multi-modal classifier

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINTWEET-ML PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Twitter   │    │    IBKR     │    │  Enriched   │    │   Trained   │
│   Tweets    │───▶│  OHLCV Data │───▶│   Dataset   │───▶│    Model    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       │    tweet_enricher package          │    tweet_classifier package
       └──────────────────┴──────────────────┴──────────────────┘
```

---

## Package Structure

```
src/
├── tweet_enricher/              # Data Pipeline
│   ├── cli.py                   # CLI commands
│   ├── config.py                # Configuration
│   ├── core/
│   │   ├── enricher.py          # Main enrichment logic
│   │   └── indicators.py        # Technical indicators (pandas-ta)
│   ├── data/
│   │   ├── ib_fetcher.py        # IBKR API client
│   │   ├── cache.py             # Disk + memory caching
│   │   └── tickers.py           # S&P500/Russell1000 lists
│   ├── twitter/
│   │   ├── client.py            # Twitter API client
│   │   ├── database.py          # SQLite storage
│   │   └── sync.py              # Incremental sync
│   ├── parsers/
│   │   └── discord.py           # Discord export parser
│   ├── io/
│   │   ├── feather.py           # Feather file I/O
│   │   └── csv_writer.py        # CSV output
│   └── market/
│       ├── session.py           # Market hours detection
│       └── regime.py            # Market regime classification
│
└── tweet_classifier/            # ML Training
    ├── config.py                # Model configuration
    ├── model.py                 # FinBERT multi-modal architecture
    ├── dataset.py               # PyTorch dataset
    ├── train.py                 # Training script
    ├── trainer.py               # Custom trainer (weighted loss)
    ├── evaluate.py              # Evaluation metrics
    └── data/
        ├── loader.py            # Data loading
        ├── splitter.py          # Temporal splits
        └── weights.py           # Class weights
```

---

## Data Pipeline (tweet_enricher)

### Stage 1: Data Collection

```
┌──────────────────┐         ┌──────────────────┐
│  Twitter API.io  │         │  Interactive     │
│  (tweets.db)     │         │  Brokers API     │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │  Financial tweets          │  OHLCV data
         │  6 source accounts         │  S&P 500 + Russell 1000
         │                            │
         └──────────┬─────────────────┘
                    │
             ┌──────▼──────┐
             │   Merger    │
             │  (by ticker │
             │ + timestamp)│
             └─────────────┘
```

### Stage 2: Feature Engineering

| Feature Type | Examples | Source |
|--------------|----------|--------|
| **Price** | `price_at_tweet`, `return_1hr` | IBKR intraday |
| **Technical** | `rsi_14`, `volatility_7d` | pandas-ta |
| **Volume** | `relative_volume` | IBKR daily |
| **Context** | `market_regime`, `sector` | Computed |
| **Text** | `text` (cleaned) | Twitter |

### Stage 3: Label Generation

```python
# 1-day forward return → 3-class label
if return_1d > 0.005:
    label = "BUY"
elif return_1d < -0.005:
    label = "SELL"
else:
    label = "HOLD"
```

---

## ML Pipeline (tweet_classifier)

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FinBERT Multi-Modal                          │
├─────────────────┬─────────────────┬────────────────────────────┤
│   Text Input    │  Numerical (10) │  Categorical (5)           │
│   (128 tokens)  │   Features      │   Features                 │
└────────┬────────┴────────┬────────┴─────────┬──────────────────┘
         │                 │                  │
    ┌────▼────┐      ┌─────▼─────┐     ┌──────▼──────┐
    │ FinBERT │      │  Linear   │     │ Embeddings  │
    │ (frozen)│      │  (32-dim) │     │  (learned)  │
    │ 768-dim │      └─────┬─────┘     └──────┬──────┘
    └────┬────┘            │                  │
         │                 │                  │
         └────────────┬────┴──────────────────┘
                      │
               ┌──────▼──────┐
               │   Concat    │
               │  ~850 dim   │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │  Dropout    │
               │   (0.3)     │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │  Classifier │
               │   Linear    │
               │  → 3 class  │
               └─────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base Model | `yiyanghkust/finbert-tone` | Pre-trained on financial text |
| BERT Layers | **Frozen** | Prevents overfitting |
| Multi-modal | Text + Numerical + Categorical | Captures market context |
| Loss | Cross-entropy with class weights | Handles imbalance |
| Split | **Temporal** (80/10/10) | Prevents data leakage |

### Training Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Load CSV   │────▶│  Temporal   │────▶│  Create     │
│             │     │   Split     │     │  Datasets   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│   Output    │◀────│  Evaluate   │◀────│   Train     │
│   Model     │     │   on Test   │     │  (5 epochs) │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Data Validation

### Look-Ahead Bias Prevention

```python
# ✅ SAFE: Uses only historical data
features = {
    "rsi_14": compute_rsi(data[:tweet_idx]),      # Before tweet
    "volatility_7d": compute_vol(data[:tweet_idx])
}

# ❌ LEAK: Uses future data
features = {
    "return_1hr": (future_price - current) / current  # EXCLUDED
}
```

### Temporal Split Strategy

```
|←────── Train (80%) ──────→|←─ Val ─→|←─ Test ─→|
|        Jan - Oct          |   Nov   |    Dec   |
```

---

## Key Components

### TweetEnricher (core/enricher.py)

Main orchestrator for data enrichment:
- Connects to IBKR for price data
- Computes technical indicators
- Generates labels
- Handles market session classification

### FinBERTMultiModal (model.py)

PyTorch model combining:
- Frozen FinBERT for text embeddings
- Learned embeddings for categorical features
- Linear projection for numerical features
- Classification head

### WeightedTrainer (trainer.py)

Custom Hugging Face trainer:
- Handles class imbalance via weighted loss
- Computes trading-specific metrics
- Supports early stopping

---

## CLI Commands

### Data Pipeline

```bash
# Fetch tweets from Twitter
tweet-enricher twitter-sync --days 30

# Fetch market data from IBKR
tweet-enricher fetch --sp500

# Enrich tweets with features
tweet-enricher enrich -i tweets.csv -o enriched.csv
```

### Model Training

```bash
# Train classifier
python -m tweet_classifier.train \
    --epochs 5 \
    --batch-size 16 \
    --freeze-bert \
    --evaluate-test
```

---

## Configuration

### tweet_enricher/config.py

```python
ET = pytz.timezone("US/Eastern")
MARKET_OPEN = 9 * 60 + 30   # 9:30 AM ET
MARKET_CLOSE = 16 * 60      # 4:00 PM ET

TWITTER_ACCOUNTS = [
    "StockMKTNewz", "wallstengine", ...
]
```

### tweet_classifier/config.py

```python
TARGET_COLUMN = "label_1d_3class"
LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}

NUMERICAL_FEATURES = [
    "volatility_7d", "rsi_14", "relative_volume", ...
]

CATEGORICAL_FEATURES = [
    "author", "category", "market_regime", ...
]
```

---

## Design Principles

1. **Separation of Concerns** - Data pipeline and ML training are independent packages
2. **Dependency Injection** - Components are injected for testability
3. **Configuration as Code** - All constants in dedicated config files
4. **Temporal Integrity** - Strict prevention of look-ahead bias
5. **Incremental Processing** - Support for resumable data fetching
