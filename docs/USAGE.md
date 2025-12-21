# 🚀 Quick Usage Guide

Complete workflow to convert Discord data → Enriched CSV with financial metrics.

---

## Prerequisites

- Python 3.8+ with dependencies: `pip install -r requirements.txt`
- Interactive Brokers TWS or Gateway running (for Steps 1 & 4)

---

## Complete 4-Step Pipeline

### Step 1: Fetch Historical Data (one-time setup)

```bash
# Start TWS/Gateway first, then:
python fetch_historical_data.py --sp500 --duration "1 Y"
```

Creates `data/daily/*.feather` files with OHLCV data for ~500 S&P 500 stocks.

**Options:**
- `--russell1000` - Fetch Russell 1000 instead
- `--all` - Fetch both S&P 500 + Russell 1000
- `--symbols AAPL MSFT TSLA` - Custom ticker list
- `--duration "6 M"` - Different time period

---

### Step 2: Filter Liquid Stocks by Volume

```bash
python filter_by_volume.py \
    --min-volume 1000000 \
    --output 1M_volume.csv
```

Creates `1M_volume.csv` with tickers having 1M+ average daily volume.

**Options:**
- `--data-dir data/daily` - Directory with feather files (default)
- `--min-volume 5000000` - Higher threshold (5M)
- `--min-volume 500000` - Lower threshold (500K)

---

### Step 3: Convert Discord → CSV

```bash
python discord_to_csv.py \
    -i discrod_data/15-dec.txt \
    -o output/tweets.csv \
    -f 1M_volume.csv
```

Converts Discord export to structured CSV with:
- Timestamps (Jerusalem → US Eastern)
- Extracted tickers (`$AAPL` → `AAPL`)
- Categories (Earnings, Breaking News, etc.)
- Cleaned text

**Options:**
- `--min-length 30` - Shorter minimum text length
- `--no-dedup` - Keep duplicates

---

### Step 4: Enrich with Financial Data

```bash
# TWS/Gateway must be running!
python enrich_tweets.py \
    --input output/tweets.csv \
    --output output/enriched_output.csv
```

Adds financial metrics to each tweet:

| Column | Description |
|--------|-------------|
| `price_at_tweet` | Stock price at tweet time |
| `price_1hr_after` | Price 1 hour later |
| `return_1d` | 1-day return context |
| `volatility_7d` | 7-day volatility |
| `rsi_14` | RSI indicator |
| `relative_volume` | Volume vs 20-day avg |
| `distance_from_ma_20` | Distance from 20-day MA |
| `spy_return_1d` | S&P 500 context |
| `return_1hr_adjusted` | Market-adjusted return |
| `label_5class` | Classification (strong_sell → strong_buy) |

---

## Quick Reference

```bash
# Full pipeline from scratch:
python fetch_historical_data.py --sp500                                    # ~30-60 min
python filter_by_volume.py --output 1M_volume.csv                          # ~30 sec
python discord_to_csv.py -i discrod_data/15-dec.txt -o output/tweets.csv -f 1M_volume.csv
python enrich_tweets.py --input output/tweets.csv --output output/enriched.csv

# If you already have 1M_volume.csv:
python discord_to_csv.py -i discrod_data/15-dec.txt -o output/tweets.csv -f 1M_volume.csv
python enrich_tweets.py --input output/tweets.csv --output output/enriched.csv
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start TWS/Gateway, enable API in settings |
| No tickers found | Lower `--min-volume` or check `data/daily/` directory |
| No messages output | Try without `-f` filter or lower `--min-length` |

---

## File Structure

```
TimeWaste2/
├── discrod_data/           # Input: Discord exports
│   ├── 15-dec.txt
│   └── AI_INVEST_ISRAEL.txt
├── data/
│   ├── daily/              # Historical daily OHLCV (feather format)
│   │   ├── AAPL.feather
│   │   └── ...
│   └── intraday/           # Intraday cache (feather format)
│       └── ...
├── 1M_volume.csv           # Filtered tickers (from Step 2)
└── output/                 # Final outputs
    ├── tweets.csv          # Parsed Discord (Step 3)
    └── enriched_output.csv # With financial data (Step 4)
```



# Train and evaluate on test set
source .venv/bin/activate
python -m tweet_classifier.train --epochs 5 --evaluate-test

# Or evaluate an existing model
python -m tweet_classifier.evaluate \
    --model-dir models/finbert-tweet-classifier/final \
    --data-path output/15-dec-enrich7.csv \
    --output-dir models/finbert-tweet-classifier/evaluation


cd /Users/mzus/dev/TimeWaste && source .venv/bin/activate && echo "Training full model with ALL features (Phase 1+2, 5 epochs)..." && python -m tweet_classifier.train \
    --data-path output/test2.csv \
    --output-dir models/full-phase1-2 \
    --epochs 5 \
    --batch-size 16 2>&1 | tee /tmp/phase1_2_full_training.log | tail -100




cd /Users/mzus/dev/TimeWaste && source venv/bin/activate && tweet-enricher twitter export -o output/tweets_2025_export.csv --since 2025-01-01 --until 2025-12-31 -v