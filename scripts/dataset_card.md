---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - finance
  - sentiment-analysis
  - stock-market
  - twitter
  - trading
  - finbert
size_categories:
  - 10K<n<100K
dataset_info:
  features:
    - name: timestamp
      dtype: string
    - name: ticker
      dtype: string
    - name: tweet_url
      dtype: string
    - name: label_1d_3class
      dtype:
        class_label:
          names:
            '0': BUY
            '1': HOLD
            '2': SELL
    - name: author
      dtype: string
    - name: category
      dtype: string
    - name: session
      dtype: string
    - name: market_regime
      dtype: string
    - name: sector
      dtype: string
    - name: market_cap_bucket
      dtype: string
    - name: volatility_7d
      dtype: float32
    - name: relative_volume
      dtype: float32
    - name: rsi_14
      dtype: float32
    - name: distance_from_ma_20
      dtype: float32
    - name: return_5d
      dtype: float32
    - name: return_20d
      dtype: float32
    - name: above_ma_20
      dtype: float32
    - name: slope_ma_20
      dtype: float32
    - name: gap_open
      dtype: float32
    - name: intraday_range
      dtype: float32
    - name: text
      dtype: string
  splits:
    - name: train
      num_examples: 30428
    - name: validation
      num_examples: 6520
    - name: test
      num_examples: 6520
---

# FinTweet Sentiment Dataset 2025

A curated dataset of financial tweets enriched with market data for sentiment-driven trading signal classification.

## Dataset Description

This dataset combines social media signals from financial Twitter accounts with real-time market data from Interactive Brokers to create labeled samples for 3-class sentiment classification (BUY/HOLD/SELL).

### Dataset Summary

| Property | Value |
|----------|-------|
| **Time Period** | 2024-2025 |
| **Total Samples** | 43,468 |
| **Target Variable** | 3-class: BUY, HOLD, SELL |
| **Split Strategy** | Temporal 70/15/15 |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| BUY | 16,666 | 38.3% |
| HOLD | 10,148 | 23.3% |
| SELL | 16,654 | 38.3% |

### Supported Tasks

- **Text Classification**: Predict trading signal (BUY/HOLD/SELL) from tweet text and market context
- **Multimodal Learning**: Combine text embeddings with numerical market features
- **Financial Sentiment Analysis**: Train domain-specific sentiment models

## Dataset Structure

### Data Fields

**Text Feature:**
- `text` (string): Raw tweet text with ticker mentions ($AAPL, $TSLA, etc.)

**Target:**
- `label_1d_3class` (ClassLabel): Trading signal based on 1-day forward return
  - `BUY` (0): Return > +0.5%
  - `HOLD` (1): -0.5% ≤ Return ≤ +0.5%
  - `SELL` (2): Return < -0.5%

**Categorical Features:**
| Feature | Description | Example Values |
|---------|-------------|----------------|
| `author` | Tweet source account | StockMKTNewz, wallstengine, AIStockSavvy |
| `category` | News category | Earnings, M&A, Breaking News, Personnel Changes |
| `session` | Market session | premarket, regular, afterhours |
| `market_regime` | Market condition | trending, volatile, calm |
| `sector` | Stock sector (GICS) | Technology, Healthcare, Financials |
| `market_cap_bucket` | Market cap size | mega_cap, large_cap, mid_cap, small_cap |

**Numerical Features (computed at tweet time, no look-ahead bias):**
| Feature | Description | Lookback |
|---------|-------------|----------|
| `volatility_7d` | 7-day price volatility | 7 days |
| `relative_volume` | Volume vs 20-day average | 20 days |
| `rsi_14` | Relative Strength Index | 14 days |
| `distance_from_ma_20` | Distance from 20-day MA | 20 days |
| `return_5d` | 5-day momentum | 5 days |
| `return_20d` | 20-day momentum | 20 days |
| `above_ma_20` | Binary: price > MA20 | 20 days |
| `slope_ma_20` | MA20 trend direction | 20 days |
| `gap_open` | Overnight gap | 1 day |
| `intraday_range` | Day's high-low range | 1 day |

**Metadata:**
- `timestamp` (string): Tweet timestamp (US Eastern Time)
- `ticker` (string): Stock ticker symbol
- `tweet_url` (string): Original tweet URL

### Data Splits

| Split | Samples | Time Period |
|-------|---------|-------------|
| train | 30,428 | Earliest 70% |
| validation | 6,520 | Middle 15% |
| test | 6,520 | Latest 15% |

Splits are **temporal** (chronologically ordered) to prevent data leakage and simulate real-world deployment.

## Usage

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("michael-zus/fintweet-sentiment-2025")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# Example: Get text and labels
texts = train["text"]
labels = train["label_1d_3class"]

# Example: Filter by sector
tech_samples = train.filter(lambda x: x["sector"] == "Technology")
```

### Training with Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("michael-zus/fintweet-sentiment-2025")

# Load FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)

# Tokenize
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./finbert-tweet-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

trainer.train()
```

## Data Collection

### Sources
- **Tweets**: Twitter financial accounts (@StockMKTNewz, @wallstengine, @amitisinvesting, @AIStockSavvy, @fiscal_ai, @EconomyApp)
- **Market Data**: Interactive Brokers TWS API (S&P 500 + Russell 1000 tickers)

### Label Generation
1. Identify entry price (first available bar after tweet)
2. Calculate return to next market open
3. Classify: BUY (>+0.5%), HOLD (±0.5%), SELL (<-0.5%)

### Quality Filters
All samples have been validated for:
- Valid entry price at tweet time
- Valid forward price for return calculation
- Market was open (no weekends/holidays)
- No data gaps in price data

## Considerations

### Biases
- Source accounts may have inherent biases in coverage
- Large-cap stocks are more frequently mentioned
- US market hours only (Eastern Time)

### Limitations
- Historical data only (2024-2025 period)
- Twitter-specific language patterns and conventions
- May not generalize to other social platforms or time periods

## Citation

```bibtex
@dataset{fintweet_sentiment_2025,
  title={FinTweet Sentiment Dataset 2025},
  author={michael-zus},
  year={2025},
  url={https://huggingface.co/datasets/michael-zus/fintweet-sentiment-2025},
  note={Financial tweets with market data for trading signal classification}
}
```

## License

MIT License - Free to use for research and commercial purposes with attribution.

