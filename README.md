# FinTweet-ML

End-to-end ML pipeline for financial tweet sentiment classification using FinBERT.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FinBERT](https://img.shields.io/badge/Model-FinBERT-green.svg)](https://huggingface.co/yiyanghkust/finbert-tone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

FinTweet-ML demonstrates a complete ML workflow for predicting stock movements from financial social media:

| Component | Description |
|-----------|-------------|
| **Data Collection** | Twitter API + Interactive Brokers OHLCV |
| **Feature Engineering** | Technical indicators, market context, text processing |
| **Model** | Multi-modal FinBERT with frozen BERT layers |
| **Evaluation** | Temporal validation, trading metrics |

### Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **42.8%** (vs 33.3% random) |
| Information Coefficient | **0.054** (p=0.027) |
| F1 Macro | 38.2% |

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Twitter   │    │    IBKR     │    │  Enriched   │    │  FinBERT    │
│   Tweets    │───▶│  Market     │───▶│  Dataset    │───▶│  Classifier │
│             │    │  Data       │    │  (35K rows) │    │  (3-class)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Model Architecture

```
              ┌─────────────────────────────────────────┐
              │         FinBERT Multi-Modal             │
              ├─────────────┬─────────────┬─────────────┤
              │  Text (128) │ Numerical   │ Categorical │
              │             │ (10 feats)  │ (5 feats)   │
              └──────┬──────┴──────┬──────┴──────┬──────┘
                     │             │             │
              ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
              │   FinBERT   │ │ Linear  │ │ Embeddings  │
              │  (frozen)   │ │  (32)   │ │  (learned)  │
              └──────┬──────┘ └────┬────┘ └──────┬──────┘
                     └─────────────┼─────────────┘
                            ┌──────▼──────┐
                            │   Concat    │
                            │  + Dropout  │
                            └──────┬──────┘
                            ┌──────▼──────┐
                            │ BUY / HOLD  │
                            │   / SELL    │
                            └─────────────┘
```

---

## Features

### Data Pipeline
- **Multi-source ingestion**: Twitter API, Interactive Brokers
- **Timezone handling**: UTC → US Eastern with DST support
- **Look-ahead bias prevention**: Strict temporal feature computation
- **Data leakage checks**: Automated validation scripts

### ML Training
- **Pre-trained FinBERT**: Financial domain knowledge
- **Frozen BERT layers**: Prevents overfitting on small datasets
- **Multi-modal fusion**: Text + numerical + categorical features
- **Temporal splits**: 80/10/10 for realistic evaluation

### Evaluation
- **Trading metrics**: Information Coefficient, directional accuracy
- **Walk-forward validation**: Tests generalization across time
- **Class-weighted loss**: Handles label imbalance

---

## Project Structure

```
FinTweet-ML/
├── src/
│   ├── tweet_enricher/      # Data pipeline
│   │   ├── twitter/         # Twitter API client
│   │   ├── data/            # IBKR fetcher, caching
│   │   ├── core/            # Enrichment logic
│   │   └── market/          # Market session handling
│   │
│   └── tweet_classifier/    # ML training
│       ├── model.py         # FinBERT architecture
│       ├── train.py         # Training script
│       ├── evaluate.py      # Metrics & evaluation
│       └── data/            # Data loading & splits
│
├── scripts/
│   └── validate_dataset.py  # Data validation
│
├── tests/                   # Unit tests
├── notebooks/               # Demo notebooks
│
└── docs/
    ├── ARCHITECTURE.md      # System design
    ├── DATASET.md           # Data documentation
    └── RESULTS.md           # Training results
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FinTweet-ML.git
cd FinTweet-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .
```

### Training

```bash
# Train the classifier
python -m tweet_classifier.train \
    --epochs 5 \
    --batch-size 16 \
    --freeze-bert \
    --evaluate-test
```

### Data Pipeline (requires IBKR TWS)

```bash
# Fetch market data
tweet-enricher fetch --sp500

# Enrich tweets with features
tweet-enricher enrich -i tweets.csv -o enriched.csv
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and components |
| [DATASET.md](docs/DATASET.md) | Data sources, features, labels |
| [RESULTS.md](docs/RESULTS.md) | Training experiments and metrics |

---

## Key Findings

1. **Frozen BERT outperforms fine-tuned** - Prevents overfitting on limited data
2. **Technical indicators add value** - Especially volatility and volume features
3. **Temporal splits are critical** - Random splits overestimate performance by ~5%
4. **Data quality > quantity** - Filtered datasets outperform larger noisy ones

---

## Technologies

- **Python 3.11+**
- **PyTorch 2.0+** - Deep learning framework
- **Transformers** - Hugging Face for FinBERT
- **pandas / pandas-ta** - Data processing & technical indicators
- **Interactive Brokers API** - Market data
- **SQLite** - Tweet storage

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

Michael Zuskin

*Part of my ML Portfolio - demonstrating end-to-end ML pipeline development.*
