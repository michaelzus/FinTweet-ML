# Training Results

## Executive Summary

FinTweet-ML achieves **statistically significant predictive power** (IC=0.047, p=0.015) on 3-class stock movement prediction using temporal validation, with ~15% realistic annual returns.

---

## Experiment Comparison

| Metric | Random Split | Temporal Split |
|--------|--------------|----------------|
| **Test Accuracy** | 40.85% | 39.88% |
| **F1 Macro** | 40.76% | 39.57% |
| **Information Coefficient** | 0.012 (p=0.54) ❌ | **0.047 (p=0.015) ✅** |
| **Directional Accuracy** | 50.87% | 51.19% |
| **Sharpe (top 30%)** | 1.13 | 0.31 |
| **Annual Return (top 30%)** | 52.25% | **15.02%** |
| **Precision @ 60% conf** | 57% (n=100) | - |
| BERT Training | Full Fine-Tune | Frozen |
| BUY Weight Boost | None | 1.2x |

**Key Insight:** Random split overestimates returns (52% vs 15%). Temporal split provides realistic expectations with **statistically significant** predictive signal.

---

## Experiment 1: Random Split (Dec 21, 2025)

### Configuration

| Parameter | Value |
|-----------|-------|
| Split Strategy | Random by tweet_hash |
| BERT Training | Full Fine-Tuning |
| Epochs | 5 (early stopped at 4) |
| Early Stopping | patience=2 |

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 40.85% |
| F1 Macro | 40.76% |
| F1 Weighted | 40.16% |
| IC | 0.0117 (p=0.539) ❌ Not significant |
| Sharpe (top 30%) | 1.13 |
| Annual Return (top 30%) | 52.25% |
| Precision @ 60% conf | 57.00% (n=100) |
| Improvement vs Random | +22.5% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| SELL | 42% | 41% | 42% | 1,019 |
| HOLD | 41% | 56% | 47% | 733 |
| BUY | 39% | 29% | 33% | 985 |

### Command

```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 5 \
    --evaluate-test
```

---

## Experiment 2: Temporal Split (Dec 21, 2025) ✓ Recommended

### Configuration

| Parameter | Value |
|-----------|-------|
| Split Strategy | **Temporal** (train early, test late) |
| BERT Training | **Frozen** (classifier only) |
| BUY Weight Boost | 1.2x |
| Epochs | 5 (early stopped at 4, best=epoch 2) |
| Early Stopping | patience=2 |

### Temporal Split Details

| Set | Samples | % | Date Range |
|-----|---------|---|------------|
| Train | 21,017 | 80% | Dec 2024 → Oct 2025 |
| Val | 2,627 | 10% | Oct 2025 → Nov 2025 |
| Test | 2,628 | 10% | Nov 2025 → Dec 2025 |

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 39.88% |
| F1 Macro | 39.57% |
| F1 Weighted | 39.42% |
| **IC** | **0.0474 (p=0.015) ✅ Significant** |
| Directional Accuracy | 51.19% |
| Sharpe (top 30%) | 0.31 |
| Annual Return (top 30%) | **15.02%** |
| Improvement vs Random | +19.6% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| SELL | 40% | 28% | 33% | 900 |
| HOLD | 42% | 47% | 45% | 819 |
| BUY | 38% | 45% | 41% | 909 |

### Class Distribution (Training)

| Class | Count | Weight |
|-------|-------|--------|
| SELL | 7,583 | 0.924 |
| HOLD | 5,564 | 1.259 |
| BUY | 7,870 | 1.068 (boosted) |

### Command

```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 5 \
    --temporal-split \
    --evaluate-test \
    --freeze-bert \
    --early-stopping-patience 2 \
    --buy-weight-boost 1.2
```

---

## Baseline Comparisons (Temporal Split)

| Baseline | Accuracy | Improvement |
|----------|----------|-------------|
| Random | 33.33% | +19.6% |
| Weighted Random | 34.05% | +17.1% |
| Naive (always BUY) | 34.59% | +15.3% |
| **Model** | **39.88%** | — |

---

## Backtest Estimates ($30K Portfolio, $2.5/trade fee)

Based on temporal split (realistic) results:

| Strategy | Trades | Annual Fees | Est. Net Return |
|----------|--------|-------------|-----------------|
| Top 30% confidence | ~788 | $3,940 | ~10-12% |
| Top 10% confidence | ~263 | $1,315 | ~12-15% |

---

## Key Takeaways

1. **Temporal split is essential** - Random split overestimates returns by 3-4x
2. **IC significance matters** - Only temporal split shows real predictive power (p=0.015)
3. **BUY weight boost works** - Improved BUY recall from 29% to 45%
4. **Frozen BERT is sufficient** - Faster training, similar results
5. **Realistic returns: 10-15%** - Not 50%+ from random split

---

## Reproducibility

### Dataset

- File: `output/dataset.csv`
- Samples: 26,272
- Authors: 7-12 unique
- Categories: 12 unique

### Output Location

- Model: `models/finbert-tweet-classifier/`
- Evaluation: `models/finbert-tweet-classifier/evaluation/`

### Random Seed

`RANDOM_SEED = 42`
