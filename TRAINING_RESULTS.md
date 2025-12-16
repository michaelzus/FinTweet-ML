# FinBERT Tweet Classifier - Training Results

## Overview

This document tracks training experiments for the FinBERT multi-modal tweet classifier.

**Dataset**: `output/15-dec-enrich7.csv` (5,866 total samples)
- After filtering: 4,523 reliable samples
- Train: 3,209 (70.9%)
- Validation: 675 (14.9%)
- Test: 639 (14.1%)

**Target**: `label_1d_3class` (1-day BUY/HOLD/SELL)

---

## Experiment 1: Full Fine-Tuning (Baseline)

**Command**:
```bash
python -m tweet_classifier.train --epochs 5 --batch-size 16 --evaluate-test
```

**Configuration**:
- BERT: Fine-tuned (all parameters trainable)
- Dropout: 0.3
- Learning rate: 2e-5
- Epochs: 5

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 41.9% |
| Test F1 Macro | 37.6% |
| Test F1 Weighted | 41.8% |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| SELL | 45% | 66% | 54% | 272 |
| HOLD | 34% | 31% | 33% | 134 |
| BUY | 39% | 20% | 26% | 233 |

### Trading Metrics

| Metric | Value |
|--------|-------|
| Information Coefficient | 0.106 (p=0.007) |
| Directional Accuracy | 51.5% |
| Simulated Sharpe (top 30%) | -0.92 |

### Baseline Comparison

| Baseline | Accuracy | vs Model |
|----------|----------|----------|
| Naive (SELL) | 42.6% | Model -1.5% ❌ |
| Random | 33.3% | Model +25.8% |
| Weighted Random | 34.6% | Model +21.1% |

### Issues Identified

1. **Heavy SELL bias** - Model predicted SELL 394 times (61.6% of predictions)
2. **BUY under-prediction** - Only 20% recall on BUY class (missed 153/233 BUY signals)
3. **Below naive baseline** - Model worse than always predicting SELL
4. **Likely overfitting** - Fine-tuning all BERT params on small dataset

---

## Experiment 2: Frozen BERT (Option A)

**Command**:
```bash
python -m tweet_classifier.train --epochs 5 --freeze-bert --evaluate-test
```

**Configuration**:
- BERT: **Frozen** (only classifier head trainable)
- Dropout: 0.3
- Learning rate: 2e-5
- Epochs: 5

### Results

| Metric | Value | vs Exp 1 |
|--------|-------|----------|
| Test Accuracy | **43.0%** | +1.1% |
| Test F1 Macro | **42.4%** | **+4.8%** |
| Test F1 Weighted | 43.2% | +1.4% |

### Per-Class Performance

| Class | Precision | Recall | F1 | vs Exp 1 |
|-------|-----------|--------|-----|----------|
| SELL | 50% | 42% | 45% | More balanced |
| HOLD | 34% | 43% | 38% | +5% F1 |
| BUY | 43% | 45% | **44%** | **+18% F1** |

### Trading Metrics

| Metric | Value | vs Exp 1 |
|--------|-------|----------|
| Information Coefficient | 0.096 (p=0.015) | Similar |
| Directional Accuracy | **54.5%** | **+3.0%** |
| Simulated Sharpe (top 30%) | -1.06 | Similar |

### Baseline Comparison

| Baseline | Accuracy | vs Model |
|----------|----------|----------|
| Naive (SELL) | 42.6% | **Model +1.1%** ✅ |
| Random | 33.3% | Model +29.1% |
| Weighted Random | 34.6% | Model +24.3% |

### Improvements

1. **Now beats naive baseline** (+1.1%)
2. **Much more balanced predictions** - No more extreme SELL bias
3. **BUY class dramatically improved** - F1 from 26% to 44% (+18%)
4. **Better directional accuracy** - 54.5% vs 51.5%
5. **IC still statistically significant** (p=0.015)

---

## Comparison Summary

| Metric | Exp 1 (Fine-tune) | Exp 2 (Frozen) | Winner |
|--------|-------------------|----------------|--------|
| Accuracy | 41.9% | **43.0%** | Frozen |
| F1 Macro | 37.6% | **42.4%** | Frozen |
| vs Naive | -1.5% | **+1.1%** | Frozen |
| SELL F1 | 54% | 45% | Fine-tune* |
| HOLD F1 | 33% | **38%** | Frozen |
| BUY F1 | 26% | **44%** | Frozen |
| IC | 0.106 | 0.096 | Similar |
| Dir. Accuracy | 51.5% | **54.5%** | Frozen |

*Fine-tune "wins" on SELL F1 due to over-prediction bias, not quality.

### Confusion Matrices

**Experiment 1 (Fine-tune)**:
```
             Predicted
           SELL  HOLD  BUY
Actual SELL  178    38   56
       HOLD   69    42   23
       BUY   147    39   47
```
Problem: 394 SELL predictions vs 272 actual SELL

**Experiment 2 (Frozen)**:
```
             Predicted
           SELL  HOLD  BUY
Actual SELL  113    63   96
       HOLD   31    57   46
       BUY    82    46  105
```
Much more balanced predictions (226 SELL, 166 HOLD, 247 BUY)

---

## Key Insights

1. **Small dataset + large model = overfitting**
   - With only ~3200 training samples, fine-tuning all 110M BERT parameters leads to overfitting
   - Frozen BERT keeps pre-trained knowledge intact

2. **Class imbalance handling improved with frozen BERT**
   - Fine-tuned model learned to predict SELL too aggressively
   - Frozen model maintained better class balance

3. **Trading signal quality preserved**
   - IC still statistically significant (0.096, p=0.015)
   - Directional accuracy improved to 54.5%

4. **Sharpe ratio still negative**
   - Both models show negative Sharpe on simulated trades
   - Need more data or better features to be profitable

---

## Next Experiments to Try

### Option B: Higher Dropout
```bash
python -m tweet_classifier.train --epochs 5 --dropout 0.4 --evaluate-test
```

### Option C: Fewer Epochs (Early Stopping)
```bash
python -m tweet_classifier.train --epochs 3 --evaluate-test
```

### Combined: Frozen + Lower Epochs
```bash
python -m tweet_classifier.train --epochs 3 --freeze-bert --evaluate-test
```

### Option D: Label Smoothing
Requires code change to add `label_smoothing` parameter.

---

## Recommendations

1. **Use frozen BERT** for this dataset size
2. **Get more training data** - current bottleneck is ~3200 samples
3. **Consider temporal split** - train on older data, test on newer
4. **Monitor IC p-value** - keep it <0.05 for statistical significance
5. **Focus on directional accuracy** - more important than raw accuracy for trading

---

## Files

- Model: `models/finbert-tweet-classifier/`
- Confusion matrix: `models/finbert-tweet-classifier/evaluation/confusion_matrix.png`
- Full metrics: `models/finbert-tweet-classifier/evaluation/evaluation_results.json`

