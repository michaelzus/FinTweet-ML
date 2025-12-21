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

## Experiment 3: Frozen BERT + 3 Epochs (Early Stopping)

**Command**:
```bash
python -m tweet_classifier.train --epochs 3 --freeze-bert --evaluate-test
```

**Configuration**:
- BERT: Frozen
- Dropout: 0.3
- Learning rate: 2e-5
- Epochs: **3** (vs 5 in Exp 2)

### Results

| Metric | Value | vs Exp 2 (5 epochs) |
|--------|-------|---------------------|
| Test Accuracy | 37.9% | **-5.1%** ❌ |
| Test F1 Macro | 37.2% | **-5.2%** ❌ |
| Test F1 Weighted | 37.9% | -5.3% |

### Trading Metrics

| Metric | Value | vs Exp 2 |
|--------|-------|----------|
| Information Coefficient | 0.040 (p=0.313) | **NOT significant** ❌ |
| Directional Accuracy | 50.0% | -4.5% |
| Simulated Sharpe (top 30%) | -1.02 | Similar |

### Baseline Comparison

| Baseline | Accuracy | vs Model |
|----------|----------|----------|
| Naive (SELL) | 42.6% | Model **-11.0%** ❌ |
| Random | 33.3% | Model +13.6% |

### Issues

1. **Underfitting** - 3 epochs too few, model hasn't converged
2. **IC not statistically significant** - p=0.313 (need p<0.05)
3. **Below naive baseline by 11%** - Worse than always predicting SELL

---

## Experiment 4: Frozen BERT + Higher Dropout (0.4)

**Command**:
```bash
python -m tweet_classifier.train --epochs 5 --freeze-bert --dropout 0.4 --evaluate-test
```

**Configuration**:
- BERT: Frozen
- Dropout: **0.4** (vs 0.3 in Exp 2)
- Learning rate: 2e-5
- Epochs: 5

### Results

| Metric | Value | vs Exp 2 (dropout 0.3) |
|--------|-------|------------------------|
| Test Accuracy | 41.9% | -1.1% |
| Test F1 Macro | 40.9% | -1.5% |
| Test F1 Weighted | 41.7% | -1.5% |

### Trading Metrics

| Metric | Value | vs Exp 2 |
|--------|-------|----------|
| Information Coefficient | 0.120 (p=0.003) | **Better significance** ✅ |
| Directional Accuracy | 52.9% | -1.6% |
| Simulated Sharpe (top 30%) | -1.77 | Worse |

### Baseline Comparison

| Baseline | Accuracy | vs Model |
|----------|----------|----------|
| Naive (SELL) | 42.6% | Model -1.5% ❌ |
| Random | 33.3% | Model +25.8% |

### Observations

1. **IC improved** - More statistically significant (p=0.003 vs 0.015)
2. **Accuracy dropped** - Higher dropout reduced performance
3. **Still below naive baseline** - Not a viable improvement

---

## Experiment 5: Combined (3 Epochs + Dropout 0.4 + Frozen)

**Command**:
```bash
python -m tweet_classifier.train --epochs 3 --freeze-bert --dropout 0.4 --evaluate-test
```

**Configuration**:
- BERT: Frozen
- Dropout: 0.4
- Learning rate: 2e-5
- Epochs: 3

### Results

| Metric | Value | Note |
|--------|-------|------|
| Test Accuracy | 34.6% | **Worst of all** ❌ |
| Test F1 Macro | 34.4% | **Worst of all** ❌ |
| Test F1 Weighted | 33.6% | Worst of all |

### Trading Metrics

| Metric | Value | Note |
|--------|-------|------|
| Information Coefficient | 0.011 (p=0.777) | **NOT significant** ❌ |
| Directional Accuracy | 46.6% | **Below random!** ❌ |
| Simulated Sharpe (top 30%) | -2.68 | Worst of all |

### Baseline Comparison

| Baseline | Accuracy | vs Model |
|----------|----------|----------|
| Naive (SELL) | 42.6% | Model **-18.7%** ❌ |
| Random | 33.3% | Model +3.8% |

### Issues

1. **Severe underfitting** - Too much regularization (3 epochs + dropout 0.4)
2. **Below random performance** - 46.6% directional accuracy
3. **IC completely insignificant** - p=0.777

---

## Updated Comparison Summary (All 5 Experiments)

| Metric | Exp 1 (Fine-tune) | Exp 2 (Frozen, 5ep) | Exp 3 (3ep) | Exp 4 (drop0.4) | Exp 5 (3ep+drop0.4) | **WINNER** |
|--------|-------------------|---------------------|-------------|-----------------|---------------------|------------|
| **Accuracy** | 41.9% | **43.0%** | 37.9% | 41.9% | 34.6% | **Exp 2** |
| **F1 Macro** | 37.6% | **42.4%** | 37.2% | 40.9% | 34.4% | **Exp 2** |
| **vs Naive** | -1.5% | **+1.1%** ✅ | -11.0% | -1.5% | -18.7% | **Exp 2** |
| **IC** | 0.106 (p=0.007) | 0.096 (p=0.015) | 0.040 (p=0.313) | **0.120 (p=0.003)** | 0.011 (p=0.777) | **Exp 4** |
| **Dir. Accuracy** | 51.5% | **54.5%** | 50.0% | 52.9% | 46.6% | **Exp 2** |
| **BUY F1** | 26% | **44%** | 42% | 46% | 38% | **Exp 4** |
| **Training Time** | Longest | Medium | **Shortest** | Medium | Shortest | Exp 3/5 |

### Rankings by Overall Performance

1. **🥇 Experiment 2** (Frozen BERT, 5 epochs, dropout 0.3) - **BEST OVERALL**
   - Only model beating naive baseline
   - Best accuracy (43.0%) and directional accuracy (54.5%)
   - IC statistically significant (p=0.015)
   - Good balance across all classes

2. **🥈 Experiment 4** (Frozen BERT, 5 epochs, dropout 0.4)
   - Best IC significance (p=0.003) ✅
   - Best BUY F1 (46%)
   - But below naive baseline (-1.5%)

3. **🥉 Experiment 1** (Fine-tuned BERT)
   - Reasonable IC (p=0.007)
   - But overfits and below naive baseline

4. **Experiment 3** (3 epochs)
   - Underfits, IC not significant

5. **❌ Experiment 5** (3 epochs + dropout 0.4)
   - Severe underfitting, worst of all

---

## Updated Key Insights

### What Works

1. ✅ **Frozen BERT is essential** with small datasets (~3K samples)
2. ✅ **5 epochs with dropout 0.3** is the sweet spot
3. ✅ **Author/category embeddings help** reduce bias
4. ✅ **1-day labels** are less noisy than 1-hour labels

### What Doesn't Work

1. ❌ **Fine-tuning BERT** on 3K samples → overfitting
2. ❌ **3 epochs** → underfitting (models haven't converged)
3. ❌ **Dropout 0.4 alone** → slight degradation
4. ❌ **Combining too much regularization** (3 epochs + dropout 0.4) → severe underfitting

### Critical Limitations

1. **Small dataset bottleneck** - All models struggle with ~3,200 training samples
2. **Negative Sharpe ratios** - None of the models are profitable in simulation
3. **Marginal improvement over naive** - Best model only +1.1% better
4. **Dataset quality issues** - 29.7% premarket tweets have minor data leakage

---

## Updated Recommendations

### For Current Dataset (~3K samples)

**Use Experiment 2 configuration:**
```bash
python -m tweet_classifier.train --epochs 5 --freeze-bert --evaluate-test
```
- BERT: Frozen
- Epochs: 5
- Dropout: 0.3
- Expected: 43% accuracy, 54.5% directional accuracy

### To Actually Improve Performance

**Priority 1: Get more data** (Target: 10,000+ samples)
- Parse more Discord channels
- Expand date range
- With 10K+ samples, can try fine-tuning again

**Priority 2: Fix data quality issues**
- Address 29.7% premarket leakage (change `<=` to `<` in enricher)
- Consider temporal split for production robustness

**Priority 3: Feature engineering**
- Add more technical indicators
- Include volatility measures
- Try sentiment scores from FinBERT

**Priority 4: If must deploy with current model**
- Use Experiment 2 (Frozen, 5 epochs)
- Monitor IC p-value in production
- Focus on high-confidence predictions only
- Accept that profitability is not guaranteed (negative Sharpe)

---

## Experiment 6: Fixed Data Leakage (HONEST BASELINE) ✅

**Command**:
```bash
# First, fix the data leakage in enricher.py line 313
# Changed: daily_df = daily_df_full[daily_df_full.index.date <= tweet_date].copy()
# To:      daily_df = daily_df_full[daily_df_full.index.date < tweet_date].copy()

# Re-enrich with fixed logic
tweet-enricher enrich -i output/15-dec6.csv -o output/15-dec-enrich8.csv

# Retrain with same config as Exp 2
python -m tweet_classifier.train --data-path output/15-dec-enrich8.csv --epochs 5 --freeze-bert --evaluate-test
```

**Configuration**:
- BERT: Frozen
- Dropout: 0.3
- Learning rate: 2e-5
- Epochs: 5
- **Dataset**: `output/15-dec-enrich8.csv` (FIXED - no data leakage)

### Results

|| Metric | Value | vs Exp 2 (Leaky Data) |
||--------|-------|----------------------|
|| Test Accuracy | 38.5% | **-4.5%** ⚠️ |
|| Test F1 Macro | 36.0% | **-6.4%** ⚠️ |
|| Test F1 Weighted | 36.0% | -7.2% |

### Per-Class Performance

|| Class | Precision | Recall | F1 | Support | vs Exp 2 |
||-------|-----------|--------|-----|---------|----------|
|| SELL | 44% | 16% | 24% | 271 | **-21% F1** |
|| HOLD | 37% | 39% | 38% | 158 | Similar |
|| BUY | 38% | 63% | 47% | 247 | +3% F1 |

### Trading Metrics

|| Metric | Value | vs Exp 2 |
||--------|-------|----------|
|| Information Coefficient | **-0.031** (p=0.428) | **NOT significant** ❌ |
|| Directional Accuracy | 48.2% | **-6.3%** (below random!) |
|| Simulated Sharpe (top 30%) | -1.19 | Worse |

### Baseline Comparison

|| Baseline | Accuracy | vs Model |
||----------|----------|----------|
|| Naive (SELL) | 40.1% | Model **-1.6%** ❌ |
|| Random | 33.3% | Model +15.6% |

### Critical Insights

**🚨 DATA LEAKAGE WAS CONFIRMED:**

1. **Performance dropped significantly** after fixing the leakage
   - Accuracy: 43.0% → 38.5% (-4.5%)
   - IC became non-significant: 0.096 → -0.031 (p=0.428)
   - Directional accuracy dropped to 48.2% (worse than random)

2. **What was the leakage?**
   - Line 313 in enricher.py: `daily_df.index.date <= tweet_date`
   - For tweets at 9:31 AM on Oct 21, RSI/volatility included Oct 21's 4:00 PM close
   - This is ~6.5 hours of future information

3. **Honest baseline is much weaker**
   - Model can't beat naive baseline anymore
   - IC not statistically significant
   - Directional accuracy below 50% (worse than coin flip)

**✅ GOOD NEWS:**
- This is the **production-ready baseline** - what you'd actually get in real trading
- No more inflated metrics from data leakage
- Can now make honest decisions about model deployment

**⚠️ BAD NEWS:**
- Current model is **not viable for production** (below naive baseline, IC not significant)
- Need significant improvements before deployment

---

## Final Comparison Summary (All 6 Experiments)

|| Metric | Exp 2 (Leaky) | **Exp 6 (Fixed)** | Difference |
||--------|---------------|-------------------|------------|
|| **Accuracy** | 43.0% | **38.5%** | **-4.5%** |
|| **F1 Macro** | 42.4% | **36.0%** | **-6.4%** |
|| **vs Naive** | +1.1% ✅ | **-1.6%** ❌ | **-2.7%** |
|| **IC** | 0.096 (p=0.015) | **-0.031 (p=0.428)** ❌ | **Lost significance** |
|| **Dir. Accuracy** | 54.5% | **48.2%** | **-6.3%** |
|| **Sharpe** | -1.06 | **-1.19** | Worse |

### Key Takeaway

**Experiment 6 is the TRUE BASELINE** - Experiment 2's results were artificially inflated by data leakage.

---

## UPDATED Recommendations (Post-Leakage Fix)

### Current Status

❌ **Model is NOT ready for production**
- Below naive baseline (-1.6%)
- IC not statistically significant (p=0.428)
- Directional accuracy below random (48.2%)

### Path Forward

**Option 1: Improve Model (Recommended)**
1. **Get significantly more data** (10,000+ samples)
   - Current 3K samples insufficient for this task
   - More data may recover performance lost to leakage fix
2. **Add better features**
   - More technical indicators
   - Market regime indicators
   - Cross-sectional features (sector performance)
3. **Try different architectures**
   - Ensemble models
   - Time-aware models
   - Attention mechanisms over historical data

**Option 2: Accept Current Limitations**
- Use only for high-confidence predictions (top 10%)
- Combine with other signals (not standalone)
- Monitor performance closely in paper trading

**Option 3: Pivot Strategy**
- Focus on regression (predict returns) instead of classification
- Try longer horizons (3-day, 5-day instead of 1-day)
- Filter to only certain market conditions or sectors

---

## Files

- Model: `models/finbert-tweet-classifier/`
- Confusion matrix: `models/finbert-tweet-classifier/evaluation/confusion_matrix.png`
- Full metrics: `models/finbert-tweet-classifier/evaluation/evaluation_results.json`
- **Fixed dataset**: `output/15-dec-enrich8.csv` (NO DATA LEAKAGE)

