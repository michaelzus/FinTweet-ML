# FinBERT Tweet Classifier - Training Results

> **Date**: December 2025  
> **Dataset**: 42,749 financial tweets  
> **Model**: FinBERT MultiModal (yiyanghkust/finbert-tone)  
> **Task**: 3-class classification (SELL, HOLD, BUY)

---

## 📊 Executive Summary

Three model variants were trained with different hyperparameters. Key finding: **Better classification accuracy does not equal better trading performance**.

```mermaid
quadrantChart
    title Model Trade-offs: Classification vs Trading
    x-axis Low Accuracy --> High Accuracy
    y-axis Poor Trading --> Good Trading
    quadrant-1 Ideal (neither achieved)
    quadrant-2 Good Trading, Low Accuracy
    quadrant-3 Poor at Both
    quadrant-4 High Accuracy, Poor Trading
    V1: [0.55, 0.45]
    V2: [0.45, 0.75]
    V3: [0.65, 0.15]
```

| Aspect | Recommended Model |
|--------|-------------------|
| **Classification Accuracy** | V3 (41.92%) |
| **Trading Profitability** | V2 (Sharpe: -0.03) |
| **Statistical Significance** | V1 (IC p=0.025) |
| **Overall Best** | **V1** (balanced) |

---

## 🧪 Experiment Configurations

```mermaid
flowchart LR
    subgraph V1["V1: Full Fine-tuning"]
        V1A[epochs: 5]
        V1B[dropout: default]
        V1C[freeze_bert: false]
        V1D[learning_rate: 2e-5]
    end
    
    subgraph V2["V2: Frozen BERT"]
        V2A[epochs: 5]
        V2B[dropout: 0.5]
        V2C[freeze_bert: true]
        V2D[learning_rate: 2e-5]
    end
    
    subgraph V3["V3: Fast Training"]
        V3A[epochs: 3]
        V3B[dropout: 0.2]
        V3C[freeze_bert: false]
        V3D[learning_rate: 3e-5]
    end
    
    Dataset[(42,749 tweets)] --> V1
    Dataset --> V2
    Dataset --> V3
```

### Common Parameters

| Parameter | Value |
|-----------|-------|
| Temporal Split | 80% train / 10% val / 10% test |
| Train samples | 34,199 |
| Validation samples | 4,275 |
| Test samples | 4,275 |
| Early stopping patience | 2 epochs |
| BUY weight boost | 1.2x |
| Base model | yiyanghkust/finbert-tone |

### Class Distribution (Training Set)

```mermaid
pie showData
    title Training Set Class Distribution
    "SELL" : 12780
    "HOLD" : 7989
    "BUY" : 13430
```

---

## 📈 Training Curves

### V1: Full Fine-tuning (5 epochs)

```mermaid
xychart-beta
    title "V1: Loss Over Training"
    x-axis [E1, E2, E3, E4, E5]
    y-axis "Loss" 0.6 --> 1.4
    line "Train Loss" [1.05, 1.00, 0.92, 0.81, 0.72]
    line "Val Loss" [1.07, 1.10, 1.15, 1.25, 1.34]
```

⚠️ **Severe overfitting**: Train loss decreases while validation loss increases

### V2: Frozen BERT (5 epochs)

```mermaid
xychart-beta
    title "V2: Loss Over Training"
    x-axis [E1, E2, E3, E4, E5]
    y-axis "Loss" 1.05 --> 1.16
    line "Train Loss" [1.10, 1.09, 1.08, 1.08, 1.08]
    line "Val Loss" [1.095, 1.087, 1.079, 1.076, 1.077]
```

✅ **No overfitting**: Losses track together, but model barely learns (plateaus at ~1.08)

### V3: Fast Training (3 epochs)

```mermaid
xychart-beta
    title "V3: Loss Over Training"
    x-axis [E1, E2, E3]
    y-axis "Loss" 0.8 --> 1.2
    line "Train Loss" [1.05, 0.99, 0.88]
    line "Val Loss" [1.089, 1.130, 1.161]
```

⚠️ **Moderate overfitting**: Similar pattern to V1 but less severe

---

## 🎯 Classification Performance

### Test Set Metrics Comparison

| Metric | V1 | V2 | V3 | Best |
|--------|-----|-----|-----|------|
| **Accuracy** | 40.84% | 40.23% | **41.92%** | V3 |
| **F1 (macro)** | 40.92% | 33.89% | **41.42%** | V3 |
| **F1 (weighted)** | 40.74% | 32.64% | 40.73% | V1 |

```mermaid
xychart-beta
    title "Test Set Performance Comparison"
    x-axis ["V1", "V2", "V3"]
    y-axis "Score (%)" 30 --> 45
    bar "Accuracy" [40.84, 40.23, 41.92]
    bar "F1 Macro" [40.92, 33.89, 41.42]
```

### Per-Class Performance

#### V1: Full Fine-tuning
```
              precision    recall  f1-score   support
        SELL       0.42      0.36      0.38      1595
        HOLD       0.39      0.46      0.43      1147
         BUY       0.41      0.42      0.42      1533
    accuracy                           0.41      4275
```

#### V2: Frozen BERT
```
              precision    recall  f1-score   support
        SELL       0.46      0.04      0.07      1595  ← Almost never predicts SELL!
        HOLD       0.44      0.43      0.43      1147
         BUY       0.39      0.76      0.51      1533  ← Over-predicts BUY
    accuracy                           0.40      4275
```

#### V3: Fast Training
```
              precision    recall  f1-score   support
        SELL       0.45      0.25      0.32      1595
        HOLD       0.42      0.55      0.48      1147
         BUY       0.40      0.50      0.45      1533
    accuracy                           0.42      4275
```

### Per-Class Recall Comparison

```mermaid
xychart-beta
    title "Recall by Class"
    x-axis ["SELL", "HOLD", "BUY"]
    y-axis "Recall (%)" 0 --> 80
    bar "V1" [36, 46, 42]
    bar "V2" [4, 43, 76]
    bar "V3" [25, 55, 50]
```

**Key Insight**: V2 essentially collapsed to always predicting BUY (76% recall) and ignoring SELL (4% recall).

---

## 💰 Trading Metrics

### Critical Trading Performance

| Metric | V1 | V2 | V3 | Best |
|--------|-----|-----|-----|------|
| **Information Coefficient** | 0.034 | 0.028 | 0.015 | V1 |
| **IC p-value** | **0.025** ✅ | 0.066 | 0.320 ❌ | V1 |
| **Directional Accuracy** | 52.10% | 49.48% | 52.03% | V1 |
| **Sharpe Ratio** | -0.18 | **-0.03** | -0.77 | V2 |
| **Annualized Return** | -11.71% | **-1.73%** | -47.11% | V2 |

```mermaid
xychart-beta
    title "Trading Metrics Comparison"
    x-axis ["V1", "V2", "V3"]
    y-axis "Value" -50 --> 55
    bar "Directional Acc (%)" [52.1, 49.5, 52.0]
    bar "Ann. Return (%)" [-11.7, -1.7, -47.1]
```

### Sharpe Ratio Comparison

```mermaid
xychart-beta
    title "Simulated Sharpe Ratio (Top 30% Confidence)"
    x-axis ["V1", "V2", "V3"]
    y-axis "Sharpe" -0.8 --> 0.1
    bar "Sharpe" [-0.18, -0.03, -0.77]
```

⚠️ **All models have negative Sharpe ratios** - none would be profitable for trading.

### Statistical Significance

```mermaid
flowchart TD
    subgraph IC["Information Coefficient Analysis"]
        V1IC["V1: IC = 0.034<br/>p = 0.025 ✅"]
        V2IC["V2: IC = 0.028<br/>p = 0.066 ⚠️"]
        V3IC["V3: IC = 0.015<br/>p = 0.320 ❌"]
    end
    
    V1IC --> SIG["Statistically Significant<br/>(p < 0.05)"]
    V2IC --> MARGINAL["Marginally Significant<br/>(p < 0.10)"]
    V3IC --> NOSIG["Not Significant<br/>(p > 0.10)"]
```

---

## 🔬 Key Findings

### The Accuracy-Trading Paradox

```mermaid
flowchart LR
    subgraph Problem["The Problem"]
        A[Better Classification] -->|Does NOT equal| B[Better Trading]
    end
    
    subgraph Evidence["Evidence"]
        C[V3 has BEST accuracy<br/>41.92%]
        D[V3 has WORST Sharpe<br/>-0.77]
        C --> E[But...]
        E --> D
    end
```

### Model Behavior Analysis

```mermaid
mindmap
  root((Model Analysis))
    V1
      Balanced recall
      Significant IC
      Moderate overfitting
      Best for production
    V2
      No overfitting
      Class collapse
      BUY bias 76%
      Least bad Sharpe
    V3
      Best accuracy
      Worst trading
      Non-significant IC
      Fast training
```

---

## 📋 Detailed Results

### V1: Full Fine-tuning

**Configuration:**
```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 5 \
    --temporal-split \
    --evaluate-test \
    --early-stopping-patience 2 \
    --buy-weight-boost 1.2 \
    --output-dir models/all-tweets
```

**Training Time:** 1h 40m 54s

**Epoch-by-Epoch Validation:**
| Epoch | Val Loss | Val Accuracy | Val F1 (macro) |
|-------|----------|--------------|----------------|
| 1 | 1.070 | 35.2% | 35.1% |
| 2 | 1.099 | 34.9% | 32.9% |
| 3 | 1.145 | 38.3% | 37.6% |
| 4 | 1.246 | 37.9% | 37.2% |
| 5 | 1.339 | 39.5% | 38.3% |

---

### V2: Frozen BERT

**Configuration:**
```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 5 \
    --temporal-split \
    --evaluate-test \
    --early-stopping-patience 2 \
    --buy-weight-boost 1.2 \
    --output-dir models/all-tweets-v2 \
    --freeze-bert \
    --dropout 0.5
```

**Training Time:** 36m 36s (3x faster due to frozen BERT)

**Epoch-by-Epoch Validation:**
| Epoch | Val Loss | Val Accuracy | Val F1 (macro) |
|-------|----------|--------------|----------------|
| 1 | 1.095 | 33.4% | 26.9% |
| 2 | 1.087 | 32.9% | 26.8% |
| 3 | 1.079 | 33.4% | 27.6% |
| 4 | 1.076 | 32.9% | 27.1% |
| 5 | 1.077 | 33.0% | 27.1% |

---

### V3: Fast Training

**Configuration:**
```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 3 \
    --temporal-split \
    --evaluate-test \
    --early-stopping-patience 2 \
    --buy-weight-boost 1.2 \
    --output-dir models/all-tweets-v3 \
    --dropout 0.2 \
    --learning-rate 3e-5
```

**Training Time:** 58m 43s

**Epoch-by-Epoch Validation:**
| Epoch | Val Loss | Val Accuracy | Val F1 (macro) |
|-------|----------|--------------|----------------|
| 1 | 1.089 | 34.0% | 32.8% |
| 2 | 1.130 | 34.8% | 31.9% |
| 3 | 1.161 | 38.8% | 38.2% |

---

## 🎯 Recommendations

### For Production Use

```mermaid
flowchart TD
    A{Use Case?} -->|Classification| B[Use V3]
    A -->|Trading Signals| C[Use V1]
    A -->|Low Risk| D[Use V2]
    
    B --> B1[Highest accuracy<br/>41.92%]
    C --> C1[Only significant IC<br/>p=0.025]
    D --> D1[Least negative Sharpe<br/>-0.03]
    
    C1 --> WARN[⚠️ Still not profitable]
    D1 --> WARN
```

### Future Improvements

1. **Address Overfitting**
   - Use early stopping based on validation loss (stop at epoch 1-2)
   - Add more regularization (weight decay, label smoothing)

2. **Improve Trading Signal**
   - Train on return prediction (regression) instead of classification
   - Use asymmetric loss functions
   - Focus on high-confidence predictions only (>70%)

3. **Data Quality**
   - Review how SELL/HOLD/BUY labels are generated
   - Consider different return thresholds
   - Add more features (sentiment, market context)

---

## 📁 Model Artifacts

| Model | Path | Size |
|-------|------|------|
| V1 | `models/all-tweets/` | ~440 MB |
| V2 | `models/all-tweets-v2/` | ~440 MB |
| V3 | `models/all-tweets-v3/` | ~440 MB |

Each directory contains:
- `pytorch_model.bin` - Model weights
- `model_config.json` - Configuration
- `scaler.joblib` - Feature scaler
- `encodings.json` - Categorical encodings
- `evaluation/` - Metrics and confusion matrix

---

*Generated: December 23, 2025*

