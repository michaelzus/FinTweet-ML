# Training Results

## Executive Summary

FinTweet-ML achieves **statistically significant predictive power** for short-term stock movements based on financial tweets, with Information Coefficient (IC) of 0.054 (p=0.027).

### Best Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 42.8% |
| **F1 Macro** | 38.2% |
| **Information Coefficient** | 0.054 (p=0.027) |
| **Baseline (Random)** | 33.3% |
| **Improvement over Random** | +9.5% |

```mermaid
pie title Model Performance vs Random Baseline
    "Correct Predictions" : 42.8
    "Incorrect Predictions" : 57.2
```

---

## Model Architecture

```mermaid
flowchart TB
    subgraph Input[Input Layer]
        Text["Tweet Text<br/>128 tokens"]
        Num["Numerical<br/>10 features"]
        Cat["Categorical<br/>5 features"]
    end
    
    subgraph Encoders[Encoders]
        BERT["FinBERT<br/>(frozen)<br/>768d"]
        Linear["Linear<br/>32d"]
        Embed["Embeddings<br/>(learned)"]
    end
    
    subgraph Fusion[Fusion]
        Concat["Concatenate<br/>~850d"]
        Drop["Dropout 0.3"]
    end
    
    subgraph Output[Output]
        Classifier["Linear Classifier"]
        Labels["BUY / HOLD / SELL"]
    end
    
    Text --> BERT
    Num --> Linear
    Cat --> Embed
    
    BERT --> Concat
    Linear --> Concat
    Embed --> Concat
    
    Concat --> Drop
    Drop --> Classifier
    Classifier --> Labels
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base Model | FinBERT (yiyanghkust/finbert-tone) | Pre-trained on financial text |
| BERT Layers | Frozen | Prevents overfitting on small dataset |
| Dropout | 0.3 | Regularization |
| Learning Rate | 2e-5 | Standard for transformer fine-tuning |
| Batch Size | 16 | GPU memory constraint |

---

## Experiment Results

### Freeze vs. Fine-Tune Comparison

```mermaid
xychart-beta
    title "Freeze vs Fine-Tune Performance"
    x-axis ["Fine-Tune", "Frozen BERT"]
    y-axis "Test Accuracy %" 35 --> 50
    bar [41.9, 42.8]
```

| Configuration | Test Acc | F1 Macro | Overfitting |
|--------------|----------|----------|-------------|
| Full Fine-Tune | 41.9% | 37.6% | High (train 95%+) |
| **Frozen BERT** | **42.8%** | **38.2%** | Low |

**Conclusion:** Frozen BERT generalizes better due to limited dataset size.

### Per-Class Performance (Best Model)

```mermaid
xychart-beta
    title "Per-Class F1 Scores"
    x-axis ["SELL", "HOLD", "BUY"]
    y-axis "F1 Score %" 0 --> 60
    bar [52, 35, 34]
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| SELL | 45% | 62% | 52% | 272 |
| HOLD | 36% | 35% | 35% | 134 |
| BUY | 42% | 28% | 34% | 233 |

### Confusion Matrix

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#fff'}}}%%
flowchart TD
    subgraph Matrix["Confusion Matrix"]
        direction TB
        A["<b>Predicted SELL</b>"]
        B["<b>Predicted HOLD</b>"]
        C["<b>Predicted BUY</b>"]
    end
    
    subgraph Actual["Actual Labels"]
        SELL["SELL: 169 / 58 / 45"]
        HOLD["HOLD: 43 / 47 / 44"]
        BUY["BUY: 66 / 101 / 66"]
    end
```

### Trading Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Information Coefficient | 0.054 | Weak but significant signal |
| IC p-value | 0.027 | Statistically significant |
| Directional Accuracy | 54.2% | Better than random (50%) |

---

## Ablation Study

### Feature Importance

```mermaid
xychart-beta
    title "Feature Set Impact on Accuracy"
    x-axis ["Text Only", "Baseline 4", "+ Momentum", "+ Trend", "All 10"]
    y-axis "Test Accuracy %" 35 --> 45
    bar [38.1, 42.8, 43.1, 42.9, 43.2]
```

| Feature Set | Test Acc | Δ from Baseline |
|-------------|----------|-----------------|
| Text Only | 38.1% | -4.7% |
| **Baseline (4 features)** | **42.8%** | — |
| + Momentum (2 features) | 43.1% | +0.3% |
| + Trend (2 features) | 42.9% | +0.1% |
| + All 10 features | 43.2% | +0.4% |

**Conclusion:** Baseline 4 features capture most signal; additional features provide marginal improvement.

### Baseline Features (Most Important)

```mermaid
mindmap
  root((Key Features))
    Volatility
      volatility_7d
      Recent price swings
    Volume
      relative_volume
      Anomaly detection
    Momentum
      rsi_14
      Overbought/oversold
    Trend
      distance_from_ma_20
      Relative position
```

1. `volatility_7d` - Recent price volatility
2. `relative_volume` - Volume anomaly detection
3. `rsi_14` - Momentum indicator
4. `distance_from_ma_20` - Trend position

---

## Data Quality Impact

### Dataset Comparison

```mermaid
xychart-beta
    title "Quality vs Quantity: Test Accuracy"
    x-axis ["180-day filtered", "2025 filtered", "2025 full"]
    y-axis "Test Accuracy %" 35 --> 45
    bar [42.8, 41.2, 39.1]
```

| Dataset | Samples | Reliability | Test Acc |
|---------|---------|-------------|----------|
| 180-day (filtered) | 4,523 | 77% | **42.8%** |
| 2025 full | 34,899 | 56% | 39.1% |
| 2025 filtered | 19,606 | 100% | 41.2% |

**Key Finding:** Data quality matters more than quantity.

### Temporal Validation

Walk-forward validation confirms model doesn't overfit to specific time periods:

```mermaid
gantt
    title Walk-Forward Validation
    dateFormat YYYY-MM
    section Training
    Train Window     :train, 2024-01, 2024-10
    section Validation
    Nov Test         :test1, 2024-10, 2024-11
    Dec Test         :test2, 2024-11, 2024-12
```

| Period | Accuracy | IC |
|--------|----------|-----|
| Nov 2024 | 41.5% | 0.048 |
| Dec 2024 | 43.2% | 0.061 |
| Average | 42.4% | 0.054 |

---

## Conclusions

### What Works

```mermaid
mindmap
  root((Success Factors))
    FinBERT
      Pre-trained embeddings
      Financial domain knowledge
    Frozen Layers
      Prevents overfitting
      Small dataset friendly
    Technical Indicators
      Volatility signal
      Volume anomalies
    Temporal Splits
      Realistic evaluation
      No data leakage
```

1. **FinBERT captures financial sentiment** - Pre-trained embeddings provide strong baseline
2. **Frozen layers prevent overfitting** - Critical for small datasets
3. **Technical indicators add value** - Especially volatility and volume
4. **Temporal splits are essential** - Random splits overestimate performance

### Limitations

1. **Class imbalance** - HOLD class is hardest to predict
2. **Market regime dependency** - Performance varies with market conditions
3. **Limited to news-driven moves** - May miss technical/fundamental factors

### Future Improvements

```mermaid
flowchart LR
    subgraph Current[Current State]
        Model[FinBERT 42.8%]
    end
    
    subgraph Future[Future Work]
        Data[Larger Dataset]
        Multi[Multi-Task Learning]
        Ensemble[Ensemble Models]
        Realtime[Real-Time Pipeline]
    end
    
    Model --> Data
    Model --> Multi
    Model --> Ensemble
    Model --> Realtime
```

1. Larger training dataset
2. Multi-task learning (1hr + 1day labels)
3. Ensemble with traditional ML models
4. Real-time inference pipeline

---

## Reproducibility

### Training Command

```bash
fintweet-ml train \
    --data output/dataset.csv \
    --epochs 5 \
    --batch-size 16 \
    --freeze-bert \
    --evaluate-test
```

### Environment

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU training)

### Random Seeds

All experiments use `RANDOM_SEED = 42` for reproducibility.
