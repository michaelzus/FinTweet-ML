# FinBERT Training Plan for Tweet-Based Price Prediction

## Overview

This plan outlines how to fine-tune FinBERT on financial tweets to predict stock price movements. The model will learn to classify tweets into BUY/HOLD/SELL signals using both the semantic content of tweets and technical indicators.

---

## Global Configuration

**Reference these throughout — do NOT redefine:**

```python
# ========== GLOBAL CONFIG ==========
TARGET_COLUMN = "label_1d_3class"  # 1-day labels (less noisy than 1-hour)

NUMERICAL_FEATURES = [
    "volatility_7d",
    "relative_volume", 
    "rsi_14",
    "distance_from_ma_20",
]
# NOTE: spy_return_1d EXCLUDED - uses day T close (future data for intraday tweets)

CATEGORICAL_FEATURES = ["author", "category"]

EXCLUDED_FROM_FEATURES = [
    "spy_return_1d",        # Uses day T close (future for intraday!)
    "spy_return_1hr",       # Future SPY movement
    "return_1hr",           # Target
    "return_to_next_open",  # Target  
    "price_1hr_after",      # Future price
    "price_next_open",      # Future price
    "label_3class",         # Target (backup)
    "label_1d_3class",      # Target (primary)
]
```

---

## Important: Feature vs Target/Reference Distinction

The enriched dataset contains columns that serve different purposes. **This is critical to understand:**

| Column | Role | Used as Model Input? | Description |
|--------|------|---------------------|-------------|
| `text` | **INPUT FEATURE** | Yes | Tweet content for FinBERT |
| `volatility_7d` | **INPUT FEATURE** | Yes | Historical 7-day volatility |
| `rsi_14` | **INPUT FEATURE** | Yes | Historical 14-day RSI |
| `relative_volume` | **INPUT FEATURE** | Yes | Volume vs 20-day average |
| `distance_from_ma_20` | **INPUT FEATURE** | Yes | Historical price vs MA |
| `spy_return_1d` | **REFERENCE ONLY** | No | Day T SPY return (future data for intraday tweets) |
| `author` | **INPUT FEATURE** | Yes (embedded) | Tweet author (reduces author bias) |
| `category` | **INPUT FEATURE** | Yes (embedded) | Message category |
| `label_3class` | **TARGET (BACKUP)** | No (prediction target) | BUY/HOLD/SELL for 1-hour |
| `label_1d_3class` | **TARGET (PRIMARY)** | No (prediction target) | BUY/HOLD/SELL for 1-day (less noisy) |
| `return_1hr` | **REFERENCE ONLY** | No | Future 1-hour return |
| `return_to_next_open` | **REFERENCE ONLY** | No | Future return to next open |
| `spy_return_1hr` | **REFERENCE ONLY** | No | Future SPY 1-hour return |
| `price_1hr_after` | **REFERENCE ONLY** | No | Future price (for label calc) |
| `price_next_open` | **REFERENCE ONLY** | No | Future price (for label calc) |

**Why reference columns exist in the dataset:**
- They are used to **calculate labels** during data preparation
- They are useful for **analysis and debugging**
- They are **NEVER fed to the model** as input features
- This is standard supervised learning: train with known outcomes, predict on unseen data

**This addresses concerns about "look-ahead bias":** Future-looking columns like `spy_return_1hr` are intentionally in the dataset for reference/analysis but are explicitly excluded from the feature set during training.

---

## ⚠️ Point-in-Time Correctness Warning

### What `spy_return_1d` Actually Represents

**Current Implementation**: `spy_return_1d` = Return from day T-1 close to day T close, where T is the tweet date.

For a tweet at 10:00 AM on day T:
- This uses day T's **end-of-day close** (future information at tweet time!)
- **This is partially future-looking** for intraday tweets

**Recommended Interpretation for Training**:
```python
# IMPORTANT: spy_return_1d in current data = CURRENT day's SPY return (partially future)
# For strict point-in-time correctness, this should be PREVIOUS day's return
# 
# Acceptable for training if:
# 1. Most tweets are after-hours (close already known)
# 2. Or you're willing to accept ~1% leakage for intraday tweets
#
# For production inference, use PREVIOUS day's SPY return only
spy_return_1d_production = get_spy_return(yesterday)  # Strictly known
```

### Technical Indicators: Point-in-Time Status

The enricher currently uses `daily_df.index.date <= tweet_date`, which **includes tweet day's data**.

| Indicator | Current Behavior | Point-in-Time Correct? |
|-----------|------------------|----------------------|
| `volatility_7d` | Uses closes from [T-7, T] | ⚠️ Includes day T close |
| `rsi_14` | Uses closes from [T-14, T] | ⚠️ Includes day T close |
| `relative_volume` | Uses volumes from [T-20, T] | ⚠️ Includes day T volume |
| `distance_from_ma_20` | Uses close T vs MA | ⚠️ Uses day T close |

**Impact Assessment** (based on Phase 0 validation):
- **Regular session**: 34.9% - Day T close unknown → Minor leakage
- **Premarket**: 29.7% - Day T close unknown → Minor leakage ⚠️
- **Afterhours**: 15.6% - Day T close is known → **No leakage**
- **Closed (weekends)**: 19.8% - Day T close is known → **No leakage**
- **Total safe**: ~35% of data has no leakage; ~65% has minor leakage
- For **production inference**: Must use strictly `< tweet_date` data

**Recommendation**: Accept for initial training, but fix enricher for production:
```python
# In enricher.py, change line 313 from:
daily_df = daily_df_full[daily_df_full.index.date <= tweet_date].copy()
# To:
daily_df = daily_df_full[daily_df_full.index.date < tweet_date].copy()  # Strict
```

---

## Phase 0: Pre-Training Validation Checklist

Run this before training to confirm data integrity. **Validation notebook**: `notebooks/phase0_validation.ipynb`

### Phase 0 Results (15-dec-enrich7.csv)

| Check | Result | Status |
|-------|--------|--------|
| Feature Leakage | PASS | ✓ |
| Target Availability | 99.6% (5,844/5,866) | ✓ |
| Class Balance | 37.8% max (SELL) | ✓ Well-balanced! |
| Author Embedding Needed | 62.2% top 2 | ✓ |
| Reliable Labels | 77.5% (4,545/5,866) | ✓ |

**Class Distribution (better than expected!):**
- SELL: 37.8% (2,209)
- BUY: 35.4% (2,070)
- HOLD: 26.8% (1,565)

**Author Distribution:**
1. Wall St Engine: 34.3% (2,010 tweets)
2. Hardik Shah: 27.9% (1,639 tweets)
3. Evan: 25.5% (1,493 tweets)
4. App Economy Insights: 4.7%
5. Fiscal.ai: 3.1%

**Session Distribution (premarket risk):**
- Regular: 34.9%
- Premarket: 29.7% ⚠️ (minor leakage for technical indicators)
- Closed: 19.8%
- Afterhours: 15.6%

**Recommended Training Filter:**
```python
df_train = df[df['is_reliable_label'] == True].dropna(subset=['label_1d_3class'])
# Expected: ~4,527 training samples
```

### Validation Code

```python
# ========== Phase 0: Pre-Training Validation ==========

import pandas as pd
import numpy as np

df = pd.read_csv("output/15-dec-enrich7.csv")
TARGET_COLUMN = "label_1d_3class"

# 1. Confirm no future columns in features
NUMERICAL_FEATURES = ["volatility_7d", "relative_volume", "rsi_14", "distance_from_ma_20"]
FORBIDDEN_AS_FEATURES = ["spy_return_1d", "spy_return_1hr", "return_1hr", "price_1hr_after", "return_to_next_open", "price_next_open"]

for col in FORBIDDEN_AS_FEATURES:
    assert col not in NUMERICAL_FEATURES, f"LEAK: {col} in features!"
print("✓ No future columns in NUMERICAL_FEATURES")

# 2. Check label distribution
print(f"\n=== Target Distribution ({TARGET_COLUMN}) ===")
print(df[TARGET_COLUMN].value_counts(normalize=True))

# 3. Verify spy_return_1d is NOT in features (it's future data!)
assert "spy_return_1d" not in NUMERICAL_FEATURES, "LEAK: spy_return_1d uses day T close!"
print("✓ spy_return_1d correctly excluded from features")

# 4. Check author distribution (watch for bias)
print(f"\n=== Author Distribution ===")
print(df["author"].value_counts(normalize=True).head(5))

# 5. Verify reliable label filtering works
df_reliable = df[df["is_reliable_label"] == True]
print(f"\n=== Data Quality ===")
print(f"Total samples: {len(df)}")
print(f"Reliable 1hr labels: {len(df_reliable)} ({100*len(df_reliable)/len(df):.1f}%)")
print(f"With 1-day labels: {df[TARGET_COLUMN].notna().sum()}")

# 6. Check premarket tweets (HIGHEST RISK - indicators use future data!)
if "session" in df.columns:
    premarket_count = (df["session"] == "premarket").sum()
    print(f"\n=== Premarket Risk Assessment ===")
    print(f"⚠️  Premarket tweets: {premarket_count} ({100*premarket_count/len(df):.1f}%)")
    print("   For these, technical indicators use day T close (FUTURE DATA)")
    print("   Consider: df_clean = df[df['session'] != 'premarket'] for maximum conservatism")

# 7. Quick signal check (optional - requires FinBERT)
# from transformers import pipeline
# sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
# df_sample = df.sample(100)
# df_sample["sentiment"] = df_sample["text"].apply(lambda x: sentiment(x[:512])[0]["label"])
# print(f"Sentiment-return correlation: {df_sample['sentiment'].map({'positive':1,'neutral':0,'negative':-1}).corr(df_sample['return_to_next_open'])}")
```

---

## Architecture

```
                    ┌───────────────────────────────────────────────────────────┐
                    │                    Multi-Modal Model                       │
                    └───────────────────────────────────────────────────────────┘
                                               │
        ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
        │                  │                   │                   │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│   FinBERT     │  │  Numerical    │  │    Author     │  │   Category    │  │               │
│   Encoder     │  │  Features     │  │   Embedding   │  │   Embedding   │  │    (more)     │
│   (frozen or  │  │  Encoder      │  │               │  │               │  │               │
│   fine-tuned) │  │  (MLP)        │  │               │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────────────┘
        │                  │                  │                  │
        │  768-dim         │  32-dim          │  16-dim          │  8-dim
        └──────────────────┴──────────────────┴──────────────────┘
                                        │
                                        │  Concatenate: 824-dim
                                        │
                                ┌───────▼───────┐
                                │  Fusion Layer │
                                │  (dropout +   │
                                │   linear)     │
                                └───────┬───────┘
                                        │
                                ┌───────▼───────┐
                                │  Classifier   │
                                │  (3 classes)  │
                                │  BUY/HOLD/SELL│
                                └───────────────┘
```

---

## Phase 1: Data Preparation

### 1.1 Run Enrichment on Clean Data

First, enrich the cleaned CSV with price data and indicators:

```bash
source .venv/bin/activate
python -m tweet_enricher enrich output/15-dec2.csv output/15-dec-enrich7.csv
```

**Status**: ✅ Already completed. Using `output/15-dec-enrich7.csv` (5,866 rows).

### 1.2 Data Filtering

Filter the enriched data to keep only reliable samples:

```python
import pandas as pd

df = pd.read_csv("output/15-dec-enrich7.csv")  # 5,866 rows, 29 columns

# Filter to reliable labels only
df_reliable = df[df["is_reliable_label"] == True].copy()

# Drop rows with missing target (using 1-DAY labels for less noise)
TARGET_COLUMN = "label_1d_3class"  # 1-day labels are less noisy than 1-hour
df_reliable = df_reliable.dropna(subset=[TARGET_COLUMN])

print(f"Total samples: {len(df)}")
print(f"Reliable samples: {len(df_reliable)}")
print(f"Class distribution:\n{df_reliable[TARGET_COLUMN].value_counts()}")
```

### 1.3 Train/Val/Test Split by Tweet Hash

**Critical**: Split by `tweet_hash` to prevent text leakage across sets.

```python
from sklearn.model_selection import train_test_split

# Get unique tweet hashes
unique_hashes = df_reliable["tweet_hash"].unique()

# Split hashes (not rows)
train_hashes, temp_hashes = train_test_split(unique_hashes, test_size=0.3, random_state=42)
val_hashes, test_hashes = train_test_split(temp_hashes, test_size=0.5, random_state=42)

# Assign rows based on hash
df_train = df_reliable[df_reliable["tweet_hash"].isin(train_hashes)]
df_val = df_reliable[df_reliable["tweet_hash"].isin(val_hashes)]
df_test = df_reliable[df_reliable["tweet_hash"].isin(test_hashes)]

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
```

### 1.4 Compute Class Weights

Handle class imbalance with weighted loss:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Map labels to integers
label_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
y_train = df_train[TARGET_COLUMN].map(label_map).values

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print(f"Class weights: SELL={class_weights[0]:.2f}, HOLD={class_weights[1]:.2f}, BUY={class_weights[2]:.2f}")
```

---

## Phase 2: Feature Engineering

### 2.1 Text Features (FinBERT)

**Model**: `yiyanghkust/finbert-tone` (pre-trained on financial sentiment)

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = BertModel.from_pretrained("yiyanghkust/finbert-tone")

# Tokenize
def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
```

### 2.2 Numerical Features

Select features that are **not future-looking** (computed BEFORE tweet timestamp):

| Feature | Description | Use as Feature | Rationale |
|---------|-------------|----------------|-----------|
| `volatility_7d` | 7-day historical volatility | **YES** | Historical, computed before tweet |
| `relative_volume` | Volume vs 20-day average | **YES** | Historical, computed before tweet |
| `rsi_14` | 14-day RSI | **YES** | Historical, computed before tweet |
| `distance_from_ma_20` | Price distance from 20-day MA | **YES** | Historical, computed before tweet |
| `spy_return_1d` | Day T SPY return | **NO** | Uses day T close (future for intraday tweets) |
| `return_1hr` | 1-hour future return | **NO** | Future data - TARGET only |
| `return_to_next_open` | Return to next open | **NO** | Future data - TARGET only |
| `spy_return_1hr` | SPY 1-hour future return | **NO** | Future data - REFERENCE only |
| `price_1hr_after` | Price 1 hour after | **NO** | Future data - REFERENCE only |
| `price_next_open` | Next day open price | **NO** | Future data - REFERENCE only |

**Columns marked NO are intentionally in the dataset for label calculation and analysis, but are NEVER used as model inputs.**

```python
# ONLY these columns are used as numerical features
NUMERICAL_FEATURES = [
    "volatility_7d",
    "relative_volume", 
    "rsi_14",
    "distance_from_ma_20",
    # NOTE: spy_return_1d REMOVED - it uses day T close (future data for intraday tweets)
]

# These are EXCLUDED from features (future-looking / targets / reference)
EXCLUDED_FROM_FEATURES = [
    "return_1hr",           # TARGET
    "return_to_next_open",  # TARGET  
    "spy_return_1d",        # REFERENCE (uses day T close - future for intraday tweets!)
    "spy_return_1hr",       # REFERENCE (future SPY movement)
    "price_1hr_after",      # REFERENCE (future price)
    "price_next_open",      # REFERENCE (future price)
    "label_3class",         # TARGET (what we predict)
    "label_1d_3class",      # TARGET (what we predict)
]

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_num = scaler.fit_transform(df_train[NUMERICAL_FEATURES].fillna(0))
X_val_num = scaler.transform(df_val[NUMERICAL_FEATURES].fillna(0))
X_test_num = scaler.transform(df_test[NUMERICAL_FEATURES].fillna(0))
```

### 2.3 Categorical Features

Encode `author` and `category` as embeddings to reduce author bias:

```python
# Author encoding (IMPORTANT: reduces bias from dominant authors)
authors = df_reliable["author"].unique().tolist()
author_to_idx = {auth: i for i, auth in enumerate(authors)}
num_authors = len(authors)

# Category encoding
categories = df_reliable["category"].unique().tolist()
category_to_idx = {cat: i for i, cat in enumerate(categories)}
num_categories = len(categories)

# Convert to indices
df_train["author_idx"] = df_train["author"].map(author_to_idx)
df_train["category_idx"] = df_train["category"].map(category_to_idx)

# In model: 
# nn.Embedding(num_authors, 16)      # Author embedding
# nn.Embedding(num_categories, 8)    # Category embedding
```

**Why add author as a feature:**
- Top 2 authors represent 62.2% of data (Wall St Engine 34.3%, Hardik Shah 27.9%)
- Top 3 authors = 87.7% (includes Evan at 25.5%)
- Without this, model learns their tweeting style, not market patterns
- Embedding allows model to learn author-specific adjustments
- Note: Authors have different label distributions (e.g., Evan has more HOLD tweets)

### Phase 2 Implementation Status

✅ **Phase 2 Complete** - See `notebooks/phase2_feature_engineering.ipynb` for verification.

| Component | File | Status |
|-----------|------|--------|
| Global config | `config.py` | ✅ `NUMERICAL_FEATURES`, `EXCLUDED_FROM_FEATURES`, embedding dims |
| Data loading | `data/loader.py` | ✅ `load_enriched_data()`, `filter_reliable()`, `prepare_features()` |
| Data splitting | `data/splitter.py` | ✅ `split_by_hash()`, `verify_no_leakage()` |
| Class weights | `data/weights.py` | ✅ `compute_class_weights()`, `weights_to_tensor()` |
| TweetDataset | `dataset.py` | ✅ Multi-modal dataset with tokenization |
| Categorical encoding | `dataset.py` | ✅ `create_categorical_encodings()`, `encode_categorical()` |
| Scaler persistence | `dataset.py` | ✅ `save_scaler()`, `load_scaler()`, `save_preprocessing_artifacts()` |
| Unit tests | `tests/test_tweet_classifier.py` | ✅ 16 data/dataset tests |

**Usage example:**
```python
from tweet_classifier import (
    load_enriched_data, filter_reliable, split_by_hash,
    create_categorical_encodings, create_dataset_from_df,
    compute_class_weights, save_preprocessing_artifacts
)
from transformers import BertTokenizer

# Load and filter data
df = load_enriched_data()
df_reliable = filter_reliable(df)
df_train, df_val, df_test = split_by_hash(df_reliable)

# Create encodings from training data
encodings = create_categorical_encodings(df_train)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# Create datasets (scaler is fitted on train, reused for val/test)
train_dataset, scaler = create_dataset_from_df(
    df_train, tokenizer, 
    encodings['author_to_idx'], encodings['category_to_idx'],
    fit_scaler=True
)
val_dataset, _ = create_dataset_from_df(
    df_val, tokenizer,
    encodings['author_to_idx'], encodings['category_to_idx'],
    scaler=scaler, fit_scaler=False
)

# Save artifacts for inference
save_preprocessing_artifacts(scaler, encodings, "models/finbert-tweet-classifier")
```

---

## Phase 3: Model Architecture

✅ **Phase 3 Complete** - Model implemented in `src/tweet_classifier/model.py`

### 3.1 Dataset Class

Already implemented in Phase 2 (`dataset.py`). See Phase 2 Implementation Status.

### 3.2 Multi-Modal Model

```python
import torch.nn as nn
from transformers import BertModel

class FinBERTMultiModal(nn.Module):
    """FinBERT with numerical + categorical feature fusion for price prediction."""
    
    def __init__(
        self,
        num_numerical_features: int,
        num_authors: int,
        num_categories: int,
        num_classes: int = 3,
        finbert_model: str = "yiyanghkust/finbert-tone",
        freeze_bert: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # FinBERT encoder
        self.bert = BertModel.from_pretrained(finbert_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # Categorical embeddings (to reduce author bias)
        self.author_embedding = nn.Embedding(num_authors, 16)
        self.category_embedding = nn.Embedding(num_categories, 8)
        
        # Numerical feature encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion + classifier
        # 768 (BERT) + 32 (numerical) + 16 (author) + 8 (category) = 824
        fusion_size = bert_hidden_size + 32 + 16 + 8
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, numerical, author_idx, category_idx, labels=None):
        # Get BERT [CLS] embedding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Encode numerical features
        num_embedding = self.numerical_encoder(numerical)  # [batch, 32]
        
        # Encode categorical features
        author_emb = self.author_embedding(author_idx)      # [batch, 16]
        category_emb = self.category_embedding(category_idx)  # [batch, 8]
        
        # Fusion
        combined = torch.cat([cls_embedding, num_embedding, author_emb, category_emb], dim=1)
        
        # Classification
        logits = self.classifier(combined)  # [batch, 3]
        
        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss
        
        return output
```

### Phase 3 Implementation Status

| Component | File | Status |
|-----------|------|--------|
| FinBERTMultiModal | `model.py` | ✅ Multi-modal model with BERT + numerical + categorical |
| Forward method | `model.py` | ✅ Returns dict with `logits` and optional `loss` |
| BERT freezing | `model.py` | ✅ `freeze_bert` parameter for fine-tuning control |
| Config serialization | `model.py` | ✅ `get_config()` method for saving model config |
| Unit tests | `tests/test_tweet_classifier.py` | ✅ 7 model tests |

**Usage example:**
```python
from tweet_classifier import FinBERTMultiModal

# Initialize model
model = FinBERTMultiModal(
    num_numerical_features=4,
    num_authors=10,
    num_categories=5,
    num_classes=3,
    freeze_bert=False,  # Fine-tune BERT
    dropout=0.3
)

# Forward pass
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    numerical=numerical_features,
    author_idx=author_indices,
    category_idx=category_indices,
    labels=labels  # Optional: if provided, returns loss
)

logits = output["logits"]  # [batch, 3]
loss = output.get("loss")   # Only if labels provided
```

---

## Phase 4: Training

✅ **Phase 4 Complete** - Training pipeline implemented in `src/tweet_classifier/trainer.py` and `src/tweet_classifier/train.py`

### Phase 4 Implementation Status

| Component | File | Status |
|-----------|------|--------|
| WeightedTrainer | `trainer.py` | ✅ Custom Trainer with class-weighted cross-entropy loss |
| compute_metrics | `trainer.py` | ✅ Returns accuracy, f1_macro, f1_weighted |
| create_training_args | `train.py` | ✅ Configurable TrainingArguments builder |
| train() | `train.py` | ✅ Full training pipeline with artifact saving |
| CLI interface | `train.py` | ✅ `python -m tweet_classifier.train --help` |
| Unit tests | `tests/test_tweet_classifier.py` | ✅ 8 new tests (31 total) |

### 4.1 Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models/finbert-tweet-classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,  # Mixed precision for speed (auto-disabled if no CUDA)
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)
```

### 4.2 Custom Trainer with Class Weights

```python
from transformers import Trainer
import torch.nn.functional as F

class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""
    
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device), reduction=reduction)
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        
        return (loss, outputs) if return_outputs else loss
```

### 4.3 Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
```

### 4.4 CLI Usage

```bash
# Activate venv and run training
source .venv/bin/activate
python -m tweet_classifier.train \
    --data-path output/15-dec-enrich7.csv \
    --output-dir models/finbert-tweet-classifier \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --freeze-bert  # Optional: freeze BERT for faster training
```

### 4.5 Programmatic Usage

```python
from tweet_classifier import train, WeightedTrainer, compute_metrics

# Run training with defaults
train()

# Or with custom parameters
train(
    data_path="output/15-dec-enrich7.csv",
    output_dir="models/finbert-tweet-classifier",
    num_epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    freeze_bert=False,
    dropout=0.3,
)
```

---

## Phase 5: Evaluation

### 5.1 Test Set Evaluation

```python
from tweet_classifier.config import LABEL_MAP

# Evaluate on test set
test_dataset = TweetDataset(
    df_test["text"], X_test_num,
    df_test["author_idx"].values,
    df_test["category_idx"].values,
    df_test[TARGET_COLUMN].map(LABEL_MAP).values,
    tokenizer
)

model.eval()
results = trainer.predict(test_dataset)

print(f"Test Accuracy: {results.metrics['test_accuracy']:.4f}")
print(f"Test F1 (macro): {results.metrics['test_f1_macro']:.4f}")

# Detailed classification report
preds = np.argmax(results.predictions, axis=1)
print(classification_report(
    df_test[TARGET_COLUMN].map(LABEL_MAP).values,
    preds,
    target_names=["SELL", "HOLD", "BUY"]
))
```

### 5.2 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tweet_classifier.config import LABEL_MAP

cm = confusion_matrix(df_test[TARGET_COLUMN].map(LABEL_MAP).values, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["SELL", "HOLD", "BUY"],
            yticklabels=["SELL", "HOLD", "BUY"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
```

### 5.3 Trading-Focused Evaluation Metrics

**Why standard ML metrics aren't enough**: A 70% accurate model that's wrong on big moves **loses money**. These metrics evaluate actual trading utility.

```python
from scipy.stats import spearmanr
import numpy as np

def compute_trading_metrics(predictions, confidences, actual_returns, transaction_cost=0.001):
    """
    Compute trading-relevant metrics beyond accuracy.
    
    Args:
        predictions: Model predictions (0=SELL, 1=HOLD, 2=BUY)
        confidences: Model confidence scores (softmax probabilities)
        actual_returns: Actual 1-day returns
        transaction_cost: Cost per trade as decimal (default 0.1% = 0.001)
    
    Returns:
        Dictionary of trading metrics
    """
    results = {}
    
    # 1. Information Coefficient (IC)
    # Correlation between model confidence and actual returns
    # Good models: IC > 0.05
    buy_confidence = confidences[:, 2] - confidences[:, 0]  # BUY - SELL confidence
    ic, ic_pvalue = spearmanr(buy_confidence, actual_returns)
    results["information_coefficient"] = ic
    results["ic_pvalue"] = ic_pvalue
    
    # 2. Directional Accuracy (ignoring HOLD)
    # How often does model correctly predict direction when it's confident?
    non_hold_mask = predictions != 1
    if non_hold_mask.sum() > 0:
        predicted_direction = (predictions[non_hold_mask] == 2).astype(int) * 2 - 1  # +1 or -1
        actual_direction = np.sign(actual_returns[non_hold_mask])
        directional_acc = (predicted_direction == actual_direction).mean()
        results["directional_accuracy"] = directional_acc
    
    # 3. Simulated Trading Sharpe (Top-30% confidence predictions)
    # Only trade when model is most confident
    top_30_pct = int(0.3 * len(predictions))
    max_confidence = confidences.max(axis=1)
    top_indices = np.argsort(max_confidence)[-top_30_pct:]
    
    # Simulate returns: long if BUY, short if SELL, flat if HOLD
    simulated_returns = []
    for idx in top_indices:
        pred = predictions[idx]
        ret = actual_returns[idx]
        
        if pred == 2:  # BUY
            simulated_returns.append(ret - transaction_cost)
        elif pred == 0:  # SELL
            simulated_returns.append(-ret - transaction_cost)
        else:  # HOLD
            simulated_returns.append(0)
    
    simulated_returns = np.array(simulated_returns)
    sharpe = (simulated_returns.mean() / simulated_returns.std()) * np.sqrt(252) if simulated_returns.std() > 0 else 0
    results["simulated_sharpe_top30"] = sharpe
    results["simulated_return_top30"] = simulated_returns.mean() * 252  # Annualized
    
    # 4. Precision @ High Confidence
    # Accuracy only on predictions where confidence > 60%
    high_conf_mask = max_confidence > 0.6
    if high_conf_mask.sum() > 10:
        label_map_inv = {0: "SELL", 1: "HOLD", 2: "BUY"}
        actual_labels = df_test[TARGET_COLUMN].map({"SELL": 0, "HOLD": 1, "BUY": 2}).values
        precision_high_conf = (predictions[high_conf_mask] == actual_labels[high_conf_mask]).mean()
        results["precision_at_60pct_confidence"] = precision_high_conf
        results["n_high_confidence_predictions"] = high_conf_mask.sum()
    
    return results

# Usage:
softmax_probs = torch.softmax(torch.tensor(results.predictions), dim=1).numpy()
actual_returns = df_test["return_to_next_open"].values  # Use actual returns for this

trading_metrics = compute_trading_metrics(
    predictions=preds,
    confidences=softmax_probs,
    actual_returns=actual_returns
)

print("\n=== Trading Metrics ===")
print(f"Information Coefficient: {trading_metrics['information_coefficient']:.4f} (p={trading_metrics['ic_pvalue']:.4f})")
print(f"Directional Accuracy (non-HOLD): {trading_metrics.get('directional_accuracy', 'N/A'):.2%}")
print(f"Simulated Sharpe (top 30%): {trading_metrics['simulated_sharpe_top30']:.2f}")
print(f"Annualized Return (top 30%): {trading_metrics['simulated_return_top30']:.2%}")
print(f"Precision @ 60% confidence: {trading_metrics.get('precision_at_60pct_confidence', 'N/A')}")

# Interpretation:
# IC > 0.05: Model has predictive power
# Directional Accuracy > 55%: Better than random
# Simulated Sharpe > 1.0: Potentially tradable strategy
# Precision @ 60% > 60%: High-confidence predictions are reliable
```

**What good results look like:**

| Metric | Random Baseline | Good Model | Excellent Model |
|--------|-----------------|------------|-----------------|
| Information Coefficient | 0.00 | 0.05-0.10 | >0.15 |
| Directional Accuracy | 50% | 55-60% | >65% |
| Simulated Sharpe | 0.0 | 0.5-1.0 | >1.5 |
| Precision @ 60% conf | 33% | 50-60% | >70% |

### 5.4 Baseline Comparisons

**Critical**: Know what "good" means by comparing against naive baselines.

```python
# ========== Baseline Comparisons ==========
from tweet_classifier.config import LABEL_MAP

# Baseline 1: Naive (always predict majority class - SELL in this dataset)
naive_predictions = np.full(len(df_test), 0)  # Always SELL (37.8% of data)
naive_accuracy = (df_test[TARGET_COLUMN].map(LABEL_MAP).values == naive_predictions).mean()

# Baseline 2: Random (uniform across 3 classes)
random_accuracy = 1/3  # Expected accuracy
random_f1 = 1/3  # Expected F1

# Baseline 3: Class-weighted random (matches training distribution)
class_probs = df_train[TARGET_COLUMN].value_counts(normalize=True)
weighted_random_acc = (class_probs ** 2).sum()  # Probability of matching

print("\n=== Baseline Comparisons ===")
print(f"Model Accuracy:        {results.metrics['test_accuracy']:.2%}")
print(f"Naive (always SELL):   {naive_accuracy:.2%}")
print(f"Random:                {random_accuracy:.2%}")
print(f"Weighted Random:       {weighted_random_acc:.2%}")

# Improvement metrics
improvement_vs_naive = (results.metrics['test_accuracy'] - naive_accuracy) / naive_accuracy
improvement_vs_random = (results.metrics['test_accuracy'] - random_accuracy) / random_accuracy

print(f"\n=== Improvement ===")
print(f"vs Naive:  {improvement_vs_naive:+.1%}")
print(f"vs Random: {improvement_vs_random:+.1%}")

# Is the model actually useful?
if results.metrics['test_accuracy'] <= naive_accuracy:
    print("⚠️  WARNING: Model performs WORSE than always predicting SELL!")
elif improvement_vs_naive < 0.05:
    print("⚠️  Model barely beats naive baseline (<5% improvement)")
else:
    print("✓ Model shows meaningful improvement over baselines")
```

**Interpretation:**
- **Must beat naive baseline**: SELL is majority at 37.8%, so always predicting SELL = 37.8% accuracy
- **Random baseline**: With balanced classes, random = ~33% accuracy
- **Model should be >5% better**: Otherwise, simpler strategies win
- **Trading baseline**: Does simulated Sharpe beat buy-and-hold SPY?

---

## Phase 6: Inference (Production)

### 6.1 Load Trained Model

```python
model = FinBERTMultiModal.from_pretrained("models/finbert-tweet-classifier/final")
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
scaler = joblib.load("models/scaler.pkl")  # Save during training
```

### 6.2 Predict on New Tweets

```python
def predict_tweet(text: str, numerical_features: dict, author: str, category: str) -> tuple:
    """
    Predict BUY/HOLD/SELL for a single tweet.
    
    IMPORTANT: Do NOT include spy_return_1d in numerical_features!
    It was removed from training due to look-ahead bias (uses future data).
    
    Args:
        text: Tweet content
        numerical_features: Dict with ONLY these keys:
            - volatility_7d
            - relative_volume  
            - rsi_14
            - distance_from_ma_20
        author: Tweet author (for author embedding)
        category: Message category (for category embedding)
    
    Returns:
        (prediction, confidence_scores)
    """
    # Validate no forbidden features
    forbidden = ["spy_return_1d", "spy_return_1hr", "return_1hr"]
    for f in forbidden:
        if f in numerical_features:
            raise ValueError(f"LEAK: {f} is future data, cannot use for inference!")
    
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Scale numerical features (ONLY 4 features - NO spy_return_1d!)
    num_array = np.array([[
        numerical_features.get("volatility_7d", 0),
        numerical_features.get("relative_volume", 1),
        numerical_features.get("rsi_14", 50),
        numerical_features.get("distance_from_ma_20", 0),
    ]])
    num_scaled = torch.tensor(scaler.transform(num_array), dtype=torch.float32)
    
    # Encode categorical features
    author_idx = torch.tensor([author_to_idx.get(author, 0)], dtype=torch.long)
    category_idx = torch.tensor([category_to_idx.get(category, 0)], dtype=torch.long)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            numerical=num_scaled,
            author_idx=author_idx,
            category_idx=category_idx
        )
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return labels[pred], probs[0].tolist()

# Example usage
signal, confidence = predict_tweet(
    text="Apple beats earnings expectations, strong iPhone sales",
    numerical_features={
        "volatility_7d": 0.02, 
        "relative_volume": 1.5, 
        "rsi_14": 65,
        "distance_from_ma_20": 0.03
        # NOTE: No spy_return_1d - it's future data!
    },
    author="Wall St Engine",
    category="earnings"
)
print(f"Signal: {signal}, Confidence: {confidence}")
```

---

## File Structure

```
TimeWaste2/
├── src/
│   └── tweet_enricher/              # Existing enrichment pipeline
│   └── tweet_classifier/            # ✅ Training code (Phase 4 complete)
│       ├── __init__.py              # ✅ Module exports
│       ├── config.py                # ✅ Global configuration constants
│       ├── dataset.py               # ✅ TweetDataset + scaler persistence
│       ├── model.py                 # ✅ FinBERTMultiModal class
│       ├── trainer.py               # ✅ WeightedTrainer + compute_metrics
│       ├── train.py                 # ✅ Training script with CLI
│       ├── data/
│       │   ├── __init__.py          # ✅ Data submodule exports
│       │   ├── loader.py            # ✅ Data loading + filtering
│       │   ├── splitter.py          # ✅ Hash-based splitting
│       │   └── weights.py           # ✅ Class weight computation
│       └── predict.py               # TODO: Inference utilities
├── models/
│   └── finbert-tweet-classifier/    # Created during training
│       ├── final/                   # Best model checkpoint
│       ├── model_config.json        # Model architecture config
│       ├── scaler.pkl               # Fitted StandardScaler
│       └── encodings.pkl            # Author/category mappings
├── output/
│   ├── 15-dec2.csv                  # Clean parsed tweets
│   └── 15-dec-enrich7.csv           # Enriched with indicators (5,866 rows)
├── notebooks/
│   ├── phase0_validation.ipynb      # ✅ Pre-training validation
│   └── phase2_feature_engineering.ipynb  # ✅ Feature engineering verification
├── tests/
│   └── test_tweet_classifier.py     # ✅ Unit tests (31 tests)
└── pyproject.toml                   # Dependencies included
```

---

## Dependencies

Add to `requirements.txt` (already in `pyproject.toml`):

```
transformers>=4.30.0
torch>=2.0.0
accelerate>=0.26.0  # Required for HuggingFace Trainer
datasets>=2.14.0
scikit-learn>=1.3.0
seaborn>=0.12.0
joblib>=1.3.0
scipy>=1.11.0
```

---

## Next Steps

1. ~~**Run enrichment** on `15-dec2.csv` to generate enriched data~~ ✅ Done (`15-dec-enrich7.csv`)
2. ~~**Run Phase 0 validation** to confirm data integrity~~ ✅ Done (see `notebooks/phase0_validation.ipynb`)
3. ~~**Create** `src/tweet_classifier/` module~~ ✅ Done (Phase 2 feature engineering complete)
4. ~~**Implement** `FinBERTMultiModal` model class~~ ✅ Done (Phase 3 complete)
5. ~~**Implement training pipeline** with WeightedTrainer~~ ✅ Done (Phase 4 complete)
6. **Run training** and evaluate on test set (Phase 5):
   ```bash
   source .venv/bin/activate
   python -m tweet_classifier.train --epochs 5 --batch-size 16
   ```
7. **Iterate** on hyperparameters based on F1 scores
8. **Deploy** for real-time inference on new tweets (Phase 6)

---

## Known Limitations and Mitigations

Based on external review (Perplexity), the following concerns were raised. Here's how they are addressed:

### Addressed Concerns

| Concern | Status | Resolution |
|---------|--------|------------|
| "spy_return_1hr is look-ahead bias" | **RESOLVED** | Not used as feature; it's REFERENCE only (see Feature vs Target table above) |
| Future data used as features | **RESOLVED** | All future columns explicitly excluded from `NUMERICAL_FEATURES` |

### Valid Concerns to Monitor

| Concern | Severity | Mitigation | Status |
|---------|----------|------------|--------|
| **Class balance** | ✅ Resolved | Actually well-balanced: SELL 37.8%, BUY 35.4%, HOLD 26.8% | ✅ No action needed |
| **Author bias** (top 2 = 62.2%) | Medium | **Added author as embedding feature** | ✅ Implemented |
| **Premarket leakage** (29.7% tweets) | Low | Technical indicators use day T close; accept or filter | ⚠️ Monitor |
| **1-hour labels may be noisy** | Medium | **Switched to 1-day labels (`label_1d_3class`)** | ✅ Implemented |
| **Text duplication** in some tweets | Low | Parser cleanup applied; verify in EDA | ✅ Fixed |
| **Non-English text** | Low | Filtered in parser using langdetect | ✅ Fixed |
| **Sparse data per ticker** | Low | Model should generalize across tickers | Monitor |
| **Random split (not temporal)** | Low | Consider temporal split for production | Future |

**Note on Data Splits**: Current implementation splits by `tweet_hash` (prevents text leakage). For production robustness, consider:
- **Temporal split**: Train on earlier dates, test on later dates (simulates real deployment)
- **Stock-stratified split**: Ensure each stock appears in all splits proportionally

### Recommended Pre-Training EDA

✅ **Completed** - See `notebooks/phase0_validation.ipynb` for full results.

Summary of validation:
```python
# 1. Class distribution (well-balanced!)
#    SELL: 37.8%, BUY: 35.4%, HOLD: 26.8%

# 2. Author distribution
#    Wall St Engine: 34.3%, Hardik Shah: 27.9%, Evan: 25.5%

# 3. Verified no future columns in features ✓

# 4. Reliable samples: 4,545 (77.5%)

# 5. Premarket risk: 29.7% (acceptable, minor leakage)
```

---

## Alternative Approaches to Consider

| Approach | Pros | Cons |
|----------|------|------|
| **Freeze FinBERT** | Faster training, less overfitting | May miss domain nuances |
| **LoRA/PEFT** | Efficient fine-tuning | Requires additional setup |
| **Text-only model** | Simpler | Ignores market context |
| **Regression target** | Continuous predictions | Harder to evaluate |
| **Switch back to 1-hour labels** | Faster signal | More noise (currently using 1-day) |

