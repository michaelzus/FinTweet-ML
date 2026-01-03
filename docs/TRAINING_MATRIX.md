# FinBERT Tweet Classifier - Training Matrix

> **Date**: January 2025  
> **Dataset**: `output/dataset_2025_full.csv` (~43,468 samples)  
> **Model**: FinBERT MultiModal (yiyanghkust/finbert-tone)  
> **Task**: 3-class classification (SELL, HOLD, BUY)

---

## Analysis Summary

### Current Setup

- **Old Dataset**: 42,749 tweets (December 2024 data)
- **New Dataset**: `dataset_2025_full.csv` - ~43,468 samples (full 2025 data)
- **Model**: FinBERT MultiModal with 10 numerical + 5 categorical features
- **Task**: 3-class classification (SELL, HOLD, BUY)

### Key Findings from V1-V3

| Model | Accuracy | IC | IC p-value | Sharpe | Issue |
|-------|----------|-----|------------|--------|-------|
| V1 | 40.84% | 0.034 | **0.025** | -0.18 | Severe overfitting |
| V2 | 40.23% | 0.028 | 0.066 | -0.03 | Class collapse (76% BUY recall) |
| V3 | **41.92%** | 0.015 | 0.320 | -0.77 | Non-significant IC |

**Critical Insight**: V1 @ 40% confidence threshold achieved **Sharpe 0.15** and **12.1% annual return** - the only profitable configuration!

### V1-V3 Configuration Reference

| Parameter | V1 | V2 | V3 |
|-----------|-----|-----|-----|
| epochs | 5 | 5 | 3 |
| learning_rate | 2e-5 | 2e-5 | 3e-5 |
| dropout | 0.3 (default) | 0.5 | 0.2 |
| freeze_bert | False | True | False |
| buy_weight_boost | 1.2 | 1.2 | 1.2 |
| early_stopping_patience | 2 | 2 | 2 |

---

## Training Matrix (23 Experiments)

### Phase 1: Baseline Replication (3 experiments)

Re-run V1, V2, V3 on new 2025 dataset to establish baseline.

| ID | Name | epochs | lr | dropout | freeze_bert | buy_boost | Purpose |
|----|------|--------|-----|---------|-------------|-----------|---------|
| **B1** | V1-2025 | 5 | 2e-5 | 0.3 | False | 1.2 | Baseline replication |
| **B2** | V2-2025 | 5 | 2e-5 | 0.5 | True | 1.2 | Frozen BERT replication |
| **B3** | V3-2025 | 3 | 3e-5 | 0.2 | False | 1.2 | Fast training replication |

**Commands:**

```bash
# B1: V1-2025
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --output-dir models/matrix/B1-v1-2025

# B2: V2-2025
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --freeze-bert --dropout 0.5 --output-dir models/matrix/B2-v2-2025

# B3: V3-2025
fintweet-ml train --data output/dataset_2025_full.csv --epochs 3 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --dropout 0.2 --learning-rate 3e-5 --output-dir models/matrix/B3-v3-2025
```

---

### Phase 2: Anti-Overfitting Experiments (5 experiments)

V1 showed best trading metrics but severe overfitting. Let's address this.

| ID | Name | epochs | lr | dropout | freeze_bert | weight_decay | patience | Purpose |
|----|------|--------|-----|---------|-------------|--------------|----------|---------|
| **R1** | HighDropout | 5 | 2e-5 | **0.5** | False | 0.01 | 2 | More dropout |
| **R2** | VeryHighDropout | 5 | 2e-5 | **0.6** | False | 0.01 | 2 | Aggressive dropout |
| **R3** | HighWeightDecay | 5 | 2e-5 | 0.3 | False | **0.05** | 2 | L2 regularization |
| **R4** | EarlyStopping1 | 10 | 2e-5 | 0.3 | False | 0.01 | **1** | Very early stopping |
| **R5** | ComboRegular | 5 | 2e-5 | **0.4** | False | **0.03** | **1** | Combined regularization |

**Commands:**

```bash
# R1: HighDropout
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --dropout 0.5 --output-dir models/matrix/R1-high-dropout

# R2: VeryHighDropout
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --dropout 0.6 --output-dir models/matrix/R2-very-high-dropout

# R3: HighWeightDecay (requires --weight-decay CLI support)
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --weight-decay 0.05 --output-dir models/matrix/R3-high-weight-decay

# R4: EarlyStopping1
fintweet-ml train --data output/dataset_2025_full.csv --epochs 10 --temporal-split \
    --evaluate-test --early-stopping-patience 1 --buy-weight-boost 1.2 \
    --output-dir models/matrix/R4-early-stopping-1

# R5: ComboRegular (requires --weight-decay CLI support)
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 1 --buy-weight-boost 1.2 \
    --dropout 0.4 --weight-decay 0.03 --output-dir models/matrix/R5-combo-regular
```

---

### Phase 3: Learning Rate Search (4 experiments)

Explore learning rate impact on convergence and generalization.

| ID | Name | epochs | lr | dropout | freeze_bert | Purpose |
|----|------|--------|-----|---------|-------------|---------|
| **L1** | LowerLR | 5 | **1e-5** | 0.3 | False | Slower learning |
| **L2** | VeryLowLR | 8 | **5e-6** | 0.3 | False | Very slow, more epochs |
| **L3** | HigherLR | 3 | **5e-5** | 0.3 | False | Faster convergence |
| **L4** | TinyLR_Long | 12 | **2e-6** | 0.3 | False | Very conservative |

**Commands:**

```bash
# L1: LowerLR
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --learning-rate 1e-5 --output-dir models/matrix/L1-lower-lr

# L2: VeryLowLR
fintweet-ml train --data output/dataset_2025_full.csv --epochs 8 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --learning-rate 5e-6 --output-dir models/matrix/L2-very-low-lr

# L3: HigherLR
fintweet-ml train --data output/dataset_2025_full.csv --epochs 3 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --learning-rate 5e-5 --output-dir models/matrix/L3-higher-lr

# L4: TinyLR_Long
fintweet-ml train --data output/dataset_2025_full.csv --epochs 12 --temporal-split \
    --evaluate-test --early-stopping-patience 3 --buy-weight-boost 1.2 \
    --learning-rate 2e-6 --output-dir models/matrix/L4-tiny-lr-long
```

---

### Phase 4: Partial Freezing (3 experiments)

Balance between frozen BERT (no overfitting, class collapse) and full fine-tuning.

| ID | Name | epochs | lr | dropout | freeze_layers | Purpose |
|----|------|--------|-----|---------|---------------|---------|
| **F1** | Freeze6 | 5 | 2e-5 | 0.3 | **6/12** | Partial freeze |
| **F2** | Freeze9 | 5 | 3e-5 | 0.3 | **9/12** | More frozen |
| **F3** | UnfreezeClassifier | 5 | 1e-4 | 0.3 | **All but classifier** | Only train head with high LR |

> **Note**: This requires code modification to support `--freeze-layers` argument. See [Required Code Changes](#required-code-changes) section.

**Commands (after code modification):**

```bash
# F1: Freeze6
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --freeze-layers 6 --output-dir models/matrix/F1-freeze-6

# F2: Freeze9
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --freeze-layers 9 --learning-rate 3e-5 --output-dir models/matrix/F2-freeze-9

# F3: UnfreezeClassifier
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --freeze-layers 12 --learning-rate 1e-4 --output-dir models/matrix/F3-unfreeze-classifier
```

---

### Phase 5: Class Balance Experiments (4 experiments)

Explore different class weighting strategies.

| ID | Name | epochs | lr | dropout | buy_boost | sell_boost | Purpose |
|----|------|--------|-----|---------|-----------|------------|---------|
| **C1** | NoBuyBoost | 5 | 2e-5 | 0.3 | **1.0** | 1.0 | No class boosting |
| **C2** | HighBuyBoost | 5 | 2e-5 | 0.3 | **1.5** | 1.0 | Strong BUY preference |
| **C3** | SellBoost | 5 | 2e-5 | 0.3 | 1.0 | **1.3** | Boost SELL recall |
| **C4** | BalancedBoost | 5 | 2e-5 | 0.3 | **1.3** | **1.3** | Boost both extremes |

> **Note**: C3 and C4 require code modification to support `--sell-weight-boost` argument. See [Required Code Changes](#required-code-changes) section.

**Commands:**

```bash
# C1: NoBuyBoost
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.0 \
    --output-dir models/matrix/C1-no-buy-boost

# C2: HighBuyBoost
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.5 \
    --output-dir models/matrix/C2-high-buy-boost

# C3: SellBoost (requires --sell-weight-boost CLI support)
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --sell-weight-boost 1.3 \
    --output-dir models/matrix/C3-sell-boost

# C4: BalancedBoost (requires --sell-weight-boost CLI support)
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.3 --sell-weight-boost 1.3 \
    --output-dir models/matrix/C4-balanced-boost
```

---

### Phase 6: Architecture Variants (4 experiments)

Explore model architecture changes.

| ID | Name | epochs | dropout | classifier_hidden | num_hidden_dim | Purpose |
|----|------|--------|---------|-------------------|----------------|---------|
| **A1** | WiderHead | 5 | 0.3 | **256** | 32 | Larger classifier |
| **A2** | DeeperHead | 5 | 0.3 | 128->64 | 32 | 3-layer classifier |
| **A3** | LargerNumEncoder | 5 | 0.3 | 128 | **64** | Better numerical encoding |
| **A4** | MinimalEmbeddings | 5 | 0.3 | 128 | 32 | author_dim=8, category_dim=4 |

> **Note**: Requires model architecture modifications. See [Required Code Changes](#required-code-changes) section.

**Commands (after code modification):**

```bash
# A1: WiderHead
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --classifier-hidden 256 --output-dir models/matrix/A1-wider-head

# A2: DeeperHead
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --classifier-layers 3 --output-dir models/matrix/A2-deeper-head

# A3: LargerNumEncoder
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --numerical-hidden-dim 64 --output-dir models/matrix/A3-larger-num-encoder

# A4: MinimalEmbeddings
fintweet-ml train --data output/dataset_2025_full.csv --epochs 5 --temporal-split \
    --evaluate-test --early-stopping-patience 2 --buy-weight-boost 1.2 \
    --author-embedding-dim 8 --category-embedding-dim 4 \
    --output-dir models/matrix/A4-minimal-embeddings
```

---

## Implementation Roadmap

### Week 1: Baseline + Quick Wins

| Day | Tasks | Estimated Time |
|-----|-------|----------------|
| 1-2 | Run baseline replications (B1, B2, B3) | ~4 hours |
| 3-4 | Run regularization experiments (R1, R2, R4) | ~5 hours |
| 5 | Implement `--weight-decay` CLI argument | ~30 min |
| 5 | Run R3, R5 with weight decay | ~3 hours |

### Week 2: Learning Rate + Class Balance

| Day | Tasks | Estimated Time |
|-----|-------|----------------|
| 1-2 | Run learning rate experiments (L1-L4) | ~8 hours |
| 3 | Run C1, C2 (no code changes needed) | ~3 hours |
| 4 | Implement `--sell-weight-boost` CLI argument | ~1 hour |
| 5 | Run C3, C4 | ~3 hours |

### Week 3: Architecture + Advanced

| Day | Tasks | Estimated Time |
|-----|-------|----------------|
| 1 | Implement partial freezing support | ~2 hours |
| 2 | Run F1, F2, F3 | ~5 hours |
| 3 | Implement architecture CLI arguments | ~2 hours |
| 4-5 | Run A1-A4 | ~6 hours |

### Week 4: Analysis + Follow-up

| Day | Tasks |
|-----|-------|
| 1-2 | Compile all results into spreadsheet |
| 3 | Run confidence threshold analysis on top 3 models |
| 4-5 | Fine-tune best configuration, run final evaluation |

---

## Metrics to Track

Create a results spreadsheet (`docs/TRAINING_RESULTS.csv`) with these columns:

| Metric | Description | Target |
|--------|-------------|--------|
| `experiment_id` | Experiment ID (B1, R1, etc.) | - |
| `experiment_name` | Descriptive name | - |
| `accuracy` | Test accuracy | > 42% |
| `f1_macro` | Macro F1 | > 41% |
| `f1_weighted` | Weighted F1 | > 40% |
| `sell_precision` | SELL precision | > 40% |
| `sell_recall` | SELL class recall | > 25% |
| `hold_precision` | HOLD precision | > 35% |
| `hold_recall` | HOLD class recall | > 40% |
| `buy_precision` | BUY precision | > 35% |
| `buy_recall` | BUY class recall | > 40% |
| `IC` | Information coefficient | > 0.03 |
| `IC_pvalue` | Statistical significance | < 0.05 |
| `IC_significant` | Is IC significant? | Yes |
| `sharpe_0pct` | Sharpe @ 0% confidence | - |
| `sharpe_40pct` | Sharpe @ 40% confidence | > 0 |
| `sharpe_70pct` | Sharpe @ 70% confidence | > 0 |
| `ann_return_40pct` | Annual return @ 40% conf | > 5% |
| `ann_return_70pct` | Annual return @ 70% conf | > 5% |
| `directional_accuracy` | Direction prediction accuracy | > 52% |
| `train_loss_epoch1` | Training loss at epoch 1 | - |
| `train_loss_final` | Final training loss | - |
| `val_loss_epoch1` | Validation loss at epoch 1 | - |
| `val_loss_final` | Final validation loss | - |
| `val_loss_best` | Best validation loss | - |
| `best_epoch` | Epoch with best val loss | - |
| `overfit_ratio` | val_loss_final / train_loss_final | < 1.3 |
| `training_time_minutes` | Time to train in minutes | - |
| `notes` | Any observations | - |

### Confidence Threshold Analysis

For top performing models, also track:

| Threshold | Coverage | Accuracy | IC | IC_sig | Sharpe | Ann_Return |
|-----------|----------|----------|-----|--------|--------|------------|
| 0% | 100% | - | - | - | - | - |
| 40% | - | - | - | - | - | - |
| 50% | - | - | - | - | - | - |
| 60% | - | - | - | - | - | - |
| 70% | - | - | - | - | - | - |
| 80% | - | - | - | - | - | - |

---

## Required Code Changes

### 1. Add `--weight-decay` CLI argument

**File**: `src/tweet_classifier/train.py`

```python
# Add to argparse arguments:
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.01,
    help="Weight decay for AdamW optimizer (L2 regularization)",
)

# Update create_training_args call:
training_args = create_training_args(
    output_dir=output_dir,
    num_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=args.weight_decay,  # Add this
)
```

### 2. Add SELL weight boost support

**File**: `src/tweet_classifier/data/weights.py`

```python
def apply_class_boosts(
    weights: Dict[str, float],
    buy_boost: float = 1.0,
    sell_boost: float = 1.0,
) -> Dict[str, float]:
    """Apply boost multipliers to class weights.
    
    Args:
        weights: Dictionary of class weights.
        buy_boost: Multiplier for BUY class weight.
        sell_boost: Multiplier for SELL class weight.
    
    Returns:
        Modified weights dictionary.
    """
    weights = weights.copy()
    if buy_boost != 1.0:
        weights["BUY"] *= buy_boost
    if sell_boost != 1.0:
        weights["SELL"] *= sell_boost
    return weights
```

**File**: `src/tweet_classifier/train.py`

```python
# Add to argparse:
parser.add_argument(
    "--sell-weight-boost",
    type=float,
    default=1.0,
    help="Multiplier for SELL class weight (>1.0 improves SELL recall)",
)

# Update weight computation:
if buy_weight_boost != 1.0 or sell_weight_boost != 1.0:
    class_weights = apply_class_boosts(class_weights, buy_weight_boost, sell_weight_boost)
```

### 3. Add partial layer freezing

**File**: `src/tweet_classifier/model.py`

```python
def freeze_bert_layers(self, num_layers_to_freeze: int) -> None:
    """Freeze first N transformer layers of BERT.
    
    Args:
        num_layers_to_freeze: Number of layers to freeze (0-12).
            0 = no freezing, 12 = freeze all BERT layers.
    """
    # Always freeze embeddings if freezing any layers
    if num_layers_to_freeze > 0:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
    
    # Freeze specified number of encoder layers
    for i, layer in enumerate(self.bert.encoder.layer):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Log frozen vs trainable parameters
    frozen = sum(p.numel() for p in self.bert.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
    total = frozen + trainable
    print(f"BERT layers: {num_layers_to_freeze}/12 frozen")
    print(f"BERT params: {frozen:,} frozen, {trainable:,} trainable ({100*trainable/total:.1f}% trainable)")
```

**File**: `src/tweet_classifier/train.py`

```python
# Add to argparse:
parser.add_argument(
    "--freeze-layers",
    type=int,
    default=0,
    help="Number of BERT layers to freeze (0-12). 0=none, 12=all",
)

# After model initialization:
if args.freeze_layers > 0:
    model.freeze_bert_layers(args.freeze_layers)
    logger.info(f"Froze {args.freeze_layers}/12 BERT encoder layers")
```

### 4. Add architecture CLI arguments (optional)

**File**: `src/tweet_classifier/train.py`

```python
# Add to argparse:
parser.add_argument(
    "--classifier-hidden",
    type=int,
    default=128,
    help="Hidden dimension for classifier head",
)
parser.add_argument(
    "--numerical-hidden-dim",
    type=int,
    default=32,
    help="Hidden dimension for numerical feature encoder",
)
parser.add_argument(
    "--author-embedding-dim",
    type=int,
    default=16,
    help="Dimension for author embeddings",
)
parser.add_argument(
    "--category-embedding-dim",
    type=int,
    default=8,
    help="Dimension for category embeddings",
)
```

---

## Success Criteria

### Minimum Requirements

A model must achieve ALL of:

1. Test Accuracy > 42%
2. IC > 0.03 with p < 0.05 (statistically significant)
3. Balanced recall: SELL > 20%, HOLD > 35%, BUY > 35%
4. Overfit ratio < 1.4 (val_loss / train_loss)

### Good Model

A "good" model should achieve:

1. Test Accuracy > 43%
2. IC > 0.04 with p < 0.05
3. Sharpe > 0 at 40% confidence threshold
4. Balanced recall: SELL > 25%, HOLD > 40%, BUY > 40%
5. Overfit ratio < 1.3

### Production-Ready Model

The ultimate goal - a production model should achieve:

1. Test Accuracy > 44%
2. IC > 0.05 with p < 0.01
3. **Sharpe > 0.15 at 40% confidence threshold**
4. **Annual return > 10% at 40% confidence**
5. Balanced recall: SELL > 30%, HOLD > 40%, BUY > 40%
6. Overfit ratio < 1.2
7. Directional accuracy > 53%

### Decision Framework

```
IF IC_pvalue >= 0.05:
    REJECT - No statistical significance
    
ELIF sharpe_40pct <= 0:
    REJECT - Not profitable
    
ELIF sell_recall < 0.20:
    REJECT - Class collapse
    
ELIF overfit_ratio > 1.4:
    REJECT - Severe overfitting
    
ELIF accuracy > 0.43 AND sharpe_40pct > 0.10:
    ACCEPT - Good candidate
    
ELSE:
    CONSIDER - Review manually
```

---

## Quick Reference: Evaluation Commands

### Run confidence threshold analysis

```bash
python scripts/evaluate_by_confidence.py \
    --model-dir models/matrix/B1-v1-2025/final \
    --data output/dataset_2025_full.csv \
    --output models/matrix/B1-v1-2025/confidence_analysis.json
```

### Compare multiple models

```bash
# Create a simple comparison script or use:
for model in B1 B2 B3 R1 R2 R3 R4 R5; do
    echo "=== $model ==="
    cat models/matrix/$model-*/evaluation/evaluation_results.json | jq '.accuracy, .trading_metrics.information_coefficient'
done
```

---

*Generated: January 2025*

