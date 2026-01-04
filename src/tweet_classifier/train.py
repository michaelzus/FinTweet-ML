"""Training script for FinBERT tweet classifier.

This module provides:
- create_training_args(): Build TrainingArguments from configuration
- train(): Main training function
- CLI interface for running training from command line

Usage:
    python -m tweet_classifier.train --data-path output/15-dec-enrich7.csv --epochs 5
    python -m tweet_classifier.train --epochs 5 --evaluate-test  # Run test evaluation after training
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from transformers import BertTokenizer, EarlyStoppingCallback, TrainingArguments

from tweet_classifier.config import (
    AUTHOR_EMBEDDING_DIM,
    CATEGORY_EMBEDDING_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_DIR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    FINBERT_MODEL_NAME,
    NUMERICAL_FEATURES,
    NUMERICAL_HIDDEN_DIM,
    TARGET_COLUMN,
)
from tweet_classifier.data.loader import filter_reliable, load_enriched_data
from tweet_classifier.data.splitter import (
    get_split_summary,
    get_temporal_split_summary,
    split_by_hash,
    split_by_time,
    verify_no_leakage,
)
from tweet_classifier.data.weights import (
    apply_class_boosts,
    compute_class_weights,
    get_weight_summary,
    weights_to_tensor,
)
from tweet_classifier.dataset import (
    create_categorical_encodings,
    create_dataset_from_df,
    save_preprocessing_artifacts,
)
from tweet_classifier.model import FinBERTMultiModal
from tweet_classifier.trainer import WeightedTrainer, compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_training_args(
    output_dir: Path,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    fp16: bool = True,
    logging_steps: int = 100,
    save_total_limit: int = 2,
) -> TrainingArguments:
    """Create TrainingArguments for the FinBERT trainer.

    Args:
        output_dir: Directory to save model checkpoints and logs.
        num_epochs: Number of training epochs.
        batch_size: Batch size per device for training and evaluation.
        learning_rate: Initial learning rate for AdamW optimizer.
        warmup_ratio: Fraction of total steps for learning rate warmup.
        weight_decay: Weight decay for AdamW optimizer.
        fp16: Whether to use mixed precision training (requires CUDA).
        logging_steps: Log training metrics every N steps.
        save_total_limit: Maximum number of checkpoints to keep.

    Returns:
        Configured TrainingArguments instance.
    """
    # Only enable fp16 if CUDA is available
    use_fp16 = fp16 and torch.cuda.is_available()

    # Disable pin_memory for MPS (Apple Silicon) - not supported
    use_pin_memory = not torch.backends.mps.is_available()

    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
        * 2,  # Larger batch for eval (no gradients)
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to="none",  # Disable wandb/tensorboard by default
        remove_unused_columns=False,  # Keep all columns in dataset
        dataloader_pin_memory=use_pin_memory,  # Disable for MPS
    )


def train(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    freeze_bert: bool = False,
    freeze_layers: int = 0,
    dropout: float = DEFAULT_DROPOUT,
    evaluate_test: bool = False,
    temporal_split: bool = False,
    early_stopping_patience: int = 2,
    buy_weight_boost: float = 1.0,
    sell_weight_boost: float = 1.0,
    classifier_hidden: int = 128,
    numerical_hidden_dim: int = NUMERICAL_HIDDEN_DIM,
    author_embedding_dim: int = AUTHOR_EMBEDDING_DIM,
    category_embedding_dim: int = CATEGORY_EMBEDDING_DIM,
) -> Optional[Dict[str, Any]]:
    """Train the FinBERT multi-modal tweet classifier.

    Args:
        data_path: Path to enriched CSV data file.
        output_dir: Directory to save model and artifacts.
        num_epochs: Number of training epochs.
        batch_size: Batch size per device.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW optimizer (L2 regularization).
        freeze_bert: If True, freeze all BERT parameters (faster training).
        freeze_layers: Number of BERT layers to freeze (0-12). Overrides freeze_bert if > 0.
        dropout: Dropout probability for regularization.
        evaluate_test: If True, run full evaluation on test set after training.
        temporal_split: If True, split by timestamp (train early, test late).
        early_stopping_patience: Stop training if no improvement for N epochs.
        buy_weight_boost: Multiplier for BUY class weight (>1.0 improves BUY recall).
        sell_weight_boost: Multiplier for SELL class weight (>1.0 improves SELL recall).
        classifier_hidden: Hidden dimension for classifier head.
        numerical_hidden_dim: Hidden dimension for numerical feature encoder.
        author_embedding_dim: Dimension for author embeddings.
        category_embedding_dim: Dimension for category embeddings.

    Returns:
        Dictionary with test evaluation results if evaluate_test=True, else None.
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if output_dir is None:
        output_dir = DEFAULT_MODEL_DIR

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Step 1: Load and filter data ==========
    logger.info(f"Loading data from {data_path}")
    df = load_enriched_data(data_path)
    logger.info(f"Loaded {len(df)} samples")

    df_reliable = filter_reliable(df)
    logger.info(f"After filtering: {len(df_reliable)} reliable samples")

    # ========== Step 2: Split data ==========
    if temporal_split:
        logger.info("Splitting data by TIMESTAMP (temporal validation)...")
        df_train, df_val, df_test = split_by_time(df_reliable)
        split_summary = get_temporal_split_summary(df_train, df_val, df_test)
        logger.info(
            f"Train: {split_summary['train']['samples']} samples ({split_summary['train']['percentage']:.1f}%)"
        )
        logger.info(
            f"  Date range: {split_summary['train']['date_range']['start']} to {split_summary['train']['date_range']['end']}"
        )
        logger.info(
            f"Val: {split_summary['val']['samples']} samples ({split_summary['val']['percentage']:.1f}%)"
        )
        logger.info(
            f"  Date range: {split_summary['val']['date_range']['start']} to {split_summary['val']['date_range']['end']}"
        )
        logger.info(
            f"Test: {split_summary['test']['samples']} samples ({split_summary['test']['percentage']:.1f}%)"
        )
        logger.info(
            f"  Date range: {split_summary['test']['date_range']['start']} to {split_summary['test']['date_range']['end']}"
        )
    else:
        logger.info("Splitting data by tweet_hash (random)...")
        df_train, df_val, df_test = split_by_hash(df_reliable)
        verify_no_leakage(df_train, df_val, df_test)
        split_summary = get_split_summary(df_train, df_val, df_test)
        logger.info(
            f"Train: {split_summary['train']['samples']} samples ({split_summary['train']['percentage']:.1f}%)"
        )
        logger.info(
            f"Val: {split_summary['val']['samples']} samples ({split_summary['val']['percentage']:.1f}%)"
        )
        logger.info(
            f"Test: {split_summary['test']['samples']} samples ({split_summary['test']['percentage']:.1f}%)"
        )

    # ========== Step 3: Create categorical encodings ==========
    logger.info("Creating categorical encodings from training data...")
    encodings = create_categorical_encodings(df_train)
    logger.info(
        f"Authors: {encodings['num_authors']}, Categories: {encodings['num_categories']}"
    )

    # ========== Step 4: Initialize tokenizer ==========
    logger.info(f"Loading tokenizer from {FINBERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)

    # ========== Step 5: Create datasets ==========
    logger.info("Creating training dataset...")
    train_dataset, scaler = create_dataset_from_df(
        df_train,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],  # Phase 1
        encodings["sector_to_idx"],  # Phase 1
        encodings["market_cap_to_idx"],  # Phase 1
        fit_scaler=True,
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    logger.info("Creating validation dataset...")
    val_dataset, _ = create_dataset_from_df(
        df_val,
        tokenizer,
        encodings["author_to_idx"],
        encodings["category_to_idx"],
        encodings["market_regime_to_idx"],  # Phase 1
        encodings["sector_to_idx"],  # Phase 1
        encodings["market_cap_to_idx"],  # Phase 1
        scaler=scaler,
        fit_scaler=False,
    )
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # ========== Step 6: Compute class weights ==========
    logger.info("Computing class weights...")
    class_weights = compute_class_weights(df_train[TARGET_COLUMN])

    # Apply class weight boosts if specified
    if buy_weight_boost != 1.0 or sell_weight_boost != 1.0:
        class_weights = apply_class_boosts(class_weights, buy_weight_boost, sell_weight_boost)
        logger.info(f"Applied class boosts: BUY={buy_weight_boost}x, SELL={sell_weight_boost}x")

    class_weights_tensor = weights_to_tensor(class_weights)

    weight_summary = get_weight_summary(df_train[TARGET_COLUMN], class_weights)
    for label, info in weight_summary.items():
        logger.info(f"  {label}: count={info['count']}, weight={info['weight']:.3f}")

    # ========== Step 7: Initialize model ==========
    logger.info("Initializing FinBERTMultiModal model...")
    model = FinBERTMultiModal(
        num_numerical_features=len(NUMERICAL_FEATURES),
        num_authors=encodings["num_authors"],
        num_categories=encodings["num_categories"],
        num_market_regimes=encodings["num_market_regimes"],  # Phase 1
        num_sectors=encodings["num_sectors"],  # Phase 1
        num_market_caps=encodings["num_market_caps"],  # Phase 1
        freeze_bert=freeze_bert,
        dropout=dropout,
        numerical_hidden_dim=numerical_hidden_dim,
        classifier_hidden_dim=classifier_hidden,
        author_embedding_dim=author_embedding_dim,
        category_embedding_dim=category_embedding_dim,
    )

    # Apply partial layer freezing if specified (overrides freeze_bert)
    if freeze_layers > 0:
        model.freeze_bert_layers(freeze_layers)
        logger.info(f"Froze {freeze_layers}/12 BERT encoder layers")
    elif freeze_bert:
        logger.info("BERT parameters are FROZEN (only classifier will be trained)")
    else:
        logger.info("BERT parameters are TRAINABLE (full fine-tuning)")

    # ========== Step 8: Create training arguments ==========
    training_args = create_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # ========== Step 9: Initialize trainer with early stopping ==========
    logger.info("Initializing WeightedTrainer...")
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info(
            f"Early stopping enabled: patience={early_stopping_patience} epochs"
        )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # ========== Step 10: Train ==========
    logger.info("Starting training...")
    trainer.train()

    # ========== Step 11: Save artifacts ==========
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir / "final"))

    logger.info("Saving preprocessing artifacts...")
    save_preprocessing_artifacts(scaler, encodings, output_dir)

    # Save model config
    model_config = model.get_config()
    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model config to {config_path}")

    # ========== Step 12: Final evaluation on validation set ==========
    logger.info("Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    logger.info("Validation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

    # ========== Step 13: Optional test set evaluation ==========
    test_results = None
    if evaluate_test:
        logger.info("\n" + "=" * 60)
        logger.info("Running full evaluation on TEST set...")
        logger.info("=" * 60)

        # Import here to avoid circular imports
        from tweet_classifier.evaluate import run_full_evaluation

        # Create test dataset
        test_dataset, _ = create_dataset_from_df(
            df_test,
            tokenizer,
            encodings["author_to_idx"],
            encodings["category_to_idx"],
            encodings["market_regime_to_idx"],  # Phase 1
            encodings["sector_to_idx"],  # Phase 1
            encodings["market_cap_to_idx"],  # Phase 1
            scaler=scaler,
            fit_scaler=False,
        )

        test_results = run_full_evaluation(
            model=model,
            test_dataset=test_dataset,
            df_test=df_test,
            df_train=df_train,
            output_dir=output_dir / "evaluation",
        )

    logger.info("Training complete!")
    return test_results


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train FinBERT tweet classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to enriched CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save model and artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze BERT parameters (faster training, less memory)",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Number of BERT layers to freeze (0-12). 0=none, 12=all. Overrides --freeze-bert",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout probability",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Run full evaluation on test set after training (confusion matrix, trading metrics, baselines)",
    )
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Use temporal split (train on early dates, test on late dates) instead of random hash split",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop training if no improvement for N epochs (0 to disable)",
    )
    parser.add_argument(
        "--buy-weight-boost",
        type=float,
        default=1.0,
        help="Multiplier for BUY class weight (>1.0 improves BUY recall, e.g., 1.4)",
    )
    parser.add_argument(
        "--sell-weight-boost",
        type=float,
        default=1.0,
        help="Multiplier for SELL class weight (>1.0 improves SELL recall)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for AdamW optimizer (L2 regularization)",
    )

    # Architecture parameters
    parser.add_argument(
        "--classifier-hidden",
        type=int,
        default=128,
        help="Hidden dimension for classifier head",
    )
    parser.add_argument(
        "--numerical-hidden-dim",
        type=int,
        default=NUMERICAL_HIDDEN_DIM,
        help="Hidden dimension for numerical feature encoder",
    )
    parser.add_argument(
        "--author-embedding-dim",
        type=int,
        default=AUTHOR_EMBEDDING_DIM,
        help="Dimension for author embeddings",
    )
    parser.add_argument(
        "--category-embedding-dim",
        type=int,
        default=CATEGORY_EMBEDDING_DIM,
        help="Dimension for category embeddings",
    )

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_bert=args.freeze_bert,
        freeze_layers=args.freeze_layers,
        dropout=args.dropout,
        evaluate_test=args.evaluate_test,
        temporal_split=args.temporal_split,
        early_stopping_patience=args.early_stopping_patience,
        buy_weight_boost=args.buy_weight_boost,
        sell_weight_boost=args.sell_weight_boost,
        classifier_hidden=args.classifier_hidden,
        numerical_hidden_dim=args.numerical_hidden_dim,
        author_embedding_dim=args.author_embedding_dim,
        category_embedding_dim=args.category_embedding_dim,
    )


if __name__ == "__main__":
    main()
