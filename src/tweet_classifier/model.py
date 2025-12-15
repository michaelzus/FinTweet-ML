"""FinBERT-based multi-modal model for tweet classification.

This module provides the FinBERTMultiModal model that combines:
- FinBERT text encoder (768-dim CLS embedding)
- Numerical feature encoder (MLP)
- Author and category embeddings

For BUY/HOLD/SELL classification of financial tweets.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from tweet_classifier.config import (
    AUTHOR_EMBEDDING_DIM,
    CATEGORY_EMBEDDING_DIM,
    DEFAULT_DROPOUT,
    FINBERT_MODEL_NAME,
    NUM_CLASSES,
    NUMERICAL_HIDDEN_DIM,
)


class FinBERTMultiModal(nn.Module):
    """FinBERT with numerical + categorical feature fusion for price prediction.

    Architecture:
        - FinBERT encoder: 768-dim CLS token embedding
        - Numerical encoder: MLP (num_features -> 64 -> 32)
        - Author embedding: 16-dim
        - Category embedding: 8-dim
        - Fusion: Concatenate all (824-dim) -> classifier (3 classes)
    """

    def __init__(
        self,
        num_numerical_features: int,
        num_authors: int,
        num_categories: int,
        num_classes: int = NUM_CLASSES,
        finbert_model: str = FINBERT_MODEL_NAME,
        freeze_bert: bool = False,
        dropout: float = DEFAULT_DROPOUT,
        author_embedding_dim: int = AUTHOR_EMBEDDING_DIM,
        category_embedding_dim: int = CATEGORY_EMBEDDING_DIM,
        numerical_hidden_dim: int = NUMERICAL_HIDDEN_DIM,
    ):
        """Initialize the multi-modal model.

        Args:
            num_numerical_features: Number of numerical input features.
            num_authors: Number of unique authors for embedding.
            num_categories: Number of unique categories for embedding.
            num_classes: Number of output classes (default: 3 for SELL/HOLD/BUY).
            finbert_model: HuggingFace model name for FinBERT.
            freeze_bert: If True, freeze BERT parameters (no fine-tuning).
            dropout: Dropout probability for regularization.
            author_embedding_dim: Dimension of author embeddings.
            category_embedding_dim: Dimension of category embeddings.
            numerical_hidden_dim: Hidden dimension for numerical encoder output.
        """
        super().__init__()

        # Store config for serialization
        self.num_numerical_features = num_numerical_features
        self.num_authors = num_authors
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.finbert_model_name = finbert_model
        self.freeze_bert = freeze_bert
        self.dropout_prob = dropout
        self.author_embedding_dim = author_embedding_dim
        self.category_embedding_dim = category_embedding_dim
        self.numerical_hidden_dim = numerical_hidden_dim

        # FinBERT encoder
        self.bert = BertModel.from_pretrained(finbert_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size  # 768

        # Categorical embeddings (to reduce author bias)
        self.author_embedding = nn.Embedding(num_authors, author_embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)

        # Numerical feature encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, numerical_hidden_dim),
            nn.ReLU(),
        )

        # Fusion + classifier
        # 768 (BERT) + 32 (numerical) + 16 (author) + 8 (category) = 824
        fusion_size = bert_hidden_size + numerical_hidden_dim + author_embedding_dim + category_embedding_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical: torch.Tensor,
        author_idx: torch.Tensor,
        category_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multi-modal model.

        Args:
            input_ids: Token IDs from tokenizer, shape (batch_size, seq_len).
            attention_mask: Attention mask, shape (batch_size, seq_len).
            numerical: Numerical features, shape (batch_size, num_features).
            author_idx: Author indices, shape (batch_size,).
            category_idx: Category indices, shape (batch_size,).
            labels: Optional labels for loss computation, shape (batch_size,).

        Returns:
            Dictionary containing:
                - 'logits': Classification logits, shape (batch_size, num_classes)
                - 'loss': Cross-entropy loss (only if labels provided)
        """
        # Get BERT [CLS] embedding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [batch, 768]

        # Encode numerical features
        num_embedding = self.numerical_encoder(numerical)  # [batch, 32]

        # Encode categorical features
        author_emb = self.author_embedding(author_idx)  # [batch, 16]
        category_emb = self.category_embedding(category_idx)  # [batch, 8]

        # Fusion
        combined = torch.cat([cls_embedding, num_embedding, author_emb, category_emb], dim=1)

        # Classification
        logits = self.classifier(combined)  # [batch, num_classes]

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def get_config(self) -> Dict[str, Union[int, float, str, bool]]:
        """Get model configuration for serialization.

        Returns:
            Dictionary with model configuration parameters.
        """
        return {
            "num_numerical_features": self.num_numerical_features,
            "num_authors": self.num_authors,
            "num_categories": self.num_categories,
            "num_classes": self.num_classes,
            "finbert_model": self.finbert_model_name,
            "freeze_bert": self.freeze_bert,
            "dropout": self.dropout_prob,
            "author_embedding_dim": self.author_embedding_dim,
            "category_embedding_dim": self.category_embedding_dim,
            "numerical_hidden_dim": self.numerical_hidden_dim,
        }

