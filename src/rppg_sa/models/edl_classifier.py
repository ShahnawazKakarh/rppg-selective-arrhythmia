"""CNN1D-Transformer wrapped with an Evidential (Dirichlet) head.

Drop-in replacement for `CNN1DTransformer` when training under the
Evidential Deep Learning paradigm. Reuses the same feature extractor
(conv stem + transformer encoder + mean pool) and swaps the softmax
head for a Dirichlet-evidence head returning alpha = evidence + 1.

Forward returns alpha (B, num_classes); the trainer applies edl_mse_loss
on alpha rather than CE on logits. For prediction, argmax of alpha is
equivalent to argmax of the mean Dirichlet probability vector p = alpha / S.
"""
from __future__ import annotations

import torch
from torch import nn

from rppg_sa.models.cnn1d_transformer import CNN1DTransformer
from rppg_sa.uncertainty.evidential import EvidentialHead


class CNN1DTransformerEDL(nn.Module):
    """CNN1D + Transformer + EvidentialHead."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Reuse the base model purely for its feature extractor — we throw
        # away its softmax head.
        self.base = CNN1DTransformer(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )
        feature_dim = conv_channels[-1]
        self.edl_head = EvidentialHead(feature_dim=feature_dim, num_classes=num_classes)
        self.num_classes = num_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Dirichlet concentrations alpha (B, num_classes), alpha >= 1."""
        h = self.features(x)
        return self.edl_head(h)
