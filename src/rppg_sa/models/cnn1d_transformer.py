"""1D-CNN + Transformer classifier for rPPG pulse waveforms.

Convolutional front-end captures local beat morphology; transformer encoder
captures longer-range rhythm structure (essential for AF, where the contribution
is in the irregularity across beats, not within them).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class CNN1DTransformer(nn.Module):
    """1D-CNN front-end + Transformer encoder + classification head.

    Args:
        in_channels: Number of input channels (1 for single pulse trace).
        num_classes: Number of output classes (3 for NSR / AF / Other).
        conv_channels: Channel sizes for the three conv blocks.
        transformer_layers: Number of transformer encoder layers.
        transformer_heads: Number of attention heads.
        dropout: Dropout rate, applied in conv blocks, transformer, and head.

    Shape:
        Input: (B, in_channels, L) where L is the segment length in samples.
        Output: (B, num_classes) logits.

    The dropout layers double as the MC-Dropout uncertainty path; setting the
    model to `.train()` while only enabling these dropouts is handled in
    `rppg_sa.uncertainty.mc_dropout`.
    """

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
        c1, c2, c3 = conv_channels

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.pos_enc = _PositionalEncoding(d_model=c3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c3,
            nhead=transformer_heads,
            dim_feedforward=c3 * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(c3),
            nn.Linear(c3, c3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (B, d_model) for use by UQ heads (e.g. SNGP)."""
        z = self.conv_block(x)                 # (B, C, L')
        z = z.transpose(1, 2)                  # (B, L', C)
        z = self.pos_enc(z)
        z = self.transformer(z)                # (B, L', C)
        return z.mean(dim=1)                   # global average over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h)
