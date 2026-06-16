"""SNGP — Spectral-normalized Neural Gaussian Process (Liu et al., 2020).

Single-pass deterministic alternative to ensembles. Spectral normalization on
hidden weights enforces distance-preservation; a random-feature GP layer on the
penultimate features yields calibrated uncertainty.

Implementation here is a stub interface — full SNGP integration requires
spectral_norm wrappers on the backbone and a random-feature GP layer. Wire up
when the baseline + MC Dropout + Ensembles results are in.
"""
from __future__ import annotations

import torch
from torch import nn


class SNGPHead(nn.Module):  # noqa: D401 - placeholder
    """Placeholder for the SNGP output head.

    A complete implementation should:
        1. Apply `torch.nn.utils.spectral_norm` to every hidden linear/conv
           in the backbone with coefficient c < 1.
        2. Replace the final classifier layer with a random-feature GP layer
           (RFF approximation) that maintains running second-moment statistics
           updated after each epoch.
        3. At inference, return mean logits plus a variance-corrected
           softmax (mean-field approximation).

    Not yet implemented; left as a planned UQ method per the proposal.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)
        # TODO: replace with RFF-GP layer; track training statistics.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


__all__ = ["SNGPHead"]
