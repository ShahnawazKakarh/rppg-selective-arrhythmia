"""CNN1D-Transformer wrapped with spectral-norm + RFF + GP head for SNGP.

Single-pass deterministic uncertainty. Reuses the same feature extractor
as `CNN1DTransformer` (conv stem + transformer encoder + mean pool), with
two modifications:

  1. Spectral normalization is applied to every Conv1d / Linear layer in
     the backbone via `nn.utils.parametrizations.spectral_norm`. This
     bounds the Lipschitz constant of the feature map and makes the
     learned hidden space approximately distance-preserving.
  2. The softmax head is replaced by a Random Fourier Features projection
     followed by a Bayesian GP layer (`GPLayer`) that maintains a per-class
     diagonal precision matrix.

Training procedure (called from `scripts/train_classifier.py --head sngp`):
  - Standard cross-entropy training on `mean_logits` for N epochs.
  - On EVERY epoch, accumulate the per-class diagonal precision matrix
    from training-set RFF features (cheap; happens during the existing
    forward pass with `accumulate_precision=True`).
  - After the last epoch, call `model.finalize_gp()` to invert the
    accumulated precision diag into a covariance diag.

Inference (called from eval_selective.py / dump_val_predictions.py):
  - Forward returns mean_logits and per-logit variance.
  - Apply mean-field correction:
        logits' = mean / sqrt(1 + π/8 · variance)
  - Confidence for selective: -predictive_entropy of softmax(logits').
"""
from __future__ import annotations

import torch
from torch import nn

from rppg_sa.models.cnn1d_transformer import CNN1DTransformer
from rppg_sa.uncertainty.sngp import (
    GPLayer,
    RandomFourierFeatures,
    apply_spectral_norm,
)


class CNN1DTransformerSNGP(nn.Module):
    """CNN1D-Transformer + spectral norm + RFF + GP head.

    Forward returns (mean_logits, variance | None). Variance is None until
    `finalize_gp()` has been called (i.e. during training before the GP is
    finalised).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        rff_features: int = 1024,
        rff_kernel_scale: float = 1.0,
        gp_ridge: float = 1e-3,
        spectral_norm_coefficient: float = 0.95,
    ) -> None:
        super().__init__()
        # Backbone: reuse CNN1DTransformer purely for its feature extractor.
        # Discard its softmax head; replace with RFF + GPLayer.
        self.base = CNN1DTransformer(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )
        # Apply spectral normalization to all Conv1d/Linear layers in backbone.
        apply_spectral_norm(self.base, coefficient=spectral_norm_coefficient)

        feature_dim = conv_channels[-1]
        self.rff = RandomFourierFeatures(
            feature_dim=feature_dim,
            num_features=rff_features,
            kernel_scale=rff_kernel_scale,
        )
        self.gp = GPLayer(
            num_features=rff_features,
            num_classes=num_classes,
            ridge=gp_ridge,
        )
        self.num_classes = num_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate features from the spectral-normalised backbone."""
        return self.base.features(x)

    def rff_features(self, x: torch.Tensor) -> torch.Tensor:
        """RFF projection of the penultimate features."""
        h = self.features(x)
        return self.rff(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (mean_logits, variance | None)."""
        phi = self.rff_features(x)
        return self.gp(phi)

    @torch.no_grad()
    def accumulate(self, x: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate diagonal precision from a labelled batch.

        Called by the trainer at the end of each epoch (or all epochs) so
        the GP layer's `precision_diag` summarises the training-set RFF
        feature distribution per class.
        """
        phi = self.rff_features(x)
        self.gp.accumulate(phi, targets)

    def finalize_gp(self) -> None:
        """Invert the precision diag to obtain the covariance diag.

        Called once at the end of training. After this point the forward
        pass returns non-None variance and inference-time mean-field
        correction is enabled.
        """
        self.gp.finalize()

    def reset_precision(self) -> None:
        """Reset the precision accumulator. Useful when re-running the
        precision accumulation pass with different data."""
        self.gp.reset_precision()
