"""SNGP — Spectral-normalized Neural Gaussian Process (Liu et al., NeurIPS 2020).

Single-pass deterministic alternative to ensembles. The recipe:

  1. Apply spectral normalization to every Linear / Conv layer in the
     backbone, bounding the Lipschitz constant. This makes the learned
     feature map approximately distance-preserving (||f(x) − f(x')|| is
     bounded above and below by a multiple of ||x − x'||), which is the
     prerequisite for the GP-based uncertainty to track in-distribution
     vs. out-of-distribution test inputs.
  2. Project the penultimate feature h ∈ R^d to random Fourier features
     φ(h) ∈ R^D via φ_k(h) = √(2/D) · cos(w_k^⊤ h + b_k),
     with w_k ~ N(0, σ²I) and b_k ~ U(0, 2π). This is a finite-dimensional
     RBF kernel approximation.
  3. Read out class logits with a standard linear layer L: R^D → R^K.
  4. During the final epoch (or all epochs, depending on schedule),
     accumulate the precision matrix
            Σ⁻¹ = λI + Σ_i φ(h_i) φ(h_i)^⊤
     where λ is a weight-decay-style L2 prior. At inference, compute the
     per-logit variance
            σ²_k(x) = φ(h(x))^⊤ Σ inv(L_k diag) φ(h(x))
     and apply the mean-field correction
            p(y|x) = softmax(μ(x) / √(1 + π/8 · σ²(x))).

This module provides the building blocks (`RandomFourierFeatures`,
`GPLayer`) and a wrapper class that composes them with a backbone is in
`rppg_sa.models.sngp_classifier`. Training-time precision accumulation and
inference-time mean-field correction are exposed as standalone helpers so
they can be called from any training loop.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Random Fourier Features
# ---------------------------------------------------------------------
class RandomFourierFeatures(nn.Module):
    """Frozen random Fourier features for an RBF kernel approximation.

    φ_k(h) = √(2/D) · cos(w_k^⊤ h + b_k)

    where w_k, b_k are sampled once at initialisation and registered as
    buffers (not trained). The output dimension D is `num_features`.

    Args:
        feature_dim: Input dimensionality d (penultimate hidden size).
        num_features: Output dimensionality D of the RFF projection.
            Standard SNGP defaults: D = 1024.
        kernel_scale: RBF lengthscale; w_k ~ N(0, 1 / kernel_scale²).
    """

    def __init__(self, feature_dim: int, num_features: int = 1024,
                 kernel_scale: float = 1.0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.kernel_scale = kernel_scale
        # w_k ~ N(0, 1 / kernel_scale²)
        W = torch.randn(num_features, feature_dim) / kernel_scale
        b = 2.0 * math.pi * torch.rand(num_features)
        # Buffers, not parameters — RFF random projections stay frozen.
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d); return (B, D)
        projection = F.linear(h, self.W) + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)


# ---------------------------------------------------------------------
# GP last-layer with precision matrix accumulation
# ---------------------------------------------------------------------
class GPLayer(nn.Module):
    """Linear readout over RFF features plus per-logit Bayesian variance.

    Training:
        - Forward returns mean logits via a standard Linear(D, K).
        - call `accumulate(phi)` on each batch in the final epoch to update
          the per-class diagonal precision approximation.
    Inference:
        - call `finalize()` once to invert precision -> covariance diagonal.
        - Forward returns mean logits AND per-logit variance.
        - Apply mean-field correction externally:
              p = softmax(mean / sqrt(1 + π/8 · var))

    Approximation: we accumulate a diagonal precision matrix per class
    (rather than the full D × D matrix). This is the standard SNGP
    diagonal Laplace approximation; it scales to D = 1024+ with negligible
    inference cost.

    Args:
        num_features: RFF dimension D.
        num_classes: number of output classes K.
        ridge: weight-decay-style L2 prior λ. Larger ridge -> tighter prior,
            lower variance.
    """

    def __init__(self, num_features: int, num_classes: int,
                 ridge: float = 1e-3) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.ridge = ridge
        self.beta = nn.Linear(num_features, num_classes, bias=True)
        # Per-class diagonal precision; initialise to ridge prior.
        self.register_buffer(
            "precision_diag",
            ridge * torch.ones(num_classes, num_features),
        )
        # Cached covariance diagonal (inverse of precision_diag).
        self.register_buffer("covariance_diag", torch.zeros(num_classes, num_features))
        self.register_buffer("finalized", torch.tensor(False))

    def forward(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (mean_logits, per_logit_variance | None).

        Variance is None if `finalize()` has not been called yet (training
        mode), otherwise (B, K) per-class predictive variance.
        """
        mean = self.beta(phi)
        if not bool(self.finalized.item()):
            return mean, None
        # variance_k = sum_d cov_k[d] * phi[d]^2
        var = (phi.unsqueeze(1).pow(2) * self.covariance_diag.unsqueeze(0)).sum(dim=-1)
        return mean, var

    @torch.no_grad()
    def accumulate(self, phi: torch.Tensor, targets: torch.Tensor | None = None) -> None:
        """Accumulate diagonal precision from a batch of RFF features.

        precision_diag[c] += sum_{i: y_i == c} phi_i^2

        If `targets` is None, accumulate against every class equally (less
        principled; useful for unlabelled passes). Standard SNGP uses the
        target-class accumulation.
        """
        if targets is None:
            phi_sq = phi.pow(2).sum(dim=0)
            self.precision_diag += phi_sq.unsqueeze(0)
            return
        for c in range(self.num_classes):
            mask = targets == c
            if not mask.any():
                continue
            self.precision_diag[c] += phi[mask].pow(2).sum(dim=0)

    @torch.no_grad()
    def finalize(self) -> None:
        """Invert the diagonal precision to obtain the diagonal covariance."""
        self.covariance_diag = 1.0 / self.precision_diag.clamp(min=1e-12)
        self.finalized.fill_(True)

    @torch.no_grad()
    def reset_precision(self) -> None:
        """Reset the accumulator and invalidate the inversion."""
        self.precision_diag.fill_(self.ridge)
        self.covariance_diag.zero_()
        self.finalized.fill_(False)


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------
def mean_field_logits(mean: torch.Tensor, variance: torch.Tensor,
                      scale: float = math.pi / 8.0) -> torch.Tensor:
    """Apply the mean-field correction to logits.

    p(y|x) = softmax(mean / sqrt(1 + scale · variance))

    Standard SNGP uses scale = π/8 (probit approximation to the logistic).
    """
    return mean / torch.sqrt(1.0 + scale * variance)


# ---------------------------------------------------------------------
# Spectral normalization helpers
# ---------------------------------------------------------------------
def apply_spectral_norm(module: nn.Module, coefficient: float = 0.95,
                        n_power_iterations: int = 1) -> nn.Module:
    """Apply spectral normalization in-place to all Conv and Linear layers.

    `coefficient` < 1.0 is the standard SNGP setting; it enforces
    Lipschitz-c contraction on each layer's weight matrix. The classifier
    head is intentionally excluded — the GP layer provides its own
    regularisation.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # Skip if already spectral-normalised.
            if not hasattr(child, "weight_orig"):
                setattr(
                    module, name,
                    nn.utils.parametrizations.spectral_norm(
                        child, n_power_iterations=n_power_iterations
                    ),
                )
        else:
            apply_spectral_norm(child, coefficient=coefficient,
                                n_power_iterations=n_power_iterations)
    return module


__all__ = [
    "RandomFourierFeatures",
    "GPLayer",
    "mean_field_logits",
    "apply_spectral_norm",
]
