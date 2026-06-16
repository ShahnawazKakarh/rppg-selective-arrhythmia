"""Evidential Deep Learning (Sensoy et al., NeurIPS 2018).

Model outputs Dirichlet concentration parameters alpha = evidence + 1; uncertainty
is K / sum(alpha) where K is the number of classes. Single-pass alternative to
ensembles, with explicit epistemic + aleatoric decomposition.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Replace softmax output with non-negative evidence (alpha = evidence + 1).

    Wraps a backbone whose `features(x)` returns a (B, d_model) representation.
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Softplus enforces non-negative evidence; alpha = evidence + 1.
        return F.softplus(self.linear(features)) + 1.0


def edl_mse_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    epoch: int,
    annealing_steps: int = 10,
    kl_weight: float | None = None,
) -> torch.Tensor:
    """EDL Type-II MSE loss with annealed KL regularizer (Sensoy et al., Eqs. 5 & 9).

    Args:
        alpha: (B, K) Dirichlet concentrations from EvidentialHead.
        targets: (B,) integer class labels.
        num_classes: K.
        epoch: Current epoch index (for KL annealing).
        annealing_steps: Number of epochs over which the KL weight is annealed
            from 0 to 1.
        kl_weight: Optional override for the annealed KL weight.

    Returns:
        Scalar loss.
    """
    onehot = F.one_hot(targets, num_classes=num_classes).float()
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / S
    # MSE term + variance term (per Sensoy et al. Eq. 5).
    err = (onehot - prob).pow(2).sum(dim=1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1)
    mse = (err + var).mean()

    # KL divergence to the uninformative prior, on non-target classes.
    alpha_tilde = onehot + (1 - onehot) * alpha
    kl = _kl_dirichlet_uniform(alpha_tilde, num_classes).mean()

    if kl_weight is None:
        kl_weight = min(1.0, float(epoch) / max(annealing_steps, 1))
    return mse + kl_weight * kl


def _kl_dirichlet_uniform(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1, ..., 1)) — closed-form for the uniform prior."""
    K = num_classes
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha).squeeze(1)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha).sum(dim=1)
    )
    second = (
        (alpha - 1)
        * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1)
    return first + second


def edl_uncertainty(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample probabilities and Dirichlet-based total uncertainty.

    Uncertainty u = K / sum(alpha); high u = low total evidence.
    """
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    u = alpha.size(1) / S.squeeze(1)
    return probs, u
