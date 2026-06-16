"""Monte-Carlo Dropout (Gal & Ghahramani, ICML 2016).

Keeps dropout active at inference time; T stochastic forward passes give a
posterior approximation. Predictive entropy and mutual information are the
standard uncertainty summaries.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def enable_dropout(model: nn.Module) -> None:
    """Put all Dropout modules into training mode while leaving the rest in eval.

    This is the canonical MC-Dropout inference setting: BatchNorm and other
    training-only behaviour stay frozen, but dropout fires.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run T stochastic forward passes; return mean probs, entropy, mutual info.

    Args:
        model: Trained classifier with dropout layers.
        x: Input batch (B, ...).
        num_samples: Number of MC samples T.

    Returns:
        mean_probs: (B, K) posterior-predictive mean over softmax outputs.
        entropy: (B,) predictive entropy in nats.
        mutual_info: (B,) Bayesian mutual information (epistemic uncertainty).
    """
    model.eval()
    enable_dropout(model)

    sample_probs: list[torch.Tensor] = []
    for _ in range(num_samples):
        logits = model(x)
        sample_probs.append(torch.softmax(logits, dim=-1))
    probs_stack = torch.stack(sample_probs, dim=0)        # (T, B, K)

    mean_probs = probs_stack.mean(dim=0)                  # (B, K)
    eps = 1e-12
    # Total (predictive) entropy.
    total_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(eps))).sum(-1)
    # Expected per-sample entropy (aleatoric).
    per_sample_entropy = -(probs_stack * torch.log(probs_stack.clamp_min(eps))).sum(-1)
    expected_entropy = per_sample_entropy.mean(dim=0)     # (B,)
    # Mutual information = total - expected (epistemic).
    mutual_info = total_entropy - expected_entropy

    return (
        mean_probs.cpu().numpy(),
        total_entropy.cpu().numpy(),
        mutual_info.cpu().numpy(),
    )
