"""Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017).

Trains M independently-initialised copies of the same architecture; averages
softmax outputs at inference. Strong calibration baseline; gold standard for
predictive uncertainty when compute permits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def ensemble_predict(
    models: list[nn.Module],
    x: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average softmax probabilities across an ensemble; return entropy + MI.

    Args:
        models: List of M independently-trained classifiers with identical I/O.
        x: Input batch (B, ...).

    Returns:
        mean_probs: (B, K) ensemble-mean class probabilities.
        entropy: (B,) total predictive entropy in nats.
        mutual_info: (B,) Bayesian mutual information (epistemic component).
    """
    if not models:
        raise ValueError("Empty ensemble.")

    sample_probs: list[torch.Tensor] = []
    for m in models:
        m.eval()
        logits = m(x)
        sample_probs.append(torch.softmax(logits, dim=-1))
    probs_stack = torch.stack(sample_probs, dim=0)        # (M, B, K)

    mean_probs = probs_stack.mean(dim=0)
    eps = 1e-12
    total_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(eps))).sum(-1)
    per_model_entropy = -(probs_stack * torch.log(probs_stack.clamp_min(eps))).sum(-1)
    expected_entropy = per_model_entropy.mean(dim=0)
    mutual_info = total_entropy - expected_entropy

    return (
        mean_probs.cpu().numpy(),
        total_entropy.cpu().numpy(),
        mutual_info.cpu().numpy(),
    )


def load_ensemble(
    checkpoint_paths: list[str | Path],
    model_factory,
    device: str = "cuda",
) -> list[nn.Module]:
    """Load M ensemble members from disk.

    Args:
        checkpoint_paths: Paths to per-member checkpoint files.
        model_factory: Zero-arg callable returning a freshly-constructed model.
        device: Torch device.

    Returns:
        List of loaded models in eval mode.
    """
    models = []
    for p in checkpoint_paths:
        m = model_factory()
        state = torch.load(p, map_location=device)
        m.load_state_dict(state["model"] if "model" in state else state)
        m.to(device).eval()
        models.append(m)
    return models
