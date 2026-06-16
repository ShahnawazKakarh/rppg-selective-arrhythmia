"""Deterministic seeding across numpy, random, and torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed numpy, python random, and torch (if installed) for reproducibility.

    Args:
        seed: Integer seed.
        deterministic: If True, set torch.backends.cudnn flags for determinism
            at the cost of some throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
