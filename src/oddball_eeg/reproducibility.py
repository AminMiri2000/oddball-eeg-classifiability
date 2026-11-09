"""Random seeding and device selection."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Make Python / NumPy / PyTorch behavior as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG used by libraries

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_device() -> torch.device:
    """
    Return CUDA if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
