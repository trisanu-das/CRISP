"""Deterministic seeding across python / numpy / torch (+ CUDA)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Seeding numpy/python is still useful even before torch is installed
        # (e.g. for the pure-Python unit tests in tests/).
        pass
