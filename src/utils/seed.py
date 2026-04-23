"""
Reproducibility utilities.

Ensures deterministic behavior across runs by seeding all relevant
random number generators used in the pipeline.
"""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: seed PyTorch if available (for HuggingFace models)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
