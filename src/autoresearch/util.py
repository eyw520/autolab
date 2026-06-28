import random

import numpy as np
import torch

from autoresearch.interface import Direction


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_better(candidate: float, baseline: float, direction: Direction) -> bool:
    if direction == "min":
        return candidate < baseline
    return candidate > baseline
