from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    return torch.device("cpu")
