# transformer_lab/utils/distributed.py
from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0