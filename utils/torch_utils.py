# transformer_lab/utils/torch_utils.py
from __future__ import annotations
import torch
from typing import Tuple

def subsequent_mask(size: int) -> torch.Tensor:
    """Create subsequent mask for decoder to prevent attending to future positions."""
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent  # True=allowed

def create_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return mask with True for valid (non-pad) tokens."""
    return seq != pad_id

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)