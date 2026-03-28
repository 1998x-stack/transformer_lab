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


def resolve_device(name: str | None) -> torch.device:
    """Resolve a config device name to a real device with backend fallback.

    Falls CUDA -> MPS -> CPU and MPS -> CPU when the requested backend is
    unavailable; auto-picks a backend when ``name`` is ``None``.
    """
    if name is not None:
        if name == "cuda" and not torch.cuda.is_available():
            name = "mps" if torch.backends.mps.is_available() else "cpu"
        elif name == "mps" and not torch.backends.mps.is_available():
            name = "cpu"
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
