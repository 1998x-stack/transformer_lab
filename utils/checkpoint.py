# transformer_lab/utils/checkpoint.py
from __future__ import annotations
from pathlib import Path
import torch
from typing import Dict, List

def save_checkpoint(path: str, state: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def average_checkpoints(ckpt_paths: List[str], out_path: str):
    avg_state = None
    n = len(ckpt_paths)
    for p in ckpt_paths:
        state = torch.load(p, map_location="cpu")
        model_state = state["model"]
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in model_state.items()}
        else:
            for k in avg_state:
                avg_state[k] += model_state[k].float()
    for k in avg_state:
        avg_state[k] /= n
    torch.save({"model": avg_state}, out_path)
