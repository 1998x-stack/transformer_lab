# transformer_lab/optim/scheduler.py
from __future__ import annotations
import torch
from typing import Optional

class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Noam lr schedule: d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)."""
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        self._step_num = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self._step_num += 1
        scale = (self.d_model ** -0.5) * min(self._step_num ** -0.5, self._step_num * (self.warmup ** -1.5))
        return [base_lr * 0 + scale for base_lr in self.base_lrs]  # ignore base_lr, use scale
