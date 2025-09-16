# transformer_lab/models/label_smoothing.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing on logits.
    
    Args:
        classes: vocab size.
        smoothing: epsilon for label smoothing.
        ignore_index: pad id to ignore.
    """
    def __init__(self, classes: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, T, V), target: (B, T)
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        mask = target != self.ignore_index
        if mask.any():
            target = target[mask]
            pred = pred[mask]
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = torch.sum(-true_dist * torch.log_softmax(pred, dim=-1)) / max(1, target.numel())
        return loss