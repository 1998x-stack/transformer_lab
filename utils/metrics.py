# transformer_lab/utils/metrics.py
from __future__ import annotations
import math
from typing import Dict, List
import sacrebleu
from time import time

class SpeedMeter:
    """Compute tokens/sec over a sliding window."""
    def __init__(self):
        self.t0 = time()
        self.tokens = 0

    def update(self, n_tokens: int):
        self.tokens += n_tokens

    def rate(self) -> float:
        dt = max(1e-6, time() - self.t0)
        return self.tokens / dt

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)

def perplexity(loss: float) -> float:
    return math.exp(loss)

def estimate_train_flops(hours: float, num_gpus: int, sustained_tflops_per_gpu: float) -> float:
    """粗略口径：与论文一致：时间 * GPU数 * 每卡持续TFLOPS."""
    return hours * 3600 * num_gpus * sustained_tflops_per_gpu * 1e12
