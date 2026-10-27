"""Factory mode: all entrypoints resolve components through a validated registry.

Register builders in one place; never infer implementation from filename or eval().
"""
from __future__ import annotations
from typing import Callable, Dict, Generic, TypeVar
import torch
from torch import nn
from models.transformer import Transformer
from models.label_smoothing import LabelSmoothingLoss
from optim.scheduler import NoamScheduler

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def decorator(builder: Callable[..., T]):
            if name in self._builders:
                raise ValueError(f"Duplicate {self.name}: {name}")
            self._builders[name] = builder
            return builder
        return decorator

    def build(self, name: str, *args, **kwargs) -> T:
        if name not in self._builders:
            raise ValueError(f"Unknown {self.name} {name!r}; valid: {sorted(self._builders)}")
        return self._builders[name](*args, **kwargs)

MODEL_FACTORY = Registry[nn.Module]("architecture")
OPTIMIZER_FACTORY = Registry[torch.optim.Optimizer]("optimizer")
LOSS_FACTORY = Registry[nn.Module]("loss")
SCHEDULER_FACTORY = Registry("scheduler")

@MODEL_FACTORY.register("transformer")
def _transformer(cfg, vocab_size: int) -> Transformer:
    if cfg.attention_kind not in ("scaled_dot", "unscaled_dot"):
        raise ValueError(f"Unknown attention_kind: {cfg.attention_kind}")
    return Transformer(
        vocab_size=vocab_size, N=cfg.N, d_model=cfg.d_model, d_ff=cfg.d_ff,
        num_heads=cfg.num_heads, dropout=cfg.dropout, attn_dropout=cfg.attn_dropout,
        activation=cfg.activation, share_embeddings=cfg.share_embeddings,
        tie_softmax_weight=cfg.tie_softmax_weight, pos_encoding=cfg.pos_encoding,
        d_k=cfg.d_k, d_v=cfg.d_v,
        attention_scale=(cfg.attention_kind == "scaled_dot"),
        norm_style=cfg.norm_style, final_norm=cfg.final_norm,
    )

@OPTIMIZER_FACTORY.register("adam")
def _adam(cfg, parameters):
    return torch.optim.Adam(parameters, lr=cfg.lr, betas=tuple(cfg.betas),
                            eps=cfg.eps, weight_decay=cfg.weight_decay)

@LOSS_FACTORY.register("label_smoothing")
def _label_smoothing(vocab_size: int, smoothing: float, pad_id: int):
    return LabelSmoothingLoss(classes=vocab_size, smoothing=smoothing, ignore_index=pad_id)

@SCHEDULER_FACTORY.register("noam")
def _noam(optimizer, cfg, d_model):
    return NoamScheduler(optimizer, d_model=d_model, warmup_steps=cfg.warmup_steps)

def build_model(cfg, vocab_size: int) -> nn.Module:
    return MODEL_FACTORY.build(cfg.architecture, cfg, vocab_size)

def build_training_components(cfg, model, vocab_size: int, pad_id: int):
    opt = OPTIMIZER_FACTORY.build(cfg.optim.optimizer, cfg.optim, model.parameters())
    sched = SCHEDULER_FACTORY.build(cfg.optim.scheduler, opt, cfg.optim, cfg.model.d_model)
    loss = LOSS_FACTORY.build("label_smoothing", vocab_size, cfg.model.label_smoothing, pad_id)
    return opt, sched, loss
