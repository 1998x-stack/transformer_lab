# transformer_lab/utils/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
import yaml
import os

@dataclass
class OptimConfig:
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.0
    warmup_steps: int = 4000
    max_steps: int = 100_000
    grad_clip: float = 1.0

@dataclass
class ModelConfig:
    N: int = 6
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.0
    activation: Literal["relu", "gelu"] = "relu"
    share_embeddings: bool = True
    tie_softmax_weight: bool = True
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    label_smoothing: float = 0.1
    vocab_size: int = 37000  # WMT14 En-De 默认约 37k

@dataclass
class DataConfig:
    dataset: Literal["wmt14", "wmt16", "opus100"] = "wmt14"
    lang_pair: Literal["en-de", "de-en", "en-fr", "fr-en"] = "en-de"
    max_src_len: int = 256
    max_tgt_len: int = 256
    min_len: int = 1
    tokenizer_dir: str = "work/tokenizer"
    vocab_size: int = 37000
    use_shared_vocab: bool = True
    # 动态 token 配额；接近论文的 “25k 源 + 25k 目标 token/批”
    max_tokens_per_batch: int = 50_000
    num_buckets: int = 8
    cache_dir: Optional[str] = None

@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    log_dir: str = "work/logs"
    ckpt_dir: str = "work/checkpoints"
    tb_dir: str = "work/tensorboard"
    save_every: int = 1000
    eval_every: int = 2000
    keep_last: int = 10
    amp: bool = True
    accumulate_steps: int = 1

@dataclass
class DecodeConfig:
    beam_size: int = 4
    length_penalty: float = 0.6
    max_len_offset: int = 50
    max_len_ratio: float = 1.0  # max_len = src_len * ratio + offset

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

def load_config(path: str) -> TrainConfig:
    """Load YAML to dataclasses config."""
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    # 合并 vocab 大小：若在 data 指定，覆盖 model.vocab_size
    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    if data_cfg.vocab_size:
        model_cfg.vocab_size = data_cfg.vocab_size
    optim_cfg = OptimConfig(**cfg_dict.get("optim", {}))
    runtime_cfg = RuntimeConfig(**cfg_dict.get("runtime", {}))
    decode_cfg = DecodeConfig(**cfg_dict.get("decode", {}))
    return TrainConfig(model=model_cfg, data=data_cfg, optim=optim_cfg, runtime=runtime_cfg, decode=decode_cfg)
