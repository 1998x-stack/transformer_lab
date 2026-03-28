import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from utils.config import (
    TrainConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    RuntimeConfig,
)
from models.transformer import Transformer


@pytest.fixture
def base_config():
    model_cfg = ModelConfig(
        N=2,
        d_model=64,
        d_ff=256,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vocab_size=1000,
    )
    data_cfg = DataConfig(
        dataset="wmt14",
        lang_pair="en-de",
        max_src_len=32,
        max_tgt_len=32,
        vocab_size=1000,
        max_tokens_per_batch=1000,
    )
    optim_cfg = OptimConfig(
        lr=1e-3,
        warmup_steps=100,
        max_steps=1000,
    )
    runtime_cfg = RuntimeConfig(
        seed=42,
        device="cpu",
        num_workers=0,
    )

    return TrainConfig(
        model=model_cfg,
        data=data_cfg,
        optim=optim_cfg,
        runtime=runtime_cfg,
    )


@pytest.fixture
def small_transformer(base_config):
    cfg = base_config.model
    model = Transformer(
        vocab_size=cfg.vocab_size,
        N=cfg.N,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        activation=cfg.activation,
        share_embeddings=cfg.share_embeddings,
        tie_softmax_weight=cfg.tie_softmax_weight,
        pos_encoding=cfg.pos_encoding,
        max_len=base_config.data.max_src_len,
    )
    return model


@pytest.fixture
def sample_batch(base_config):
    batch_size = 4
    src_len = 16
    tgt_len = 16
    vocab_size = base_config.model.vocab_size

    src_ids = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt_in_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))
    tgt_out_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))

    src_key_padding_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    tgt_key_padding_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool)

    return {
        "src_ids": src_ids,
        "tgt_in_ids": tgt_in_ids,
        "tgt_out_ids": tgt_out_ids,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
    }


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        log_dir = tmpdir / "logs"
        ckpt_dir = tmpdir / "checkpoints"
        tb_dir = tmpdir / "tensorboard"
        tokenizer_dir = tmpdir / "tokenizer"
        cache_dir = tmpdir / "cache"

        log_dir.mkdir()
        ckpt_dir.mkdir()
        tb_dir.mkdir()
        tokenizer_dir.mkdir()
        cache_dir.mkdir()

        yield {
            "base": tmpdir,
            "log_dir": log_dir,
            "ckpt_dir": ckpt_dir,
            "tb_dir": tb_dir,
            "tokenizer_dir": tokenizer_dir,
            "cache_dir": cache_dir,
        }
