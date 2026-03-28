import tempfile
from pathlib import Path

import pytest
import torch

from models.label_smoothing import LabelSmoothingLoss
from models.transformer import Transformer
from optim.scheduler import NoamScheduler
from utils.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainConfig,
)


def create_minimal_config(tmpdir, max_steps=10):
    model_cfg = ModelConfig(
        N=2,
        d_model=64,
        d_ff=256,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vocab_size=100,
        activation="relu",
        share_embeddings=True,
        tie_softmax_weight=True,
        pos_encoding="sinusoidal",
        label_smoothing=0.0,
    )
    data_cfg = DataConfig(
        dataset="wmt14",
        lang_pair="de-en",
        max_src_len=16,
        max_tgt_len=16,
        min_len=1,
        vocab_size=100,
        use_shared_vocab=True,
        max_tokens_per_batch=100,
        num_buckets=2,
        cache_dir=None,
        tokenizer_dir=str(tmpdir / "tokenizer"),
    )
    optim_cfg = OptimConfig(
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.0,
        warmup_steps=2,
        max_steps=max_steps,
        grad_clip=1.0,
    )
    runtime_cfg = RuntimeConfig(
        seed=42,
        device="cpu",
        num_workers=0,
        log_dir=str(tmpdir / "logs"),
        ckpt_dir=str(tmpdir / "checkpoints"),
        tb_dir=str(tmpdir / "tensorboard"),
        save_every=5,
        eval_every=5,
        keep_last=5,
        amp=False,
        accumulate_steps=1,
    )

    return TrainConfig(
        model=model_cfg,
        data=data_cfg,
        optim=optim_cfg,
        runtime=runtime_cfg,
    )


@pytest.fixture
def temp_training_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        log_dir = tmpdir / "logs"
        ckpt_dir = tmpdir / "checkpoints"
        tb_dir = tmpdir / "tensorboard"
        tokenizer_dir = tmpdir / "tokenizer"

        log_dir.mkdir()
        ckpt_dir.mkdir()
        tb_dir.mkdir()
        tokenizer_dir.mkdir()

        yield {
            "base": tmpdir,
            "log_dir": log_dir,
            "ckpt_dir": ckpt_dir,
            "tb_dir": tb_dir,
            "tokenizer_dir": tokenizer_dir,
        }


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingLoop:
    def test_training_loop_initialization(self, temp_training_dirs):
        tmpdir = temp_training_dirs["base"]
        config = create_minimal_config(tmpdir, max_steps=3)

        from utils.torch_utils import count_parameters

        model = Transformer(
            vocab_size=config.data.vocab_size,
            N=config.model.N,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            attn_dropout=config.model.attn_dropout,
            activation=config.model.activation,
            share_embeddings=config.model.share_embeddings,
            tie_softmax_weight=config.model.tie_softmax_weight,
            pos_encoding=config.model.pos_encoding,
        ).to(config.runtime.device)

        param_count = count_parameters(model)
        assert param_count > 0, "Model should have parameters"

        opt = torch.optim.Adam(
            model.parameters(),
            lr=config.optim.lr,
            betas=config.optim.betas,
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )

        scheduler = NoamScheduler(
            opt, d_model=config.model.d_model, warmup_steps=config.optim.warmup_steps
        )

        initial_lr = scheduler.get_lr()[0]
        assert initial_lr > 0, "Initial learning rate should be positive"

    def test_training_loop_checkpoint_mechanism(self, temp_training_dirs):
        tmpdir = temp_training_dirs["base"]
        ckpt_dir = tmpdir / "checkpoints"

        config = create_minimal_config(tmpdir, max_steps=2)

        model = Transformer(
            vocab_size=config.data.vocab_size,
            N=config.model.N,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            attn_dropout=config.model.attn_dropout,
            activation=config.model.activation,
            share_embeddings=config.model.share_embeddings,
            tie_softmax_weight=config.model.tie_softmax_weight,
            pos_encoding=config.model.pos_encoding,
        )

        torch.save(
            {"model": model.state_dict(), "step": 1},
            ckpt_dir / "step1.pt",
        )

        assert (ckpt_dir / "step1.pt").exists(), "Checkpoint file should be created"

        loaded = torch.load(ckpt_dir / "step1.pt")
        assert "model" in loaded, "Checkpoint should contain model state"
        assert "step" in loaded, "Checkpoint should contain step number"

    def test_training_loop_loss_computation(self):
        vocab_size = 100
        criterion = LabelSmoothingLoss(
            classes=vocab_size, smoothing=0.0, ignore_index=0
        )

        batch_size, seq_len = 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(1, vocab_size, (batch_size, seq_len))

        loss = criterion(logits, targets)
        assert loss > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_training_loop_scheduler_behavior(self, temp_training_dirs):
        tmpdir = temp_training_dirs["base"]
        config = create_minimal_config(tmpdir, max_steps=5)

        model = Transformer(
            vocab_size=config.data.vocab_size,
            N=config.model.N,
            d_model=config.model.d_model,
            d_ff=config.model.d_ff,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            attn_dropout=config.model.attn_dropout,
            activation=config.model.activation,
            share_embeddings=config.model.share_embeddings,
            tie_softmax_weight=config.model.tie_softmax_weight,
            pos_encoding=config.model.pos_encoding,
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=config.optim.lr,
            betas=config.optim.betas,
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )

        scheduler = NoamScheduler(
            opt, d_model=config.model.d_model, warmup_steps=config.optim.warmup_steps
        )

        lrs = []
        for _ in range(5):
            lr = scheduler.get_lr()[0]
            lrs.append(lr)

        assert len(lrs) == 5, "Should have 5 learning rate values"
        assert all(lr > 0 for lr in lrs), "All LRs should be positive"

    def test_training_loop_gradient_flow(self):
        vocab_size = 100
        model = Transformer(
            vocab_size=vocab_size,
            N=2,
            d_model=64,
            d_ff=256,
            num_heads=4,
            dropout=0.0,
            attn_dropout=0.0,
            activation="relu",
            share_embeddings=True,
            tie_softmax_weight=True,
            pos_encoding="sinusoidal",
        )

        batch_size, seq_len = 2, 8
        src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_in_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)
        loss = logits.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "Model should have gradients after backward pass"
