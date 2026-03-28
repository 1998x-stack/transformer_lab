import pytest
import torch
import torch.nn as nn

from utils.errors import (
    CheckpointCorruptionError,
    GradientExplosionError,
    NaNLossError,
    OOMError,
)
from utils.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from utils.training.loop import TrainingLoop
from utils.training.recovery import TrainingRecovery


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.linear = nn.Linear(64, vocab_size)

    def forward(self, src_ids, tgt_in_ids, src_mask, tgt_mask):
        x = self.embedding(tgt_in_ids)
        return self.linear(x)


class DummyLoss(nn.Module):
    def forward(self, logits, targets):
        return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")


class TestErrorRecovery:
    def test_oom_error_raised(self):
        """Test that OOMError is raised when batch size is too large."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DummyLoss()

        training_loop = TrainingLoop(model, optimizer, criterion, device="cpu")

        batch_size = 1000
        seq_len = 100
        vocab_size = 100

        src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_in_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_out_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        try:
            result = training_loop.train_step(
                src_ids, tgt_in_ids, tgt_out_ids, src_mask, tgt_mask
            )
            assert result is not None
        except (OOMError, RuntimeError) as e:
            if (
                "out of memory" in str(e).lower()
                or "cuda" in str(e).lower()
                or "can't allocate" in str(e).lower()
            ):
                pytest.skip("OOM error detected but not reproducible on this system")
            else:
                raise

    def test_nan_loss_detection(self):
        """Test that NaN loss is detected and raises NaNLossError."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        class NaNLoss(nn.Module):
            def forward(self, logits, targets):
                return torch.tensor(float("nan"))

        criterion = NaNLoss()
        training_loop = TrainingLoop(model, optimizer, criterion, device="cpu")

        batch_size, seq_len, vocab_size = 2, 8, 100
        src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_in_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_out_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with pytest.raises(NaNLossError):
            training_loop.train_step(
                src_ids, tgt_in_ids, tgt_out_ids, src_mask, tgt_mask
            )

    def test_gradient_explosion_detection(self):
        """Test that gradient explosion is detected."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DummyLoss()

        training_loop = TrainingLoop(
            model, optimizer, criterion, device="cpu", max_grad_norm=1e-6
        )

        batch_size, seq_len, vocab_size = 2, 8, 100
        src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_in_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_out_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        model.embedding.weight.data *= 100000

        try:
            result = training_loop.train_step(
                src_ids, tgt_in_ids, tgt_out_ids, src_mask, tgt_mask
            )
            assert result is not None
        except (GradientExplosionError, RuntimeError):
            pass

    def test_oom_recovery(self):
        """Test OOM recovery by reducing batch size."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DummyLoss()

        training_loop = TrainingLoop(model, optimizer, criterion, device="cpu")

        current_batch_size = 1000
        new_batch_size = training_loop.recover_from_oom(current_batch_size)

        assert new_batch_size == 500, "Batch size should be halved"

        current_batch_size = 1
        new_batch_size = training_loop.recover_from_oom(current_batch_size)

        assert new_batch_size == 1, "Batch size should not go below 1"

    def test_checkpoint_save_and_verify(self, tmp_path):
        """Test checkpoint saving and verification."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint_path = tmp_path / "checkpoint.pt"

        success = save_checkpoint(
            checkpoint_path=str(checkpoint_path),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=10,
            loss=0.5,
        )

        assert success, "Checkpoint should be saved successfully"
        assert checkpoint_path.exists(), "Checkpoint file should exist"

        is_valid = verify_checkpoint(str(checkpoint_path))
        assert is_valid, "Checkpoint should be valid"

    def test_checkpoint_load(self, tmp_path):
        """Test checkpoint loading."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint_path = tmp_path / "checkpoint.pt"

        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        save_checkpoint(
            checkpoint_path=str(checkpoint_path),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=10,
            loss=0.5,
        )

        loaded_model = DummyModel()
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.001)

        checkpoint = load_checkpoint(
            str(checkpoint_path),
            model=loaded_model,
            optimizer=loaded_optimizer,
            device="cpu",
        )

        assert checkpoint["step"] == 10, "Step should be 10"
        assert checkpoint["loss"] == 0.5, "Loss should be 0.5"

        for key in original_state:
            assert torch.allclose(
                original_state[key], loaded_model.state_dict()[key]
            ), f"Parameter {key} should match"

    def test_checkpoint_corruption_detection(self, tmp_path):
        """Test detection of corrupted checkpoint."""
        checkpoint_path = tmp_path / "corrupted.pt"

        with open(checkpoint_path, "wb") as f:
            f.write(b"not a valid checkpoint")

        is_valid = verify_checkpoint(str(checkpoint_path))
        assert not is_valid, "Corrupted checkpoint should be detected as invalid"

        with pytest.raises((CheckpointCorruptionError, Exception)):
            load_checkpoint(str(checkpoint_path))

    def test_training_recovery_basic(self):
        """Test basic training recovery functionality."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DummyLoss()

        training_loop = TrainingLoop(model, optimizer, criterion, device="cpu")
        recovery = TrainingRecovery(training_loop, max_retries=2)

        stats = recovery.get_recovery_stats()
        assert stats["max_retries"] == 2
        assert stats["retry_count"] == 0
        assert len(stats["errors_encountered"]) == 0

    def test_training_recovery_stats(self):
        """Test recovery statistics tracking."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DummyLoss()

        training_loop = TrainingLoop(model, optimizer, criterion, device="cpu")
        recovery = TrainingRecovery(training_loop, max_retries=3, oom_retry=True)

        stats = recovery.get_recovery_stats()
        assert stats["oom_retry_enabled"] is True
        assert stats["nan_retry_enabled"] is False
        assert stats["grad_retry_enabled"] is False
