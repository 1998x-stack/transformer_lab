import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pickle
from utils.errors import CheckpointCorruptionError


def save_checkpoint(
    checkpoint_path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    step: int = 0,
    loss: float = 0.0,
    additional_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """Save checkpoint with corruption detection."""
    try:
        checkpoint = {
            "model": model_state,
            "step": step,
            "loss": loss,
        }

        if optimizer_state is not None:
            checkpoint["optimizer"] = optimizer_state
        if scheduler_state is not None:
            checkpoint["scheduler"] = scheduler_state
        if additional_info is not None:
            checkpoint.update(additional_info)

        temp_path = str(checkpoint_path) + ".tmp"
        torch.save(checkpoint, temp_path)

        if verify_checkpoint(temp_path):
            os.replace(temp_path, checkpoint_path)
            return True
        else:
            os.remove(temp_path)
            return False

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise CheckpointCorruptionError(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint with fallback to previous checkpoint if corrupted."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if not verify_checkpoint(checkpoint_path):
            raise CheckpointCorruptionError(checkpoint_path=checkpoint_path)

        if model is not None and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])

        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        return checkpoint

    except Exception as e:
        backup_path = find_backup_checkpoint(checkpoint_path)
        if backup_path:
            return load_checkpoint(backup_path, model, optimizer, scheduler, device)
        else:
            raise CheckpointCorruptionError(
                f"Failed to load checkpoint {checkpoint_path} and no backup found: {e}"
            )


def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify checkpoint integrity."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict):
            return False

        if "model" not in checkpoint:
            return False

        if "step" not in checkpoint:
            return False

        if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 0:
            return False

        model_state = checkpoint["model"]
        if not isinstance(model_state, dict):
            return False

        for key, value in model_state.items():
            if not isinstance(key, str):
                return False
            if not isinstance(value, torch.Tensor):
                return False

        return True

    except Exception:
        return False


def find_backup_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Find backup checkpoint (previous step)."""
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    step = extract_step_from_path(checkpoint_path)
    if step is None:
        return None

    prev_step = step - 1
    if prev_step < 0:
        return None

    backup_path = str(path).replace(f"step{step}", f"step{prev_step}")

    if os.path.exists(backup_path) and verify_checkpoint(backup_path):
        return backup_path

    return None


def extract_step_from_path(checkpoint_path: str) -> Optional[int]:
    """Extract step number from checkpoint path."""
    try:
        filename = Path(checkpoint_path).stem
        if filename.startswith("step"):
            return int(filename[4:])
        return None
    except (ValueError, IndexError):
        return None
