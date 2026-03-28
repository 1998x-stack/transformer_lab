import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from utils.errors import OOMError, NaNLossError, GradientExplosionError


class TrainingLoop:
    """Training loop with error recovery mechanisms."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.current_batch_size = None

    def train_step(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        tgt_out_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform a single training step with error handling."""
        try:
            src_ids = src_ids.to(self.device)
            tgt_in_ids = tgt_in_ids.to(self.device)
            tgt_out_ids = tgt_out_ids.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(src_ids, tgt_in_ids, src_mask, tgt_mask)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out_ids.reshape(-1)
            )

            if torch.isnan(loss):
                raise NaNLossError(step=None)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                raise GradientExplosionError(
                    grad_norm=grad_norm.item(), threshold=self.max_grad_norm
                )

            self.optimizer.step()

            return {"loss": loss.item(), "grad_norm": grad_norm.item()}

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                raise OOMError(current_batch_size=src_ids.size(0))
            raise

    def recover_from_oom(self, current_batch_size: int) -> int:
        """Recover from OOM by reducing batch size."""
        new_batch_size = max(1, current_batch_size // 2)
        return new_batch_size

    def train_with_recovery(
        self,
        dataloader,
        max_steps: int = 1000,
        max_retries: int = 3,
        oom_retry: bool = True,
        nan_retry: bool = False,
        grad_retry: bool = False,
    ) -> Dict[str, Any]:
        """Train with automatic error recovery."""
        step = 0
        retry_count = 0
        batch_size = None
        losses = []

        while step < max_steps and retry_count < max_retries:
            try:
                for batch in dataloader:
                    if batch_size is None:
                        batch_size = batch["src_ids"].size(0)

                    result = self.train_step(
                        batch["src_ids"],
                        batch["tgt_in_ids"],
                        batch["tgt_out_ids"],
                        batch.get(
                            "src_mask",
                            torch.ones_like(batch["src_ids"], dtype=torch.bool),
                        ),
                        batch.get(
                            "tgt_mask",
                            torch.ones_like(batch["tgt_in_ids"], dtype=torch.bool),
                        ),
                    )

                    losses.append(result["loss"])
                    step += 1

                    if step >= max_steps:
                        break

                return {
                    "status": "completed",
                    "steps": step,
                    "losses": losses,
                    "final_batch_size": batch_size,
                }

            except OOMError as e:
                if not oom_retry:
                    raise
                retry_count += 1
                batch_size = self.recover_from_oom(e.current_batch_size or batch_size)

                if hasattr(dataloader, "batch_sampler"):
                    dataloader.batch_sampler.batch_size = batch_size

            except (NaNLossError, GradientExplosionError):
                if not (nan_retry or grad_retry):
                    raise
                retry_count += 1
                self.optimizer.zero_grad(set_to_none=True)

        return {
            "status": "failed",
            "steps": step,
            "losses": losses,
            "final_batch_size": batch_size,
        }
