import time
from typing import Optional, Dict, Any, Callable
from utils.errors import OOMError, NaNLossError, GradientExplosionError, TrainingError
from utils.training.loop import TrainingLoop


class TrainingRecovery:
    """Training recovery orchestrator with exponential backoff."""

    def __init__(
        self,
        training_loop: TrainingLoop,
        max_retries: int = 3,
        oom_retry: bool = True,
        nan_retry: bool = False,
        grad_retry: bool = False,
        backoff_factor: float = 2.0,
        max_backoff: int = 60,
    ):
        self.training_loop = training_loop
        self.max_retries = max_retries
        self.oom_retry = oom_retry
        self.nan_retry = nan_retry
        self.grad_retry = grad_retry
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retry_count = 0
        self.errors_encountered = []

    def train_with_recovery(
        self,
        dataloader,
        max_steps: int = 1000,
        on_error: Optional[Callable[[Exception, int], None]] = None,
    ) -> Dict[str, Any]:
        """Train with automatic error recovery and exponential backoff."""
        step = 0
        self.retry_count = 0
        self.errors_encountered = []
        backoff_time = 1

        while step < max_steps and self.retry_count < self.max_retries:
            try:
                return self.training_loop.train_with_recovery(
                    dataloader=dataloader,
                    max_steps=max_steps,
                    max_retries=self.max_retries,
                    oom_retry=self.oom_retry,
                    nan_retry=self.nan_retry,
                    grad_retry=self.grad_retry,
                )

            except OOMError as e:
                if not self.oom_retry:
                    raise
                self._handle_error(e, on_error)
                backoff_time = self._apply_backoff(backoff_time)

            except NaNLossError as e:
                if not self.nan_retry:
                    raise
                self._handle_error(e, on_error)
                backoff_time = self._apply_backoff(backoff_time)

            except GradientExplosionError as e:
                if not self.grad_retry:
                    raise
                self._handle_error(e, on_error)
                backoff_time = self._apply_backoff(backoff_time)

            except TrainingError as e:
                self._handle_error(e, on_error)
                raise

        return {
            "status": "failed_after_retries",
            "steps": step,
            "retry_count": self.retry_count,
            "errors": self.errors_encountered,
        }

    def _handle_error(
        self, error: Exception, on_error: Optional[Callable[[Exception, int], None]]
    ):
        """Handle error and call callback if provided."""
        self.retry_count += 1
        self.errors_encountered.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "retry": self.retry_count,
            }
        )

        if on_error is not None:
            on_error(error, self.retry_count)

    def _apply_backoff(self, current_backoff: float) -> float:
        """Apply exponential backoff."""
        new_backoff = min(current_backoff * self.backoff_factor, self.max_backoff)
        time.sleep(current_backoff)
        return new_backoff

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "retry_count": self.retry_count,
            "errors_encountered": self.errors_encountered,
            "max_retries": self.max_retries,
            "oom_retry_enabled": self.oom_retry,
            "nan_retry_enabled": self.nan_retry,
            "grad_retry_enabled": self.grad_retry,
        }
