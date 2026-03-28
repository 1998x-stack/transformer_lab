"""Custom exceptions for transformer_lab."""


class TransformerLabError(Exception):
    """Base exception for transformer_lab."""

    pass


class TrainingError(TransformerLabError):
    """Base exception for training-related errors."""

    pass


class OOMError(TrainingError):
    """Raised when out of memory error occurs during training."""

    def __init__(self, message="Out of memory error", current_batch_size=None):
        super().__init__(message)
        self.current_batch_size = current_batch_size


class NaNLossError(TrainingError):
    """Raised when loss becomes NaN during training."""

    def __init__(self, message="Loss became NaN", step=None):
        super().__init__(message)
        self.step = step


class GradientExplosionError(TrainingError):
    """Raised when gradient norm exceeds threshold."""

    def __init__(
        self, message="Gradient explosion detected", grad_norm=None, threshold=None
    ):
        super().__init__(message)
        self.grad_norm = grad_norm
        self.threshold = threshold


class CheckpointError(TransformerLabError):
    """Raised when checkpoint operations fail."""

    pass


class CheckpointCorruptionError(CheckpointError):
    """Raised when checkpoint file is corrupted."""

    def __init__(self, message="Checkpoint is corrupted", checkpoint_path=None):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path
