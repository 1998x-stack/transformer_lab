# transformer_lab/utils/logging_utils.py
from __future__ import annotations
from loguru import logger
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def setup_logging(log_dir: str) -> SummaryWriter:
    """Set up Loguru file sink and TensorBoard writer."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(Path(log_dir) / "train.log", rotation="10 MB", retention=10, enqueue=True)
    logger.add(lambda msg: print(msg, end=""))
    tb = SummaryWriter(log_dir)
    return tb