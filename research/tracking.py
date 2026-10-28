"""Auditable, non-destructive experiment manifests and JSONL metrics."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import torch


def config_fingerprint(config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        return None


class RunRecorder:
    """Runs have distinct output directories, even for identical config and seed."""

    def __init__(self, cfg, parameters: int):
        digest = config_fingerprint(cfg)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{stamp}-{digest[:10]}-seed{cfg.runtime.seed}"
        self.directory = Path(cfg.runtime.log_dir) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metrics_path = self.directory / "metrics.jsonl"
        manifest = {
            "schema_version": 1, "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": digest, "git_commit": git_sha(),
            "torch_version": str(torch.__version__), "parameters": parameters,
            "config": asdict(cfg),
            "eval_scope": "sample of up to 10 validation batches; not full BLEU"
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def record(self, split: str, step: int, **metrics: float) -> None:
        record = {"schema_version": 1, "split": split, "step": step,
                  "time_utc": datetime.now(timezone.utc).isoformat(),
                  "metrics": {k: float(v) for k, v in metrics.items()}}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
