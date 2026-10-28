#!/usr/bin/env bash
# Train, average the last 5 step checkpoints, and evaluate a translation preset.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
CONFIG="configs/base_en_fr.yaml"
CHECKPOINT_DIR="work/checkpoints_en_fr_base"
"$PY" train.py --config "$CONFIG"
"$PY" - "$CHECKPOINT_DIR" "5" <<'PY'
from pathlib import Path
import sys
from utils.checkpoint import average_checkpoints

directory = Path(sys.argv[1])
count = int(sys.argv[2])
checkpoints = sorted(directory.glob("step*.pt"), key=lambda p: int(p.stem.removeprefix("step")))
if not checkpoints:
    raise SystemExit(f"No step checkpoints in {directory}; check save_every and max_steps.")
latest = checkpoints[-count:]
output = directory / f"avg_last{count}.pt"
average_checkpoints([str(p) for p in latest], str(output))
print(f"Averaged {len(latest)} checkpoints to {output}")
PY
"$PY" evaluate.py --config "$CONFIG" \
  --ckpt "$CHECKPOINT_DIR/avg_last5.pt" --split test
