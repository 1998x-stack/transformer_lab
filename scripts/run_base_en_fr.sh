#!/usr/bin/env bash
set -e
python -u -m transformer_lab.train --config configs/base_en_fr.yaml
python - <<'PY'
from glob import glob
from transformer_lab.utils.checkpoint import average_checkpoints
cks = sorted(glob("work/checkpoints_en_fr_base/step*.pt"))[-5:]
average_checkpoints(cks, "work/checkpoints_en_fr_base/avg_last5.pt")
print("Averaged:", cks)
PY
python -u -m transformer_lab.evaluate --config configs/base_en_fr.yaml --ckpt work/checkpoints_en_fr_base/avg_last5.pt --split test
