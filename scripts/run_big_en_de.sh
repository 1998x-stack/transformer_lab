#!/usr/bin/env bash
set -e
python -u -m transformer_lab.train --config configs/big_en_de.yaml
python - <<'PY'
from glob import glob
from transformer_lab.utils.checkpoint import average_checkpoints
cks = sorted(glob("work/checkpoints_en_de_big/step*.pt"))[-20:]
average_checkpoints(cks, "work/checkpoints_en_de_big/avg_last20.pt")
print("Averaged:", cks)
PY
python -u -m transformer_lab.evaluate --config configs/big_en_de.yaml --ckpt work/checkpoints_en_de_big/avg_last20.pt --split test
