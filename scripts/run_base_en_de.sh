#!/usr/bin/env bash
set -e
python -u -m transformer_lab.train --config configs/base_en_de.yaml
# 训练完成后，取最近 5 个平均
python - <<'PY'
from glob import glob
from transformer_lab.utils.checkpoint import average_checkpoints
cks = sorted(glob("work/checkpoints_en_de_base/step*.pt"))[-5:]
average_checkpoints(cks, "work/checkpoints_en_de_base/avg_last5.pt")
print("Averaged:", cks)
PY
# BLEU 测试
python -u -m transformer_lab.evaluate --config configs/base_en_de.yaml --ckpt work/checkpoints_en_de_base/avg_last5.pt --split test
