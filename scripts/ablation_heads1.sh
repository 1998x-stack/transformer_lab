#!/usr/bin/env bash
set -e
# 使用 EN-DE Base + 覆盖头数=1
python -u -m transformer_lab.train --config configs/base_en_de.yaml
python -u -m transformer_lab.train --config configs/ablations/heads_1.yaml
