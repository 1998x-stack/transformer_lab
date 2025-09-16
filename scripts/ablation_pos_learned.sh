#!/usr/bin/env bash
set -e
python -u -m transformer_lab.train --config configs/base_en_de.yaml
python -u -m transformer_lab.train --config configs/ablations/pos_learned.yaml
