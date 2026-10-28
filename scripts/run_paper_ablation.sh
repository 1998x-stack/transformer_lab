#!/usr/bin/env bash
# Compatibility wrapper for a complete paper-series ablation config.
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT="${1:?Usage: bash scripts/run_paper_ablation.sh VARIANT [SEED]}"
SEED="${2:-42}"
PY="${PYTHON:-python3}"
case "$VARIANT" in
  base|A_heads_1|A_heads_4|A_heads_16|A_heads_32|B_dk_16|B_dk_32|C_layers_2|C_layers_4|C_layers_8|C_dmodel_256|C_dmodel_1024|C_dff_1024|C_dff_4096|D_dropout_0|D_dropout_2|D_smoothing_0|D_smoothing_2|E_learned_position|big) ;;
  *) echo "Unknown paper variant: $VARIANT" >&2; exit 2 ;;
esac
# Generating this suite rewrites work/ablations/plan.json for the selected seed.
"$PY" -m research.ablations --base configs/base_en_de.yaml --out work/ablations --suite paper --seeds "$SEED"
exec "$PY" train.py --config "work/ablations/configs/${VARIANT}-s${SEED}.yaml"
