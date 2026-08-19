#!/usr/bin/env bash
# Train the base Transformer on Grimms' Fairy Tales (copy/echo task) and verify it echoes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo ">> Installing dependencies (if needed)"
"$PY" -m pip install -q -r requirements.txt

echo ">> Training copy/echo model on sample_corpus.txt..."
"$PY" train.py --config configs/copy_en_en.yaml

echo ">> Echo demo (model reproduces input; cleanest on corpus-like sentences, approximate on novel phrasing)"
examples=(
  "Of all the ladies in the land,"
  "Alas! alas! if thy mother knew it,"
  "And took my bones that they might lie"
)
for s in "${examples[@]}"; do
  echo "in : $s"
  out=$("$PY" decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "$s")
  echo "out: $out"
  echo
done