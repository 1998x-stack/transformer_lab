#!/usr/bin/env bash
# Train the base Transformer on Grimms' Fairy Tales (copy/echo task) and verify it echoes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo ">> Installing dependencies (if needed)"
"$PY" -m pip install -q -r requirements.txt

echo ">> Training copy/echo model on sample_corpus.txt..."
"$PY" train.py --config configs/copy_en_en.yaml

echo ">> Echo demo (expected: model reproduces the input sentence)"
examples=(
  "There was once a poor man who lived in the forest."
  "The king had a beautiful daughter with long golden hair."
  "In the morning the little bird began to sing very sweetly."
)
for s in "${examples[@]}"; do
  echo "in : $s"
  out=$("$PY" decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "$s")
  echo "out: $out"
  echo
done