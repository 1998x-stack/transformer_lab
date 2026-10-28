#!/usr/bin/env bash
# Copy/echo smoke demo. Requires local sample_corpus.txt; no automatic downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [[ ! -f sample_corpus.txt ]]; then
  echo "Missing sample_corpus.txt: create a UTF-8 file with one sentence per line." >&2
  exit 2
fi
"$PY" train.py --config configs/copy_en_en.yaml
CHECKPOINT="work/checkpoints_en_en/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "No best.pt found: ensure training reached eval_every steps." >&2
  exit 1
fi
"$PY" decode.py --config configs/copy_en_en.yaml \
  --ckpt "$CHECKPOINT" \
  --src "This is a test sentence."
