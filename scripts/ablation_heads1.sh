#!/usr/bin/env bash
# Backward-compatible entrypoint; use generated complete configs, not partial legacy YAML.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_paper_ablation.sh A_heads_1 "$@"
