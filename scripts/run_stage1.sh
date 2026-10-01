#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "ERROR: .venv missing."; exit 1; }

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p outputs/logs
TS="$(date +"%Y%m%d-%H%M%S")"
LOG="outputs/logs/stage1-${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 1 V2: GENERALIZATION CONTROL ==="
"$PYTHON" src/audit_stage1.py
"$PYTHON" -m pytest -q tests/test_stage1.py

echo "=== BASELINE TEST ==="
"$PYTHON" src/evaluate_stage1.py \
  --model openai-community/gpt2-medium \
  --split test \
  --out outputs/stage1-baseline-test.json

echo "=== TRAINING ==="
"$PYTHON" src/train_stage1.py

echo "=== POST-SFT TEST ==="
"$PYTHON" src/evaluate_stage1.py \
  --model outputs/gpt2-medium-stage1 \
  --split test \
  --out outputs/stage1-post-test.json

"$PYTHON" src/finalize_stage1.py

if [[ -f scripts/archive_experiment.py ]]; then
  "$PYTHON" scripts/archive_experiment.py \
    --name stage1-generalization-control-v2 \
    --metrics outputs/stage1-metrics.json \
    --config configs/stage1.json \
    --dataset data/stage1/train.jsonl \
    --dataset data/stage1/validation.jsonl \
    --dataset data/stage1/test.jsonl \
    --log "$LOG"
fi

echo "=== STAGE 1 V2 COMPLETE ==="
