#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs
LOG="outputs/stage2-run.log"
exec > >(tee "$LOG") 2>&1

echo "=== STAGE 2: MULTILINGUAL CONTROL ==="
"$PYTHON" src/audit_stage2.py
"$PYTHON" -m pytest -q tests/test_stage2.py

echo "=== BASELINE MULTILINGUAL TEST ==="
"$PYTHON" src/evaluate_stage2.py --model openai-community/gpt2-medium --output outputs/stage2-baseline-test.json

echo "=== TRAINING ==="
"$PYTHON" src/train_stage2.py

echo "=== POST-SFT TEST (BEST VALIDATION CHECKPOINT) ==="
"$PYTHON" src/evaluate_stage2.py --model outputs/gpt2-medium-stage2-best --output outputs/stage2-post-test.json

"$PYTHON" src/finalize_stage2.py

if [[ -f scripts/archive_experiment.py ]]; then
  "$PYTHON" scripts/archive_experiment.py \
    --name stage2-multilingual-control-v1 \
    --metrics outputs/stage2-metrics.json \
    --config configs/stage2.json \
    --dataset data/stage2/train.jsonl \
    --dataset data/stage2/validation.jsonl \
    --dataset data/stage2/test.jsonl \
    --log outputs/stage2-run.log
else
  echo "WARNING: scripts/archive_experiment.py not found; skipping archive."
fi
echo "=== STAGE 2 COMPLETE ==="
