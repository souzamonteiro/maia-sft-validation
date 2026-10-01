#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_DIR"; PYTHON="$PROJECT_DIR/.venv/bin/python"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"; export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p outputs; LOG="outputs/stage3-run.log"; exec > >(tee "$LOG") 2>&1
echo "=== STAGE 3: ROBUST GENERALIZATION CONTROL ==="
"$PYTHON" src/audit_stage3.py
"$PYTHON" -m pytest -q tests/test_stage3.py
"$PYTHON" src/evaluate_stage3.py --model openai-community/gpt2-medium --split test_core --output outputs/stage3-baseline-core.json
"$PYTHON" src/evaluate_stage3.py --model openai-community/gpt2-medium --split test_heldout --output outputs/stage3-baseline-heldout.json
"$PYTHON" src/train_stage3.py
"$PYTHON" src/evaluate_stage3.py --model outputs/gpt2-medium-stage3-best --split test_core --output outputs/stage3-post-core.json
"$PYTHON" src/evaluate_stage3.py --model outputs/gpt2-medium-stage3-best --split test_heldout --output outputs/stage3-post-heldout.json
"$PYTHON" src/finalize_stage3.py
if [[ -f scripts/archive_experiment.py ]]; then "$PYTHON" scripts/archive_experiment.py --name stage3-robust-generalization-control-v1 --metrics outputs/stage3-metrics.json --config configs/stage3.json --dataset data/stage3/train.jsonl --dataset data/stage3/validation.jsonl --dataset data/stage3/test_core.jsonl --dataset data/stage3/test_heldout.jsonl --log outputs/stage3-run.log; fi
echo "=== STAGE 3 COMPLETE ==="
