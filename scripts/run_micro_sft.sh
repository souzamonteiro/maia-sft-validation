#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
source .venv/bin/activate
python -m src.audit --data data/micro_train.jsonl --examples 3
pytest -q
python -m src.train --data data/micro_train.jsonl --epochs 20 --max-length 256 --batch-size 1 --grad-accum 8
python -m src.evaluate --model outputs/gpt2-medium-micro-sft --data data/micro_train.jsonl
