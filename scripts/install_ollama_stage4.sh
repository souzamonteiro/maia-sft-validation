#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";G="$ROOT/outputs/gguf";[[ -f "$G/gpt2-medium-stage4-q8_0.gguf" ]]||{ echo "Run convert_stage4_gguf.sh first";exit 1;};cp "$ROOT/deployment/Modelfile" "$G/Modelfile";cd "$G";ollama create gpt2-medium-stage4 -f Modelfile;ollama run gpt2-medium-stage4 "Introduce yourself briefly."
