#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)";MODEL="${1:-$ROOT/outputs/gpt2-medium-stage4-best}";LLAMA_CPP="${LLAMA_CPP:-$ROOT/tools/llama.cpp}";OUT="$ROOT/outputs/gguf";mkdir -p "$OUT"
if [[ ! -d "$LLAMA_CPP/.git" ]];then mkdir -p "$(dirname "$LLAMA_CPP")";git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP";fi
cd "$LLAMA_CPP";git pull --ff-only;cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release;cmake --build build --config Release -j
PY="${PYTHON:-python3}";"$PY" convert_hf_to_gguf.py "$MODEL" --outfile "$OUT/gpt2-medium-stage4-f16.gguf" --outtype f16
Q="";for p in build/bin/llama-quantize build/bin/quantize;do [[ -x "$p" ]]&&Q="$p"&&break;done
if [[ -n "$Q" ]];then "$Q" "$OUT/gpt2-medium-stage4-f16.gguf" "$OUT/gpt2-medium-stage4-q8_0.gguf" Q8_0;"$Q" "$OUT/gpt2-medium-stage4-f16.gguf" "$OUT/gpt2-medium-stage4-q4_k_m.gguf" Q4_K_M;fi
echo "$OUT"
