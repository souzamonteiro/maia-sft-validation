#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# MAIA LITE 355M
# HUGGING FACE -> GGUF F16
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_CPP_DIR="/Volumes/External_SSD/Documentos/Projects/llama.cpp"

MODEL_DIR="${SCRIPT_DIR}/merged-hf"

GGUF_DIR="${SCRIPT_DIR}/gguf"

GGUF_FILE="${GGUF_DIR}/maia-lite-355m-stage4-f16.gguf"


echo "============================================================"
echo "MAIA LITE 355M - GGUF F16 CONVERSION"
echo "============================================================"
echo

echo "llama.cpp:"
echo "  ${LLAMA_CPP_DIR}"

echo
echo "Model:"
echo "  ${MODEL_DIR}"

echo
echo "Output:"
echo "  ${GGUF_FILE}"

echo


if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
    echo "ERROR: llama.cpp directory not found."
    exit 1
fi


if [[ ! -f "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" ]]; then
    echo "ERROR: convert_hf_to_gguf.py not found."
    exit 1
fi


if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: merged-hf directory not found."
    exit 1
fi


if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: config.json not found."
    exit 1
fi


mkdir -p "${GGUF_DIR}"


echo "Using local llama.cpp checkout (no git pull)."


echo
echo "Installing converter requirements..."

python3 -m pip install \
    -r "${LLAMA_CPP_DIR}/requirements.txt"


echo
echo "Converting Maia Lite to GGUF F16..."

python3 \
    "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${MODEL_DIR}" \
    --outfile "${GGUF_FILE}" \
    --outtype f16


if [[ ! -f "${GGUF_FILE}" ]]; then
    echo
    echo "ERROR: GGUF file was not created."
    exit 1
fi


echo
echo "============================================================"
echo "GGUF CONVERSION COMPLETE"
echo "============================================================"

ls -lh "${GGUF_FILE}"

echo
echo "SHA-256:"
sha256sum "${GGUF_FILE}" 2>/dev/null \
    || shasum -a 256 "${GGUF_FILE}"