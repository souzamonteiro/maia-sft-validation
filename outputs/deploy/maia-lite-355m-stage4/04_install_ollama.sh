#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# MAIA LITE 355M
# REGISTER GGUF F16 MODEL WITH OLLAMA
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${MODEL_NAME:-maia-lite:355m-stage4}"

MODELFILE="${SCRIPT_DIR}/Modelfile"

GGUF_FILE="${SCRIPT_DIR}/gguf/maia-lite-355m-stage4-f16.gguf"


echo "============================================================"
echo "MAIA LITE 355M - OLLAMA INSTALL"
echo "============================================================"
echo


if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: Ollama is not installed."
    exit 1
fi


if [[ ! -f "${MODELFILE}" ]]; then
    echo "ERROR: Modelfile not found:"
    echo "${MODELFILE}"
    exit 1
fi


if [[ ! -f "${GGUF_FILE}" ]]; then
    echo "ERROR: GGUF model not found:"
    echo "${GGUF_FILE}"
    exit 1
fi


echo "Ollama version:"
ollama --version

echo
echo "Creating model:"
echo "  ${MODEL_NAME}"

echo

cd "${SCRIPT_DIR}"

ollama create \
    "${MODEL_NAME}" \
    -f "${MODELFILE}"


echo
echo "============================================================"
echo "MODEL CREATED"
echo "============================================================"

ollama show "${MODEL_NAME}"

echo
echo "Run with:"
echo
echo "  ollama run ${MODEL_NAME}"
echo