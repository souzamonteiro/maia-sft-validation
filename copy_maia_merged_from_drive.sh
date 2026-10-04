#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# MAIA LITE 355M
# COPY MERGED HUGGING FACE MODEL
# GOOGLE DRIVE -> MAC MINI / EXTERNAL SSD
#
# Purpose:
#   Copy the validated merged Hugging Face deployment model
#   from Google Drive to the local Maia deployment directory.
#
# Source:
#   Google Drive / Colab_LLMs / maia-sft-validation /
#   outputs / deploy / maia-lite-355m-stage4 / merged-hf
#
# Destination:
#   External SSD / Projects / maia-sft-validation /
#   outputs / deploy / maia-lite-355m-stage4 / merged-hf
#
# The source is NEVER modified.
# ============================================================


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

GOOGLE_DRIVE_ROOT="/Users/roberto/Library/CloudStorage/GoogleDrive-robertolsmonteiro@gmail.com/Meu Drive/Colab_LLMs"

PROJECT_NAME="maia-sft-validation"

SOURCE_DIR="${GOOGLE_DRIVE_ROOT}/${PROJECT_NAME}/outputs/deploy/maia-lite-355m-stage4/merged-hf"

LOCAL_PROJECT_ROOT="/Volumes/External_SSD/Documentos/Projects/${PROJECT_NAME}"

DEST_DIR="${LOCAL_PROJECT_ROOT}/outputs/deploy/maia-lite-355m-stage4/merged-hf"


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

echo "============================================================"
echo "MAIA LITE 355M"
echo "COPY MERGED MODEL: GOOGLE DRIVE -> MAC MINI"
echo "============================================================"
echo

echo "Source:"
echo "  ${SOURCE_DIR}"
echo

echo "Destination:"
echo "  ${DEST_DIR}"
echo


# ------------------------------------------------------------
# PREFLIGHT
# ------------------------------------------------------------

echo "Checking source..."

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo
    echo "ERROR: Source directory does not exist:"
    echo "  ${SOURCE_DIR}"
    exit 1
fi


if [[ ! -f "${SOURCE_DIR}/config.json" ]]; then
    echo
    echo "ERROR: config.json not found in source model."
    exit 1
fi


if [[ ! -f "${SOURCE_DIR}/model.safetensors" ]]; then
    echo
    echo "ERROR: model.safetensors not found in source model."
    exit 1
fi


if [[ ! -f "${SOURCE_DIR}/tokenizer.json" ]]; then
    echo
    echo "ERROR: tokenizer.json not found in source model."
    exit 1
fi


echo "Source model: OK"


# ------------------------------------------------------------
# SOURCE INFORMATION
# ------------------------------------------------------------

echo
echo "Source size:"
du -sh "${SOURCE_DIR}"

echo
echo "Source files:"
find "${SOURCE_DIR}" -maxdepth 1 -type f -print | sort

echo


# ------------------------------------------------------------
# CREATE DESTINATION
# ------------------------------------------------------------

mkdir -p "${DEST_DIR}"


# ------------------------------------------------------------
# COPY
# ------------------------------------------------------------

echo "============================================================"
echo "COPYING MODEL"
echo "============================================================"
echo

rsync \
    -avh \
    --progress \
    --delete \
    "${SOURCE_DIR}/" \
    "${DEST_DIR}/"


# ------------------------------------------------------------
# DESTINATION CHECK
# ------------------------------------------------------------

echo
echo "============================================================"
echo "VERIFYING DESTINATION"
echo "============================================================"
echo


if [[ ! -f "${DEST_DIR}/config.json" ]]; then
    echo "ERROR: config.json missing after copy."
    exit 1
fi


if [[ ! -f "${DEST_DIR}/model.safetensors" ]]; then
    echo "ERROR: model.safetensors missing after copy."
    exit 1
fi


if [[ ! -f "${DEST_DIR}/tokenizer.json" ]]; then
    echo "ERROR: tokenizer.json missing after copy."
    exit 1
fi


echo "Required files: PASS"


# ------------------------------------------------------------
# BYTE-FOR-BYTE VERIFICATION
# ------------------------------------------------------------

echo
echo "Performing rsync checksum verification..."
echo

if ! rsync \
    -rcn \
    --delete \
    "${SOURCE_DIR}/" \
    "${DEST_DIR}/" \
    | grep -q .; then

    echo "Checksum verification: PASS"

else

    echo "Checksum verification: FAILED"
    echo
    echo "Source and destination differ."
    exit 1

fi


# ------------------------------------------------------------
# SHA-256 MANIFEST
# ------------------------------------------------------------

echo
echo "Creating SHA-256 manifest..."

MANIFEST="${DEST_DIR}/SHA256SUMS"

(
    cd "${DEST_DIR}"

    find . \
        -maxdepth 1 \
        -type f \
        ! -name "SHA256SUMS" \
        -print0 \
        | sort -z \
        | xargs -0 shasum -a 256
) > "${MANIFEST}"


echo
echo "SHA-256 manifest:"
cat "${MANIFEST}"


# ------------------------------------------------------------
# FINAL INFORMATION
# ------------------------------------------------------------

echo
echo "Local model size:"
du -sh "${DEST_DIR}"

echo
echo "============================================================"
echo "COPY COMPLETE: PASS"
echo "============================================================"
echo

echo "Local merged model:"
echo "  ${DEST_DIR}"
echo

echo "SHA-256 manifest:"
echo "  ${MANIFEST}"
echo