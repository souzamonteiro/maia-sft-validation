#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 - SAFE RUNNER V4
#
# PHASE 1:
#   - Verify frozen V4 corpus
#   - Verify SHA-256 fingerprints
#   - Build frozen benchmark
#   - Audit dataset
#   - Audit response-only masking
#   - Run automated tests
#
# IMPORTANT:
#   THIS SCRIPT DOES NOT START TRAINING.
#
# Frozen corpus:
#
#   EN: 7,200
#   PT: 5,400
#   ES: 5,400
#   TOTAL: 18,000
#
# ============================================================


ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="$ROOT/.venv/bin/python"

DATA_DIR="$ROOT/data/stage4"

TRAIN_FILE="$DATA_DIR/train.jsonl"
VALIDATION_FILE="$DATA_DIR/validation.jsonl"
TEST_FILE="$DATA_DIR/test.jsonl"
MANIFEST_FILE="$DATA_DIR/manifest.json"


# ============================================================
# FROZEN CORPUS HASHES
# ============================================================


EXPECTED_TRAIN_SHA256="ee80a1efc74807d354761d9728cd498e9f543c84a5e568e4f3821d2fa48565f9"

EXPECTED_VALIDATION_SHA256="5d8ea877e14105e06068f2d7de5ae55d265b4069af2c1b0762b162df1be1347a"

EXPECTED_TEST_SHA256="03b7289a5841f79e1c95188f22615dca6789846aeddecc1b3c90f9713c918834"


# ============================================================
# HELPERS
# ============================================================


separator() {
    printf '\n'
    printf '%0.s=' {1..80}
    printf '\n'
}


fail() {
    echo
    echo "ERROR: $1"
    echo
    exit 1
}


run_module() {

    local module="$1"

    shift

    echo
    echo "Running:"
    echo "  $PYTHON -m $module $*"
    echo

    "$PYTHON" -m "$module" "$@"
}


sha256_file() {

    "$PYTHON" - "$1" <<'PY'
import hashlib
import sys

path = sys.argv[1]

digest = hashlib.sha256()

with open(path, "rb") as handle:
    while True:
        chunk = handle.read(1024 * 1024)

        if not chunk:
            break

        digest.update(chunk)

print(digest.hexdigest())
PY
}


verify_hash() {

    local label="$1"
    local path="$2"
    local expected="$3"

    if [[ ! -f "$path" ]]; then
        fail "$label file not found: $path"
    fi

    local actual

    actual="$(sha256_file "$path")"

    printf '%-16s %s\n' "$label:" "$actual"

    if [[ "$actual" != "$expected" ]]; then

        echo
        echo "Expected:"
        echo "  $expected"
        echo
        echo "Actual:"
        echo "  $actual"

        fail "$label SHA-256 mismatch. Frozen corpus has changed."
    fi
}


# ============================================================
# HEADER
# ============================================================


separator

echo "MAIA SFT VALIDATION"
echo "STAGE 4 - SAFE PREFLIGHT V4"

separator

echo
echo "Project:"
echo "  $ROOT"

echo
echo "Python:"
echo "  $PYTHON"

echo
echo "Corpus:"
echo "  $DATA_DIR"


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================


separator

echo "PYTHON ENVIRONMENT"

separator


if [[ ! -x "$PYTHON" ]]; then
    fail "Virtual environment Python not found: $PYTHON"
fi


"$PYTHON" - <<'PY'
import platform
import sys

print("Python:     ", sys.version.split()[0])
print("Platform:   ", platform.platform())

try:
    import torch

    print("PyTorch:    ", torch.__version__)
    print("MPS built:  ", torch.backends.mps.is_built())
    print("MPS avail.: ", torch.backends.mps.is_available())

except Exception as exc:
    print("PyTorch ERROR:", exc)
    raise


try:
    import transformers

    print("Transformers:", transformers.__version__)

except Exception as exc:
    print("Transformers ERROR:", exc)
    raise


try:
    import datasets

    print("Datasets:   ", datasets.__version__)

except Exception as exc:
    print("Datasets ERROR:", exc)
    raise
PY


# ============================================================
# FROZEN CORPUS VERIFICATION
# ============================================================


separator

echo "FROZEN CORPUS VERIFICATION"

separator


verify_hash \
    "TRAIN" \
    "$TRAIN_FILE" \
    "$EXPECTED_TRAIN_SHA256"


verify_hash \
    "VALIDATION" \
    "$VALIDATION_FILE" \
    "$EXPECTED_VALIDATION_SHA256"


verify_hash \
    "TEST" \
    "$TEST_FILE" \
    "$EXPECTED_TEST_SHA256"


echo
echo "Frozen corpus hashes: PASS"


# ============================================================
# MANIFEST VERIFICATION
# ============================================================


separator

echo "MANIFEST VERIFICATION"

separator


if [[ ! -f "$MANIFEST_FILE" ]]; then
    fail "Manifest not found: $MANIFEST_FILE"
fi


"$PYTHON" - "$MANIFEST_FILE" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

expected_total = 18000

expected_languages = {
    "en": 7200,
    "pt": 5400,
    "es": 5400,
}

print("Stage:")
print(" ", manifest.get("stage"))

print()
print("Total:")
print(" ", manifest.get("total"))

print()
print("Target languages:")
print(" ", manifest.get("target_languages"))

if manifest.get("total") != expected_total:
    raise RuntimeError(
        "Manifest total is not 18,000."
    )

if manifest.get("target_languages") != expected_languages:
    raise RuntimeError(
        "Manifest language targets changed."
    )

audit = manifest.get("audit", {})

if audit.get("prompt_leakage") != "PASS":
    raise RuntimeError(
        "Manifest does not record prompt leakage PASS."
    )

if audit.get(
    "cross_source_exact_duplicates"
) != 0:
    raise RuntimeError(
        "Manifest reports cross-source duplicates."
    )

print()
print("Manifest audit: PASS")
PY


# ============================================================
# INDEPENDENT CORPUS AUDIT
# ============================================================


separator

echo "INDEPENDENT CORPUS AUDIT"

separator


"$PYTHON" - \
    "$TRAIN_FILE" \
    "$VALIDATION_FILE" \
    "$TEST_FILE" <<'PY'

import json
import re
import sys
import unicodedata

from collections import Counter


train_path = sys.argv[1]
validation_path = sys.argv[2]
test_path = sys.argv[3]


def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except Exception as exc:

                raise RuntimeError(
                    f"{path}:{line_number}: "
                    f"invalid JSON: {exc}"
                )

            rows.append(row)

    return rows


def normalize(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def prompt_set(rows):

    prompts = set()

    for row in rows:

        for message in row["messages"]:

            if message.get("role") != "user":
                continue

            text = normalize(
                message.get(
                    "content",
                    "",
                )
            )

            if text:
                prompts.add(text)

    return prompts


datasets = {
    "train":
        load_jsonl(train_path),

    "validation":
        load_jsonl(validation_path),

    "test":
        load_jsonl(test_path),
}


total = sum(
    len(rows)
    for rows in datasets.values()
)


print(
    f"Total conversations: {total:,}"
)


if total != 18000:

    raise RuntimeError(
        f"Expected 18,000 conversations, "
        f"found {total:,}."
    )


global_languages = Counter()


for split, rows in datasets.items():

    languages = Counter(
        row["language"]
        for row in rows
    )

    global_languages.update(
        languages
    )

    print()
    print(split.upper())

    print(
        f"  conversations: "
        f"{len(rows):,}"
    )

    print(
        "  languages:",
        dict(
            sorted(
                languages.items()
            )
        ),
    )

    for row_number, row in enumerate(
        rows,
        start=1,
    ):

        required = {
            "id",
            "source",
            "subsource",
            "language",
            "category",
            "messages",
        }

        missing = (
            required
            - set(row)
        )

        if missing:

            raise RuntimeError(
                f"{split} row "
                f"{row_number}: "
                f"missing fields "
                f"{sorted(missing)}"
            )

        messages = row["messages"]

        if not messages:

            raise RuntimeError(
                f"{split} row "
                f"{row_number}: "
                f"empty messages"
            )

        if (
            messages[0].get("role")
            != "system"
        ):

            raise RuntimeError(
                f"{split} row "
                f"{row_number}: "
                f"missing controlled "
                f"system message"
            )

        conversational = [
            message
            for message in messages
            if message.get("role")
            in (
                "user",
                "assistant",
            )
        ]

        if (
            not conversational
            or conversational[0]["role"]
            != "user"
            or conversational[-1]["role"]
            != "assistant"
        ):

            raise RuntimeError(
                f"{split} row "
                f"{row_number}: "
                f"invalid conversation "
                f"boundaries"
            )

        expected = "user"

        for message in conversational:

            if message["role"] != expected:

                raise RuntimeError(
                    f"{split} row "
                    f"{row_number}: "
                    f"invalid role order"
                )

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:

                raise RuntimeError(
                    f"{split} row "
                    f"{row_number}: "
                    f"empty message"
                )

            expected = (
                "assistant"
                if expected == "user"
                else "user"
            )


expected_languages = Counter(
    {
        "en": 7200,
        "pt": 5400,
        "es": 5400,
    }
)


print()
print(
    "Global languages:",
    dict(
        sorted(
            global_languages.items()
        )
    ),
)


if global_languages != expected_languages:

    raise RuntimeError(
        "Global language composition "
        "does not match frozen V4."
    )


prompts = {
    split:
        prompt_set(rows)
    for split, rows
    in datasets.items()
}


intersections = {
    "train-validation":
        prompts["train"]
        & prompts["validation"],

    "train-test":
        prompts["train"]
        & prompts["test"],

    "validation-test":
        prompts["validation"]
        & prompts["test"],
}


print()
print("Prompt intersections:")


for name, values in intersections.items():

    print(
        f"  {name}: "
        f"{len(values)}"
    )

    if values:

        raise RuntimeError(
            f"Prompt leakage: {name}"
        )


print()
print(
    "Independent corpus audit: PASS"
)
PY


# ============================================================
# BUILD FROZEN BENCHMARK
# ============================================================


separator

echo "BUILDING FROZEN BENCHMARK"

separator


if [[ -f "$ROOT/src/build_benchmark_stage4.py" ]]; then

    run_module \
        src.build_benchmark_stage4

else

    fail \
        "src/build_benchmark_stage4.py not found."
fi


# ============================================================
# DATASET / MASKING AUDIT
# ============================================================


separator

echo "RESPONSE-ONLY DATASET AUDIT"

separator


if [[ -f "$ROOT/src/audit_stage4.py" ]]; then

    run_module \
        src.audit_stage4

else

    fail \
        "src/audit_stage4.py not found."
fi


# ============================================================
# TESTS
# ============================================================


separator

echo "AUTOMATED TESTS"

separator


cd "$ROOT"


"$PYTHON" -m pytest \
    -q \
    tests/test_stage4.py


# ============================================================
# FINAL STATUS
# ============================================================


separator

echo "STAGE 4 V4 PREFLIGHT: PASS"

separator


echo
echo "Frozen corpus:"
echo
echo "  TRAIN"
echo "  $EXPECTED_TRAIN_SHA256"
echo
echo "  VALIDATION"
echo "  $EXPECTED_VALIDATION_SHA256"
echo
echo "  TEST"
echo "  $EXPECTED_TEST_SHA256"

echo
echo "Corpus:"
echo "  18,000 conversations"
echo "  EN 7,200"
echo "  PT 5,400"
echo "  ES 5,400"

echo
echo "TRAINING HAS NOT STARTED."

echo
echo "Next phase:"
echo "  1. Review benchmark composition"
echo "  2. Review response-only masking"
echo "  3. Freeze benchmark SHA-256"
echo "  4. Run GPT-2 Medium baseline"
echo "  5. Start resumable Stage 4 training"

echo