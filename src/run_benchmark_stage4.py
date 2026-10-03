from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common_stage4 import (
    DATA,
    OUT,
    ROOT,
    device,
    read_jsonl,
)


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 V4 - DEFINITIVE BENCHMARK RUNNER
#
# PURPOSE
#
# Run exactly the same frozen benchmark against:
#
#   1. original GPT-2 Medium baseline
#   2. final Stage 4 SFT model
#
# IMPORTANT
#
# The benchmark itself is immutable.
#
# Numeric evaluation is intentionally STRICT:
#
#   "36"              -> valid numeric response
#   " 36 "            -> valid numeric response
#   "36.0"            -> valid numeric response
#   "The answer is 36"-> invalid
#   "36 Assistant..." -> invalid
#
# This prevents prompt echo / continuation from being
# counted as a correct instruction-following response.
# ============================================================


BENCHMARK_FILE = (
    DATA
    / "benchmark"
    / "stage4-v4-benchmark.jsonl"
)

EXPECTED_BENCHMARK_SHA256 = (
    "9ba60ac68ad285fdb0e16eb2af876722d8745e8131203b4a090b51b6a5ad73c3"
)

DEFAULT_BASELINE_MODEL = (
    "openai-community/gpt2-medium"
)

MAX_CONTEXT_LENGTH = 1024

# FROZEN generation parameter.
# Must remain identical for baseline and post-SFT runs.
MAX_NEW_TOKENS = 192

# Evaluator version is recorded in the run manifest so that
# baseline and post-SFT results can be verified as comparable.
EVALUATOR_VERSION = "stage4-v4-strict-numeric-v2"


# ============================================================
# GENERAL HELPERS
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.strip().casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        text,
    )

    return "".join(
        char
        for char in decomposed
        if unicodedata.category(char) != "Mn"
    )


def normalized_answer(text: str) -> str:
    text = normalize_text(text)

    text = strip_accents(text)

    text = text.strip()

    # Used only for normalized-exact textual answers.
    text = text.strip(
        " \t\r\n.,;:!?\"'`"
    )

    return text


# ============================================================
# STRICT NUMERIC PARSING
#
# IMPORTANT:
#
# re.fullmatch() is deliberate.
#
# The complete assistant response must be numeric.
# We do NOT search for a number somewhere inside a longer
# generated continuation.
#
# Accepted examples:
#
#   36
#   -12
#   +7
#   36.0
#   36,0
#   1e3
#   -2.5e-2
#
# Rejected examples:
#
#   The answer is 36
#   36 Assistant: ...
#   36 31 109
# ============================================================


NUMERIC_PATTERN = re.compile(
    r"""
    [-+]?
    (?:
        (?:\d+(?:[.,]\d*)?)
        |
        (?:[.,]\d+)
    )
    (?:[eE][-+]?\d+)?
    """,
    re.VERBOSE,
)


def parse_strict_number(text: str):
    candidate = unicodedata.normalize(
        "NFKC",
        text,
    ).strip()

    match = NUMERIC_PATTERN.fullmatch(
        candidate
    )

    if match is None:
        return None

    normalized = (
        candidate
        .replace(",", ".")
    )

    try:
        return float(normalized)

    except ValueError:
        return None


# ============================================================
# PROMPT RENDERING
#
# This matches the Stage 4 conversation convention:
#
#   User: ...
#   Assistant: ...
#
# Historical assistant turns are preserved in multi-turn
# benchmark cases.
# ============================================================


def render_benchmark_prompt(
    messages,
) -> str:
    parts = []

    for message in messages:
        role = message["role"]

        content = message[
            "content"
        ].strip()

        if role == "system":
            parts.append(
                f"System: {content}\n"
            )

        elif role == "user":
            parts.append(
                f"User: {content}\n"
            )

        elif role == "assistant":
            parts.append(
                f"Assistant: {content}\n"
            )

        else:
            raise RuntimeError(
                f"Unsupported role: {role}"
            )

    parts.append(
        "Assistant: "
    )

    return "".join(parts)


# ============================================================
# OBJECTIVE EVALUATION
# ============================================================


def evaluate_normalized_exact(
    generation,
    expected,
    aliases,
):
    prediction = normalized_answer(
        generation
    )

    accepted = {
        normalized_answer(expected)
    }

    accepted.update(
        normalized_answer(alias)
        for alias in aliases
    )

    return (
        prediction in accepted,
        prediction,
        sorted(accepted),
    )


def evaluate_numeric(
    generation,
    expected,
    aliases,
):
    # --------------------------------------------------------
    # CRITICAL FIX:
    #
    # Previous evaluator:
    #
    #   re.search(...)
    #
    # accepted a correct number appearing anywhere inside a
    # long continuation.
    #
    # Current evaluator:
    #
    #   parse_strict_number(...)
    #
    # requires the COMPLETE generated answer to be numeric.
    # --------------------------------------------------------

    predicted_number = parse_strict_number(
        generation
    )

    expected_number = parse_strict_number(
        str(expected)
    )

    if expected_number is None:
        raise RuntimeError(
            "Benchmark contains an invalid "
            f"numeric expected answer: {expected!r}"
        )

    if predicted_number is None:
        return (
            False,
            None,
            expected_number,
        )

    correct = math.isclose(
        predicted_number,
        expected_number,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    return (
        correct,
        predicted_number,
        expected_number,
    )


def evaluate_objective_case(
    case,
    generation,
):
    expected_block = case[
        "expected"
    ]

    expected = expected_block[
        "answer"
    ]

    aliases = expected_block.get(
        "aliases",
        [],
    )

    evaluator = expected_block.get(
        "evaluator",
        "normalized_exact",
    )

    if evaluator == "numeric":
        (
            correct,
            prediction_value,
            reference_value,
        ) = evaluate_numeric(
            generation,
            expected,
            aliases,
        )

    elif evaluator == "normalized_exact":
        (
            correct,
            prediction_value,
            reference_value,
        ) = evaluate_normalized_exact(
            generation,
            expected,
            aliases,
        )

    else:
        raise RuntimeError(
            "Unsupported objective "
            f"evaluator: {evaluator}"
        )

    return {
        "correct":
            bool(correct),

        "evaluator":
            evaluator,

        "expected":
            expected,

        "aliases":
            aliases,

        "prediction_value":
            prediction_value,

        "reference_value":
            reference_value,
    }


# ============================================================
# MODEL LOADING
# ============================================================


def load_model_and_tokenizer(
    model_source,
    run_device,
):
    print()
    print("Loading tokenizer:")
    print(" ", model_source)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_source
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print()
    print("Loading model:")
    print(" ", model_source)

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_source
        )
    )

    model.to(
        run_device
    )

    model.eval()

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = True

    print()
    print(
        "Model parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    print(
        "Tokenizer vocab:",
        len(tokenizer),
    )

    print(
        "EOS:",
        tokenizer.eos_token_id,
    )

    print(
        "PAD:",
        tokenizer.pad_token_id,
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# GENERATION
# ============================================================


def generate_response(
    model,
    tokenizer,
    prompt,
    run_device,
):
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoded[
        "input_ids"
    ][0]

    original_prompt_tokens = int(
        input_ids.shape[0]
    )

    # Reserve exactly MAX_NEW_TOKENS positions for the
    # assistant response.
    max_prompt_tokens = (
        MAX_CONTEXT_LENGTH
        - MAX_NEW_TOKENS
    )

    prompt_truncated = False
    removed_prompt_tokens = 0

    if (
        input_ids.shape[0]
        > max_prompt_tokens
    ):
        removed_prompt_tokens = (
            int(input_ids.shape[0])
            - max_prompt_tokens
        )

        input_ids = input_ids[
            -max_prompt_tokens:
        ]

        prompt_truncated = True

    input_ids = (
        input_ids
        .unsqueeze(0)
        .to(run_device)
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    prompt_tokens = int(
        input_ids.shape[1]
    )

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=(
                tokenizer.pad_token_id
            ),

            eos_token_id=(
                tokenizer.eos_token_id
            ),

            use_cache=True,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    generated_ids = output[
        0,
        prompt_tokens:
    ]

    generation = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    generated_tokens = int(
        generated_ids.shape[0]
    )

    eos_generated = False

    if generated_tokens > 0:
        eos_generated = bool(
            generated_ids[-1].item()
            == tokenizer.eos_token_id
        )

    return {
        "text":
            generation.strip(),

        "original_prompt_tokens":
            original_prompt_tokens,

        "prompt_tokens":
            prompt_tokens,

        "prompt_truncated":
            prompt_truncated,

        "removed_prompt_tokens":
            removed_prompt_tokens,

        "generated_tokens":
            generated_tokens,

        "eos_generated":
            eos_generated,

        "generation_seconds":
            elapsed,

        "tokens_per_second":
            (
                generated_tokens / elapsed
                if elapsed > 0
                else None
            ),
    }


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    results,
):
    objective = [
        result
        for result in results
        if result["kind"] == "objective"
    ]

    qualitative = [
        result
        for result in results
        if result["kind"] == "qualitative"
    ]

    correct_total = sum(
        1
        for result in objective
        if result[
            "objective_evaluation"
        ][
            "correct"
        ]
    )

    overall_accuracy = (
        correct_total / len(objective)
        if objective
        else 0.0
    )

    by_language = {}

    for language in (
        "en",
        "pt",
        "es",
    ):
        subset = [
            result
            for result in objective
            if result["language"] == language
        ]

        correct = sum(
            1
            for result in subset
            if result[
                "objective_evaluation"
            ][
                "correct"
            ]
        )

        by_language[
            language
        ] = {
            "total":
                len(subset),

            "correct":
                correct,

            "accuracy":
                (
                    correct / len(subset)
                    if subset
                    else 0.0
                ),
        }

    category_groups = defaultdict(
        list
    )

    for result in objective:
        category_groups[
            result["category"]
        ].append(
            result
        )

    by_category = {}

    for (
        category,
        subset,
    ) in sorted(
        category_groups.items()
    ):
        correct = sum(
            1
            for result in subset
            if result[
                "objective_evaluation"
            ][
                "correct"
            ]
        )

        by_category[
            category
        ] = {
            "total":
                len(subset),

            "correct":
                correct,

            "accuracy":
                correct / len(subset),
        }

    language_category = {}

    for language in (
        "en",
        "pt",
        "es",
    ):
        for category in sorted(
            category_groups
        ):
            subset = [
                result
                for result in objective
                if (
                    result["language"]
                    == language
                    and result[
                        "category"
                    ]
                    == category
                )
            ]

            if not subset:
                continue

            correct = sum(
                1
                for result in subset
                if result[
                    "objective_evaluation"
                ][
                    "correct"
                ]
            )

            key = (
                f"{language}:"
                f"{category}"
            )

            language_category[
                key
            ] = {
                "total":
                    len(subset),

                "correct":
                    correct,

                "accuracy":
                    correct / len(subset),
            }

    prompt_truncated = sum(
        1
        for result in results
        if result[
            "generation"
        ][
            "prompt_truncated"
        ]
    )

    eos_generated = sum(
        1
        for result in results
        if result[
            "generation"
        ][
            "eos_generated"
        ]
    )

    generated_tokens = [
        result[
            "generation"
        ][
            "generated_tokens"
        ]
        for result in results
    ]

    generation_seconds = [
        result[
            "generation"
        ][
            "generation_seconds"
        ]
        for result in results
    ]

    return {
        "examples":
            len(results),

        "objective": {
            "total":
                len(objective),

            "correct":
                correct_total,

            "accuracy":
                overall_accuracy,

            "by_language":
                by_language,

            "by_category":
                by_category,

            "by_language_category":
                language_category,
        },

        "qualitative": {
            "total":
                len(qualitative),
        },

        "generation": {
            "max_new_tokens":
                MAX_NEW_TOKENS,

            "do_sample":
                False,

            "decoding":
                "greedy",

            "context_length":
                MAX_CONTEXT_LENGTH,

            "prompt_truncated_count":
                prompt_truncated,

            "eos_generated_count":
                eos_generated,

            "mean_generated_tokens":
                (
                    sum(generated_tokens)
                    / len(generated_tokens)
                    if generated_tokens
                    else 0.0
                ),

            "total_generation_seconds":
                sum(generation_seconds),
        },
    }


# ============================================================
# REPORT
# ============================================================


def print_metrics(
    metrics,
):
    print()
    print("=" * 80)
    print(
        "OBJECTIVE BENCHMARK RESULTS"
    )
    print("=" * 80)

    objective = metrics[
        "objective"
    ]

    print()
    print(
        "Correct:",
        f"{objective['correct']}"
        f"/{objective['total']}",
    )

    print(
        "Accuracy:",
        f"{100.0 * objective['accuracy']:.2f}%",
    )

    print()
    print(
        "BY LANGUAGE"
    )

    for language in (
        "en",
        "pt",
        "es",
    ):
        row = (
            objective[
                "by_language"
            ][
                language
            ]
        )

        print(
            f"  {language.upper()}: "
            f"{row['correct']:3d}/"
            f"{row['total']:3d} "
            f"({100.0 * row['accuracy']:.2f}%)"
        )

    print()
    print(
        "BY CATEGORY"
    )

    for (
        category,
        row,
    ) in objective[
        "by_category"
    ].items():
        print(
            f"  {category:16s} "
            f"{row['correct']:3d}/"
            f"{row['total']:3d} "
            f"({100.0 * row['accuracy']:.2f}%)"
        )

    print()
    print(
        "Qualitative responses saved:",
        metrics[
            "qualitative"
        ][
            "total"
        ],
    )

    print(
        "Prompt truncations:",
        metrics[
            "generation"
        ][
            "prompt_truncated_count"
        ],
    )

    print(
        "EOS generations:",
        metrics[
            "generation"
        ][
            "eos_generated_count"
        ],
        "/",
        metrics[
            "examples"
        ],
    )

    print(
        "Mean generated tokens:",
        f"{metrics['generation']['mean_generated_tokens']:.2f}",
    )


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Maia Stage 4 V4 "
            "assistant benchmark."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_BASELINE_MODEL,
        help=(
            "Hugging Face model ID or local "
            "Stage 4 model directory."
        ),
    )

    parser.add_argument(
        "--run-name",
        default="gpt2-medium-baseline",
        help=(
            "Name used for output files."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    print()
    print("=" * 80)
    print(
        "MAIA SFT VALIDATION"
    )
    print(
        "STAGE 4 V4 - "
        "DEFINITIVE BENCHMARK RUNNER"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Verify frozen benchmark
    # --------------------------------------------------------

    if not BENCHMARK_FILE.exists():
        raise RuntimeError(
            "Frozen benchmark file "
            f"not found: {BENCHMARK_FILE}"
        )

    actual_sha256 = sha256_file(
        BENCHMARK_FILE
    )

    print()
    print("Benchmark:")
    print(" ", BENCHMARK_FILE)

    print()
    print("Expected SHA-256:")
    print(
        " ",
        EXPECTED_BENCHMARK_SHA256,
    )

    print("Actual SHA-256:")
    print(
        " ",
        actual_sha256,
    )

    if (
        actual_sha256
        != EXPECTED_BENCHMARK_SHA256
    ):
        raise RuntimeError(
            "Frozen benchmark SHA-256 "
            "mismatch."
        )

    print()
    print(
        "Frozen benchmark SHA-256: PASS"
    )

    # --------------------------------------------------------
    # Read benchmark
    # --------------------------------------------------------

    cases = read_jsonl(
        BENCHMARK_FILE
    )

    if len(cases) != 300:
        raise RuntimeError(
            f"Expected 300 benchmark cases, "
            f"got {len(cases)}"
        )

    objective_count = sum(
        1
        for case in cases
        if case["kind"] == "objective"
    )

    qualitative_count = sum(
        1
        for case in cases
        if case["kind"] == "qualitative"
    )

    if (
        objective_count != 180
        or qualitative_count != 120
    ):
        raise RuntimeError(
            "Frozen benchmark composition "
            "is invalid."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    run_device = device()

    print()
    print(
        "Device:",
        run_device,
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Run name:",
        args.run_name,
    )

    print(
        "Evaluator:",
        EVALUATOR_VERSION,
    )

    print(
        "Decoding: greedy"
    )

    print(
        "Max new tokens:",
        MAX_NEW_TOKENS,
    )

    print(
        "Context length:",
        MAX_CONTEXT_LENGTH,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        tokenizer,
    ) = load_model_and_tokenizer(
        args.model,
        run_device,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        OUT
        / "stage4-benchmark"
        / args.run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generations_path = (
        output_dir
        / "generations.jsonl"
    )

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    run_manifest_path = (
        output_dir
        / "run-manifest.json"
    )

    # --------------------------------------------------------
    # SAFETY:
    #
    # Never append to an old benchmark run.
    #
    # This fixes another weakness in the first runner.
    # --------------------------------------------------------

    existing = [
        path
        for path in (
            generations_path,
            metrics_path,
            run_manifest_path,
        )
        if path.exists()
    ]

    if existing:
        print()
        print("=" * 80)
        print(
            "ERROR: OUTPUT RUN ALREADY EXISTS"
        )
        print("=" * 80)

        print()
        print(
            "Refusing to append to or overwrite "
            "an existing benchmark run."
        )

        print()
        print(
            "Existing files:"
        )

        for path in existing:
            print(
                " ",
                path,
            )

        print()
        print(
            "Use a new --run-name."
        )

        raise RuntimeError(
            "Benchmark output already exists."
        )

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print(
        "RUNNING 300 BENCHMARK CASES"
    )
    print("=" * 80)
    print()

    start_total = (
        time.perf_counter()
    )

    for case in tqdm(
        cases,
        total=len(cases),
        desc="Stage 4 benchmark",
    ):
        prompt = (
            render_benchmark_prompt(
                case["messages"]
            )
        )

        generation = (
            generate_response(
                model,
                tokenizer,
                prompt,
                run_device,
            )
        )

        result = {
            "id":
                case["id"],

            "language":
                case["language"],

            "kind":
                case["kind"],

            "category":
                case["category"],

            "messages":
                case["messages"],

            "rendered_prompt":
                prompt,

            "generation":
                generation,
        }

        if (
            case["kind"]
            == "objective"
        ):
            result[
                "expected"
            ] = case[
                "expected"
            ]

            result[
                "objective_evaluation"
            ] = (
                evaluate_objective_case(
                    case,
                    generation["text"],
                )
            )

        else:
            result[
                "criteria"
            ] = case[
                "criteria"
            ]

        results.append(
            result
        )

        # Persist every completed case.
        with generations_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_seconds = (
        time.perf_counter()
        - start_total
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    metrics[
        "evaluator_version"
    ] = EVALUATOR_VERSION

    metrics[
        "benchmark_sha256"
    ] = actual_sha256

    metrics[
        "generation"
    ][
        "wall_clock_seconds"
    ] = total_seconds

    metrics[
        "generation"
    ][
        "wall_clock_minutes"
    ] = (
        total_seconds / 60.0
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Run manifest
    # --------------------------------------------------------

    run_manifest = {
        "stage":
            "stage4-v4-benchmark-run",

        "run_name":
            args.run_name,

        "model_source":
            args.model,

        "evaluator_version":
            EVALUATOR_VERSION,

        "benchmark_file":
            str(
                BENCHMARK_FILE
                .relative_to(ROOT)
            ),

        "benchmark_sha256":
            actual_sha256,

        "benchmark_examples":
            len(cases),

        "objective_examples":
            objective_count,

        "qualitative_examples":
            qualitative_count,

        "device":
            str(run_device),

        "generation": {
            "decoding":
                "greedy",

            "do_sample":
                False,

            "max_new_tokens":
                MAX_NEW_TOKENS,

            "context_length":
                MAX_CONTEXT_LENGTH,
        },

        "artifacts": {
            "generations":
                str(
                    generations_path
                    .relative_to(ROOT)
                ),

            "metrics":
                str(
                    metrics_path
                    .relative_to(ROOT)
                ),
        },
    }

    run_manifest_path.write_text(
        json.dumps(
            run_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_metrics(
        metrics
    )

    print()
    print("=" * 80)
    print(
        "BENCHMARK ARTIFACTS"
    )
    print("=" * 80)

    print()
    print("Generations:")
    print(
        " ",
        generations_path,
    )

    print()
    print("Metrics:")
    print(
        " ",
        metrics_path,
    )

    print()
    print("Run manifest:")
    print(
        " ",
        run_manifest_path,
    )

    print()
    print(
        "Wall clock:",
        f"{total_seconds / 60.0:.2f} min",
    )

    print()
    print("=" * 80)
    print(
        "STAGE 4 V4 BENCHMARK RUN: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()