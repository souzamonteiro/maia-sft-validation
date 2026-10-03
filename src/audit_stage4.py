from __future__ import annotations

import json

from transformers import AutoTokenizer

from src.common_stage4 import *
from src.dataset_stage4 import ResponseOnlyDataset


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 - FULL RESPONSE-ONLY DATASET AUDIT
#
# FINAL POLICY:
#
#   - max_length is the model context limit (1024 for GPT-2)
#
#   - assistant responses that fit inside max_length are
#     preserved completely
#
#   - when prefix + answer exceeds max_length, old context
#     may be truncated from the LEFT
#
#   - assistant responses that individually exceed max_length
#     are REJECTED, never silently truncated
#
#   - rejected long answers are expected dataset filtering,
#     not an audit failure
#
# HARD FAILURES:
#
#   - conversation ID leakage
#   - prompt leakage
#   - zero-supervision examples
#   - sequence > max_length after materialization
#   - malformed response-only labels
#   - missing supervised EOS
# ============================================================


def audit_dataset(
    split,
    rows,
    tokenizer,
):
    print()
    print("=" * 80)
    print(
        f"{split.upper()} RESPONSE-ONLY AUDIT"
    )
    print("=" * 80)

    dataset = ResponseOnlyDataset(
        rows,
        tokenizer,
        CFG["max_length"],
        reject_answer_truncation=True,
    )

    audit = dataset.audit.as_dict()

    assistant_segments = audit[
        "assistant_segments"
    ]

    accepted_segments = audit[
        "accepted_segments"
    ]

    rejected_segments = audit[
        "answer_truncated"
    ]

    accepted_percent = (
        100.0
        * accepted_segments
        / max(1, assistant_segments)
    )

    rejected_percent = (
        100.0
        * rejected_segments
        / max(1, assistant_segments)
    )

    context_percent = (
        100.0
        * audit["context_truncated"]
        / max(1, assistant_segments)
    )

    print()
    print(
        "Conversations:        ",
        f"{audit['conversations']:,}",
    )

    print(
        "Assistant segments:   ",
        f"{assistant_segments:,}",
    )

    print(
        "Accepted segments:    ",
        f"{accepted_segments:,}",
        f"({accepted_percent:.2f}%)",
    )

    print(
        "Rejected long answers:",
        f"{rejected_segments:,}",
        f"({rejected_percent:.2f}%)",
    )

    print(
        "Context truncated:    ",
        f"{audit['context_truncated']:,}",
        f"({context_percent:.2f}%)",
    )

    print(
        "Zero supervision:     ",
        f"{audit['zero_supervision']:,}",
    )

    print(
        "Removed context tok:  ",
        f"{audit['removed_context_tokens']:,}",
    )

    print(
        "Rejected answer tok:  ",
        f"{audit['removed_answer_tokens']:,}",
    )

    print(
        "Max original length:  ",
        f"{audit['max_original_length']:,}",
    )

    print(
        "Max final length:     ",
        f"{audit['max_final_length']:,}",
    )

    print(
        "Min supervised tok:   ",
        audit["min_supervised_tokens"],
    )

    print(
        "Max supervised tok:   ",
        audit["max_supervised_tokens"],
    )

    # ========================================================
    # HARD INVARIANTS
    # ========================================================

    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise RuntimeError(
            "Tokenizer has no EOS token."
        )

    if not dataset.items:
        raise RuntimeError(
            f"{split}: no usable dataset items."
        )

    supervised_counts = []

    for index, item in enumerate(
        dataset.items
    ):
        input_ids = item[
            "input_ids"
        ]

        labels = item[
            "labels"
        ]

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"{split} item {index}: "
                "input_ids/labels length mismatch."
            )

        # ----------------------------------------------------
        # Model context limit
        # ----------------------------------------------------

        if len(input_ids) > CFG["max_length"]:
            raise RuntimeError(
                f"{split} item {index}: "
                f"sequence length "
                f"{len(input_ids)} exceeds "
                f"max_length "
                f"{CFG['max_length']}."
            )

        # ----------------------------------------------------
        # Supervised positions
        # ----------------------------------------------------

        supervised_positions = [
            position
            for position, value
            in enumerate(labels)
            if value != -100
        ]

        if not supervised_positions:
            raise RuntimeError(
                f"{split} item {index}: "
                "zero supervised tokens."
            )

        first_supervised = (
            supervised_positions[0]
        )

        # ----------------------------------------------------
        # Prefix must be fully masked.
        # ----------------------------------------------------

        if any(
            value != -100
            for value
            in labels[
                :first_supervised
            ]
        ):
            raise RuntimeError(
                f"{split} item {index}: "
                "prefix supervision detected."
            )

        # ----------------------------------------------------
        # Once assistant supervision starts, it must remain
        # contiguous through EOS.
        # ----------------------------------------------------

        if any(
            value == -100
            for value
            in labels[
                first_supervised:
            ]
        ):
            raise RuntimeError(
                f"{split} item {index}: "
                "non-contiguous assistant "
                "supervision."
            )

        # ----------------------------------------------------
        # Supervised labels must exactly equal input tokens.
        # ----------------------------------------------------

        if (
            labels[first_supervised:]
            != input_ids[first_supervised:]
        ):
            raise RuntimeError(
                f"{split} item {index}: "
                "assistant labels do not match "
                "input tokens."
            )

        # ----------------------------------------------------
        # EOS must terminate the sequence.
        # ----------------------------------------------------

        if input_ids[-1] != eos_id:
            raise RuntimeError(
                f"{split} item {index}: "
                "terminal EOS missing."
            )

        # ----------------------------------------------------
        # EOS itself must be supervised.
        # ----------------------------------------------------

        if labels[-1] != eos_id:
            raise RuntimeError(
                f"{split} item {index}: "
                "EOS is not supervised."
            )

        supervised_counts.append(
            len(supervised_positions)
        )

    # ========================================================
    # DATASET-LEVEL HARD INVARIANTS
    # ========================================================

    if audit["zero_supervision"] != 0:
        raise RuntimeError(
            f"{split}: "
            f"{audit['zero_supervision']} "
            "zero-supervision segments detected."
        )

    if (
        audit["accepted_segments"]
        + audit["answer_truncated"]
        + audit["zero_supervision"]
        != audit["assistant_segments"]
    ):
        raise RuntimeError(
            f"{split}: assistant segment "
            "accounting mismatch."
        )

    if audit["max_final_length"] > CFG["max_length"]:
        raise RuntimeError(
            f"{split}: final materialized "
            "sequence exceeds model context."
        )

    # ========================================================
    # LONG ANSWER REPORT
    #
    # This is INFORMATIONAL.
    # Long answers are intentionally rejected by the dataset.
    # ========================================================

    if dataset.rejected:
        print()
        print(
            "Long-answer rejection policy: ACTIVE"
        )

        print(
            "  These assistant responses are NOT "
            "included in SFT."
        )

        print(
            "  They are NOT truncated."
        )

        print(
            "  Corpus JSONL files remain unchanged."
        )

        print()
        print(
            "First rejected examples:"
        )

        for record in dataset.rejected[:10]:
            print(
                " ",
                json.dumps(
                    record,
                    ensure_ascii=False,
                ),
            )

        if len(dataset.rejected) > 10:
            print(
                f"  ... plus "
                f"{len(dataset.rejected) - 10:,} "
                "additional rejected segments."
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    supervised_mean = (
        sum(supervised_counts)
        / len(supervised_counts)
    )

    print()
    print(
        "Supervised tokens/item:"
    )

    print(
        "  minimum:",
        min(supervised_counts),
    )

    print(
        "  maximum:",
        max(supervised_counts),
    )

    print(
        "  mean:   ",
        f"{supervised_mean:.2f}",
    )

    print()
    print(
        f"{split.upper()} response-only audit: PASS"
    )

    return {
        "split":
            split,

        "corpus_conversations":
            len(rows),

        "assistant_segments":
            assistant_segments,

        "accepted_segments":
            accepted_segments,

        "accepted_percent":
            accepted_percent,

        "rejected_long_answers":
            rejected_segments,

        "rejected_percent":
            rejected_percent,

        "context_truncated":
            audit[
                "context_truncated"
            ],

        "context_truncated_percent":
            context_percent,

        "zero_supervision":
            audit[
                "zero_supervision"
            ],

        "audit":
            audit,

        "supervised_tokens": {
            "minimum":
                min(supervised_counts),

            "maximum":
                max(supervised_counts),

            "mean":
                supervised_mean,
        },
    }


def main():
    print()
    print("=" * 80)
    print(
        "MAIA SFT VALIDATION - STAGE 4"
    )
    print(
        "FULL RESPONSE-ONLY AUDIT"
    )
    print("=" * 80)

    # ========================================================
    # LOAD ALL FROZEN SPLITS
    # ========================================================

    rows = {
        split:
            read_jsonl(
                DATA / f"{split}.jsonl"
            )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    if not all(rows.values()):
        raise RuntimeError(
            "One or more Stage 4 splits "
            "are empty."
        )

    # ========================================================
    # CONVERSATION ID LEAKAGE
    # ========================================================

    ids = {
        split:
            {
                item["id"]
                for item in values
            }
        for split, values
        in rows.items()
    }

    train_validation_ids = (
        ids["train"]
        & ids["validation"]
    )

    train_test_ids = (
        ids["train"]
        & ids["test"]
    )

    validation_test_ids = (
        ids["validation"]
        & ids["test"]
    )

    print()
    print(
        "Conversation ID intersections:"
    )

    print(
        "  train-validation:",
        len(train_validation_ids),
    )

    print(
        "  train-test:",
        len(train_test_ids),
    )

    print(
        "  validation-test:",
        len(validation_test_ids),
    )

    if (
        train_validation_ids
        or train_test_ids
        or validation_test_ids
    ):
        raise RuntimeError(
            "Conversation ID leakage."
        )

    # ========================================================
    # PROMPT LEAKAGE
    # ========================================================

    prompts = {
        split:
            {
                normalize(
                    message["content"]
                )
                for item in values
                for message
                in item["messages"]
                if (
                    message["role"]
                    == "user"
                )
            }
        for split, values
        in rows.items()
    }

    train_validation_prompts = (
        prompts["train"]
        & prompts["validation"]
    )

    train_test_prompts = (
        prompts["train"]
        & prompts["test"]
    )

    validation_test_prompts = (
        prompts["validation"]
        & prompts["test"]
    )

    print()
    print(
        "Prompt intersections:"
    )

    print(
        "  train-validation:",
        len(
            train_validation_prompts
        ),
    )

    print(
        "  train-test:",
        len(
            train_test_prompts
        ),
    )

    print(
        "  validation-test:",
        len(
            validation_test_prompts
        ),
    )

    if (
        train_validation_prompts
        or train_test_prompts
        or validation_test_prompts
    ):
        raise RuntimeError(
            "Prompt leakage."
        )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            CFG["model_id"]
        )
    )

    tokenizer.pad_token = (
        tokenizer.pad_token
        or tokenizer.eos_token
    )

    print()
    print("Tokenizer:")

    print(
        "  model:",
        CFG["model_id"],
    )

    print(
        "  vocab:",
        len(tokenizer),
    )

    print(
        "  EOS:",
        tokenizer.eos_token_id,
    )

    print(
        "  PAD:",
        tokenizer.pad_token_id,
    )

    print(
        "  max_length:",
        CFG["max_length"],
    )

    # ========================================================
    # FULL SPLIT AUDITS
    # ========================================================

    results = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        results[split] = (
            audit_dataset(
                split,
                rows[split],
                tokenizer,
            )
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    global_assistant_segments = sum(
        result[
            "assistant_segments"
        ]
        for result in results.values()
    )

    global_accepted_segments = sum(
        result[
            "accepted_segments"
        ]
        for result in results.values()
    )

    global_rejected_segments = sum(
        result[
            "rejected_long_answers"
        ]
        for result in results.values()
    )

    global_context_truncated = sum(
        result[
            "context_truncated"
        ]
        for result in results.values()
    )

    global_zero_supervision = sum(
        result[
            "zero_supervision"
        ]
        for result in results.values()
    )

    global_accepted_percent = (
        100.0
        * global_accepted_segments
        / max(
            1,
            global_assistant_segments,
        )
    )

    global_rejected_percent = (
        100.0
        * global_rejected_segments
        / max(
            1,
            global_assistant_segments,
        )
    )

    global_context_percent = (
        100.0
        * global_context_truncated
        / max(
            1,
            global_assistant_segments,
        )
    )

    print()
    print("=" * 80)
    print("GLOBAL RESPONSE-ONLY SUMMARY")
    print("=" * 80)

    print()
    print(
        "Assistant segments:   ",
        f"{global_assistant_segments:,}",
    )

    print(
        "Accepted for SFT:     ",
        f"{global_accepted_segments:,}",
        f"({global_accepted_percent:.2f}%)",
    )

    print(
        "Rejected long answers:",
        f"{global_rejected_segments:,}",
        f"({global_rejected_percent:.2f}%)",
    )

    print(
        "Context truncated:    ",
        f"{global_context_truncated:,}",
        f"({global_context_percent:.2f}%)",
    )

    print(
        "Zero supervision:     ",
        f"{global_zero_supervision:,}",
    )

    # ========================================================
    # FROZEN CORPUS HASHES
    # ========================================================

    hashes = {
        split:
            sha256(
                DATA
                / f"{split}.jsonl"
            )
        for split in rows
    }

    print()
    print("=" * 80)
    print("CORPUS HASHES")
    print("=" * 80)

    for split in (
        "train",
        "validation",
        "test",
    ):
        print()
        print(
            split.upper()
        )

        print(
            "  conversations:",
            f"{len(rows[split]):,}",
        )

        print(
            "  sha256:",
            hashes[split],
        )

    # ========================================================
    # SAVE AUDIT REPORT
    # ========================================================

    report = {
        "stage":
            "stage4-v4-response-only-audit",

        "policy": {
            "max_length":
                CFG["max_length"],

            "context_truncation":
                "left",

            "assistant_answer_truncation":
                "forbidden",

            "assistant_answer_over_max_length":
                "reject_segment",

            "corpus_files_modified":
                False,
        },

        "model_id":
            CFG["model_id"],

        "results":
            results,

        "global": {
            "assistant_segments":
                global_assistant_segments,

            "accepted_segments":
                global_accepted_segments,

            "accepted_percent":
                global_accepted_percent,

            "rejected_long_answers":
                global_rejected_segments,

            "rejected_percent":
                global_rejected_percent,

            "context_truncated":
                global_context_truncated,

            "context_truncated_percent":
                global_context_percent,

            "zero_supervision":
                global_zero_supervision,
        },

        "corpus_sha256":
            hashes,

        "status":
            "PASS",
    }

    report_path = (
        OUT
        / "stage4-response-only-audit.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Audit report:",
        report_path,
    )

    print()
    print("=" * 80)
    print(
        "STAGE 4 FULL RESPONSE-ONLY AUDIT: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()