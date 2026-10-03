from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from src.common_stage4 import assistant_segments


@dataclass
class DatasetAudit:
    conversations: int = 0
    assistant_segments: int = 0
    accepted_segments: int = 0

    over_max_length: int = 0
    context_truncated: int = 0
    answer_truncated: int = 0
    zero_supervision: int = 0

    total_original_tokens: int = 0
    total_final_tokens: int = 0

    total_prefix_tokens: int = 0
    total_answer_tokens: int = 0

    removed_context_tokens: int = 0
    removed_answer_tokens: int = 0

    min_supervised_tokens: int | None = None
    max_supervised_tokens: int = 0

    max_original_length: int = 0
    max_final_length: int = 0

    def record_supervised(
        self,
        count: int,
    ):
        if self.min_supervised_tokens is None:
            self.min_supervised_tokens = count
        else:
            self.min_supervised_tokens = min(
                self.min_supervised_tokens,
                count,
            )

        self.max_supervised_tokens = max(
            self.max_supervised_tokens,
            count,
        )

    def as_dict(self):
        return {
            "conversations":
                self.conversations,

            "assistant_segments":
                self.assistant_segments,

            "accepted_segments":
                self.accepted_segments,

            "over_max_length":
                self.over_max_length,

            "context_truncated":
                self.context_truncated,

            "answer_truncated":
                self.answer_truncated,

            "zero_supervision":
                self.zero_supervision,

            "total_original_tokens":
                self.total_original_tokens,

            "total_final_tokens":
                self.total_final_tokens,

            "total_prefix_tokens":
                self.total_prefix_tokens,

            "total_answer_tokens":
                self.total_answer_tokens,

            "removed_context_tokens":
                self.removed_context_tokens,

            "removed_answer_tokens":
                self.removed_answer_tokens,

            "min_supervised_tokens":
                self.min_supervised_tokens,

            "max_supervised_tokens":
                self.max_supervised_tokens,

            "max_original_length":
                self.max_original_length,

            "max_final_length":
                self.max_final_length,
        }


class ResponseOnlyDataset(Dataset):
    """
    Response-only causal-LM dataset.

    Each assistant response becomes one training example.

    Policy:

    1. Assistant response + EOS must fit entirely inside
       max_length.

    2. If prefix + response exceeds max_length, truncate
       only the LEFT side of the prefix.

    3. Never silently truncate assistant response tokens.

    4. Labels for prefix tokens are -100.

    5. All assistant response tokens, including EOS, are
       supervised.

    This preserves the response-only SFT objective while
    making truncation behavior explicit and auditable.
    """

    def __init__(
        self,
        rows,
        tok,
        max_length,
        *,
        reject_answer_truncation=True,
    ):
        self.items = []
        self.tok = tok
        self.max_length = max_length

        self.audit = DatasetAudit(
            conversations=len(rows)
        )

        self.rejected = []

        for row_index, row in enumerate(rows):
            segments = assistant_segments(
                row["messages"]
            )

            for segment_index, (
                prefix,
                answer,
            ) in enumerate(segments):
                self.audit.assistant_segments += 1

                prefix_ids = tok(
                    prefix,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                answer_ids = tok(
                    answer,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                answer_ids = (
                    answer_ids
                    + [tok.eos_token_id]
                )

                prefix_length = len(
                    prefix_ids
                )

                answer_length = len(
                    answer_ids
                )

                original_length = (
                    prefix_length
                    + answer_length
                )

                self.audit.total_prefix_tokens += (
                    prefix_length
                )

                self.audit.total_answer_tokens += (
                    answer_length
                )

                self.audit.total_original_tokens += (
                    original_length
                )

                self.audit.max_original_length = max(
                    self.audit.max_original_length,
                    original_length,
                )

                # --------------------------------------------
                # Assistant answer itself cannot fit.
                # --------------------------------------------

                if answer_length > max_length:
                    self.audit.answer_truncated += 1

                    removed = (
                        answer_length
                        - max_length
                    )

                    self.audit.removed_answer_tokens += (
                        removed
                    )

                    rejected_record = {
                        "row_index":
                            row_index,

                        "segment_index":
                            segment_index,

                        "row_id":
                            row.get("id"),

                        "reason":
                            "assistant_answer_exceeds_max_length",

                        "prefix_tokens":
                            prefix_length,

                        "answer_tokens":
                            answer_length,

                        "max_length":
                            max_length,
                    }

                    self.rejected.append(
                        rejected_record
                    )

                    if reject_answer_truncation:
                        continue

                    # This branch exists only for explicit
                    # experimentation. It is NOT the Stage 4
                    # default policy.

                    answer_ids = answer_ids[
                        -max_length:
                    ]

                    answer_length = len(
                        answer_ids
                    )

                    prefix_ids = []

                    prefix_length = 0

                # --------------------------------------------
                # Prefix truncation only.
                # --------------------------------------------

                available_prefix = (
                    max_length
                    - answer_length
                )

                if (
                    prefix_length
                    > available_prefix
                ):
                    self.audit.over_max_length += 1
                    self.audit.context_truncated += 1

                    remove_count = (
                        prefix_length
                        - available_prefix
                    )

                    self.audit.removed_context_tokens += (
                        remove_count
                    )

                    if available_prefix > 0:
                        prefix_ids = prefix_ids[
                            -available_prefix:
                        ]
                    else:
                        prefix_ids = []

                # --------------------------------------------
                # Final sequence
                # --------------------------------------------

                input_ids = (
                    prefix_ids
                    + answer_ids
                )

                supervised_tokens = len(
                    answer_ids
                )

                labels = (
                    [-100] * len(prefix_ids)
                    + list(answer_ids)
                )

                if not any(
                    value != -100
                    for value in labels
                ):
                    self.audit.zero_supervision += 1
                    continue

                if len(input_ids) > max_length:
                    raise RuntimeError(
                        "Internal truncation error: "
                        f"{len(input_ids)} > "
                        f"{max_length}"
                    )

                if len(input_ids) != len(labels):
                    raise RuntimeError(
                        "input_ids/labels length mismatch"
                    )

                first_supervised = len(
                    prefix_ids
                )

                # Response-only invariant:
                # prefix must never be supervised.
                if any(
                    value != -100
                    for value
                    in labels[
                        :first_supervised
                    ]
                ):
                    raise RuntimeError(
                        "Prefix supervision detected."
                    )

                # Response invariant:
                # every answer token must be supervised.
                if (
                    labels[first_supervised:]
                    != answer_ids
                ):
                    raise RuntimeError(
                        "Assistant supervision mismatch."
                    )

                self.audit.accepted_segments += 1

                self.audit.total_final_tokens += (
                    len(input_ids)
                )

                self.audit.max_final_length = max(
                    self.audit.max_final_length,
                    len(input_ids),
                )

                self.audit.record_supervised(
                    supervised_tokens
                )

                self.items.append(
                    {
                        "input_ids":
                            input_ids,

                        "labels":
                            labels,
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        max_length = max(
            len(item["input_ids"])
            for item in batch
        )

        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            padding = (
                max_length
                - len(item["input_ids"])
            )

            input_ids.append(
                item["input_ids"]
                + (
                    [self.tok.pad_token_id]
                    * padding
                )
            )

            labels.append(
                item["labels"]
                + (
                    [-100]
                    * padding
                )
            )

            attention_mask.append(
                (
                    [1]
                    * len(item["input_ids"])
                )
                + (
                    [0]
                    * padding
                )
            )

        return {
            "input_ids":
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
        }