from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 - FULL ASSISTANT CONTROL
#
# DATASET PREPARATION V4
#
# Frozen target:
#
#   EN  7,200  40%
#   PT  5,400  30%
#   ES  5,400  30%
#   ----------------
#      18,000
#
# Sources:
#
# EN
#   Dolly                         5,200
#   OASST2                        2,000
#
# PT
#   OASST2                        all usable, up to 272
#   Ultra-Alpaca / ultrachat      2,500
#   Ultra-Alpaca / alpaca         1,400
#   Ultra-Alpaca / aya              700
#   Ultra-Alpaca / metamath       remainder
#
# ES
#   OASST2                        all usable, up to 4,046
#   latam-gpt/es-ultrachat        remainder
#
# Properties:
#
#   - deterministic
#   - exact language quotas
#   - explicit source quotas
#   - exact conversation deduplication
#   - content-connected anti-leakage grouping
#   - 90/5/5 split
#   - provenance manifest
#   - SHA-256 hashes
#   - FAIL CLOSED
#
# This script NEVER starts training.
# ============================================================


SEED = 42

SYSTEM = (
    "You are a helpful, careful multilingual assistant. "
    "Answer in the user's language unless asked otherwise."
)

OUTPUT_DIR = Path("data/stage4")

TARGET_TOTAL = 18_000

LANGUAGE_TARGETS = {
    "en": 7_200,
    "pt": 5_400,
    "es": 5_400,
}

EN_DOLLY_TARGET = 5_200
EN_OASST_TARGET = 2_000

PT_ULTRACHAT_TARGET = 2_500
PT_ALPACA_TARGET = 1_400
PT_AYA_TARGET = 700

# OASST2 PT and MetaMath are calculated dynamically:
#
# PT_OASST = all usable PT OASST2
# PT_METAMATH = 5400 - other PT components

ES_TARGET = 5_400

TRAIN_PERCENT = 90
VALIDATION_PERCENT = 5
TEST_PERCENT = 5

assert sum(LANGUAGE_TARGETS.values()) == TARGET_TOTAL


# ============================================================
# BASIC UTILITIES
# ============================================================


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_language(value) -> str | None:
    if value is None:
        return None

    value = str(value).strip().lower()

    aliases = {
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "por": "pt",
        "pt-br": "pt",
        "pt-pt": "pt",
        "spa": "es",
        "es-es": "es",
        "es-mx": "es",
    }

    if value in aliases:
        return aliases[value]

    if value.startswith("en"):
        return "en"

    if value.startswith("pt"):
        return "pt"

    if value.startswith("es"):
        return "es"

    return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def stable_id(*parts) -> str:
    return sha256_text(
        "\0".join(str(x) for x in parts)
    )[:24]


def deterministic_key(row) -> str:
    return sha256_text(
        f"{SEED}\0{row['id']}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# MESSAGE VALIDATION
# ============================================================


def valid_messages(messages) -> bool:
    if not messages:
        return False

    conversational = [
        m
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    if len(conversational) < 2:
        return False

    if conversational[0]["role"] != "user":
        return False

    if conversational[-1]["role"] != "assistant":
        return False

    expected = "user"

    for message in conversational:
        content = str(
            message.get("content") or ""
        ).strip()

        if not content:
            return False

        if message["role"] != expected:
            return False

        expected = (
            "assistant"
            if expected == "user"
            else "user"
        )

    return True


def with_system(messages):
    return [
        {
            "role": "system",
            "content": SYSTEM,
        }
    ] + messages


def user_prompts(row):
    return [
        normalize_text(m["content"])
        for m in row["messages"]
        if (
            m.get("role") == "user"
            and str(m.get("content") or "").strip()
        )
    ]


def conversation_key(row) -> str:
    parts = []

    for message in row["messages"]:
        role = str(
            message.get("role") or ""
        )

        content = normalize_text(
            message.get("content") or ""
        )

        parts.append(
            f"{role}:{content}"
        )

    return sha256_text(
        "\n".join(parts)
    )


# ============================================================
# GENERIC ROLE/CONTENT PARSER
# ============================================================


def parse_role_content_messages(raw_messages):
    if not isinstance(
        raw_messages,
        (list, tuple),
    ):
        return None

    messages = []

    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None

        role = str(
            raw.get("role") or ""
        ).strip().lower()

        content = str(
            raw.get("content") or ""
        ).strip()

        if not content:
            continue

        if role not in (
            "user",
            "assistant",
        ):
            return None

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    if not valid_messages(messages):
        return None

    return with_system(messages)


# ============================================================
# OASST2
# ============================================================


def load_oasst2():
    print()
    print("=" * 80)
    print("LOADING OPENASSISTANT OASST2")
    print("=" * 80)

    dataset = load_dataset(
        "OpenAssistant/oasst2",
        split="train",
    )

    rows = [
        dict(item)
        for item in dataset
    ]

    children = defaultdict(list)

    for item in rows:
        parent_id = item.get("parent_id")

        if parent_id:
            children[parent_id].append(
                item
            )

    roots = [
        item
        for item in rows
        if not item.get("parent_id")
    ]

    output = []

    for root in roots:
        language = normalize_language(
            root.get("lang")
        )

        if language not in (
            "en",
            "pt",
            "es",
        ):
            continue

        current = root
        visited = set()
        messages = []

        while current is not None:
            message_id = current.get(
                "message_id"
            )

            if not message_id:
                break

            if message_id in visited:
                break

            visited.add(message_id)

            text = str(
                current.get("text") or ""
            ).strip()

            raw_role = str(
                current.get("role") or ""
            ).lower()

            if raw_role == "assistant":
                role = "assistant"
            else:
                role = "user"

            if text:
                messages.append(
                    {
                        "role": role,
                        "content": text,
                    }
                )

            candidates = [
                candidate
                for candidate
                in children.get(
                    message_id,
                    [],
                )
                if not candidate.get(
                    "deleted",
                    False,
                )
            ]

            if not candidates:
                break

            candidates.sort(
                key=lambda candidate: (
                    candidate.get("rank")
                    if candidate.get("rank")
                    is not None
                    else 999999,
                    str(
                        candidate.get(
                            "message_id",
                            "",
                        )
                    ),
                )
            )

            current = candidates[0]

        if not valid_messages(messages):
            continue

        output.append(
            {
                "id":
                    "oasst2-"
                    + str(
                        root.get(
                            "message_id"
                        )
                    ),
                "source":
                    "oasst2",
                "subsource":
                    "oasst2",
                "language":
                    language,
                "category":
                    "assistant_conversation",
                "messages":
                    with_system(messages),
            }
        )

    counts = Counter(
        row["language"]
        for row in output
    )

    print(
        f"Accepted: {len(output):,}"
    )

    print(
        "Languages:",
        dict(sorted(counts.items())),
    )

    return output


# ============================================================
# DOLLY
# ============================================================


def load_dolly():
    print()
    print("=" * 80)
    print("LOADING DATABRICKS DOLLY 15K")
    print("=" * 80)

    dataset = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
    )

    output = []

    for item in dataset:
        instruction = str(
            item.get("instruction") or ""
        ).strip()

        context = str(
            item.get("context") or ""
        ).strip()

        response = str(
            item.get("response") or ""
        ).strip()

        category = str(
            item.get("category")
            or "instruction"
        ).strip()

        if not instruction or not response:
            continue

        user_text = instruction

        if context:
            user_text += (
                "\n\nContext:\n"
                + context
            )

        messages = [
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        if not valid_messages(messages):
            continue

        output.append(
            {
                "id":
                    "dolly-"
                    + stable_id(
                        instruction,
                        context,
                        response,
                    ),
                "source":
                    "dolly",
                "subsource":
                    "dolly",
                "language":
                    "en",
                "category":
                    category,
                "messages":
                    with_system(messages),
            }
        )

    print(
        f"Accepted: {len(output):,}"
    )

    return output


# ============================================================
# ULTRA-ALPACA PT-BR
#
# Confirmed schema:
#
#   prompt_id
#   data = [
#       {"role": "...", "content": "..."},
#       ...
#   ]
#   src
# ============================================================


def load_ultra_alpaca_ptbr():
    print()
    print("=" * 80)
    print("LOADING RECOGNA ULTRA-ALPACA PT-BR")
    print("=" * 80)

    dataset = load_dataset(
        "recogna-nlp/ultra-alpaca-ptbr",
        split="train",
    )

    print(
        "Columns:",
        dataset.column_names,
    )

    output = []
    rejected = 0

    for item in dataset:
        messages = parse_role_content_messages(
            item.get("data")
        )

        if messages is None:
            rejected += 1
            continue

        subsource = str(
            item.get("src")
            or "unknown"
        ).strip()

        prompt_id = item.get(
            "prompt_id"
        )

        output.append(
            {
                "id":
                    "ultra-ptbr-"
                    + stable_id(
                        prompt_id,
                        conversation_key(
                            {
                                "messages":
                                    messages
                            }
                        ),
                    ),
                "source":
                    "ultra-alpaca-ptbr",
                "subsource":
                    subsource,
                "language":
                    "pt",
                "category":
                    "assistant_instruction",
                "messages":
                    messages,
            }
        )

    counts = Counter(
        row["subsource"]
        for row in output
    )

    print(
        f"Accepted: {len(output):,}"
    )

    print(
        f"Rejected: {rejected:,}"
    )

    print()
    print("PT-BR internal sources:")

    for name, count in (
        counts.most_common()
    ):
        print(
            f"  {name}: {count:,}"
        )

    return output


# ============================================================
# LATAM-GPT ES-ULTRACHAT
#
# Confirmed schema:
#
#   id
#   messages = [
#       {"role": "user", "content": "..."},
#       {"role": "assistant", "content": "..."},
#       ...
#   ]
#   lang = "es"
# ============================================================


def load_es_ultrachat():
    print()
    print("=" * 80)
    print("LOADING LATAM-GPT ES-ULTRACHAT")
    print("=" * 80)

    dataset = load_dataset(
        "latam-gpt/es-ultrachat",
        split="train",
    )

    print(
        "Columns:",
        dataset.column_names,
    )

    output = []
    rejected = 0

    for item in dataset:
        language = normalize_language(
            item.get("lang")
        )

        if language != "es":
            rejected += 1
            continue

        messages = parse_role_content_messages(
            item.get("messages")
        )

        if messages is None:
            rejected += 1
            continue

        output.append(
            {
                "id":
                    "es-ultrachat-"
                    + stable_id(
                        item.get("id"),
                        conversation_key(
                            {
                                "messages":
                                    messages
                            }
                        ),
                    ),
                "source":
                    "latam-gpt-es-ultrachat",
                "subsource":
                    "translated-ultrachat",
                "language":
                    "es",
                "category":
                    "assistant_conversation",
                "messages":
                    messages,
            }
        )

    print(
        f"Accepted: {len(output):,}"
    )

    print(
        f"Rejected: {rejected:,}"
    )

    if not output:
        raise RuntimeError(
            "No usable ES-UltraChat "
            "conversations."
        )

    return output


# ============================================================
# EXACT DEDUPLICATION
# ============================================================


def deduplicate_exact(rows):
    seen = set()
    output = []
    removed = 0

    for row in rows:
        key = conversation_key(row)

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        output.append(row)

    return output, removed


# ============================================================
# DETERMINISTIC SELECTION
# ============================================================


def select_rows(
    rows,
    count,
    label,
):
    rows = sorted(
        rows,
        key=deterministic_key,
    )

    if len(rows) < count:
        raise RuntimeError(
            f"{label}: need {count:,}, "
            f"found {len(rows):,}"
        )

    selected = rows[:count]

    print(
        f"{label}: "
        f"{len(selected):,} / "
        f"{len(rows):,}"
    )

    return selected


def rows_by_subsource(
    rows,
    subsource,
):
    return [
        row
        for row in rows
        if row["subsource"]
        == subsource
    ]


# ============================================================
# FIXED COMPOSITION
# ============================================================


def build_fixed_composition(
    oasst,
    dolly,
    ultra_pt,
    es_ultrachat,
):
    print()
    print("=" * 80)
    print("BUILDING FIXED V4 COMPOSITION")
    print("=" * 80)

    oasst_en = [
        x for x in oasst
        if x["language"] == "en"
    ]

    oasst_pt = [
        x for x in oasst
        if x["language"] == "pt"
    ]

    oasst_es = [
        x for x in oasst
        if x["language"] == "es"
    ]

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    en_dolly = select_rows(
        dolly,
        EN_DOLLY_TARGET,
        "EN / Dolly",
    )

    en_oasst = select_rows(
        oasst_en,
        EN_OASST_TARGET,
        "EN / OASST2",
    )

    english = (
        en_dolly
        + en_oasst
    )

    assert (
        len(english)
        == LANGUAGE_TARGETS["en"]
    )

    # --------------------------------------------------------
    # PORTUGUESE
    #
    # Preserve all usable direct OASST2 PT examples.
    # Remaining quota is explicitly controlled by subsource.
    # --------------------------------------------------------

    pt_oasst = sorted(
        oasst_pt,
        key=deterministic_key,
    )

    if len(pt_oasst) > 400:
        raise RuntimeError(
            "Unexpectedly large change in "
            "OASST2 PT availability. "
            "Review quotas before freezing."
        )

    pt_ultrachat = select_rows(
        rows_by_subsource(
            ultra_pt,
            "ultrachat_ptbr",
        ),
        PT_ULTRACHAT_TARGET,
        "PT / UltraChat PT-BR",
    )

    pt_alpaca = select_rows(
        rows_by_subsource(
            ultra_pt,
            "alpaca_ptbr",
        ),
        PT_ALPACA_TARGET,
        "PT / Alpaca PT-BR",
    )

    pt_aya = select_rows(
        rows_by_subsource(
            ultra_pt,
            "aya",
        ),
        PT_AYA_TARGET,
        "PT / Aya",
    )

    pt_metamath_target = (
        LANGUAGE_TARGETS["pt"]
        - len(pt_oasst)
        - PT_ULTRACHAT_TARGET
        - PT_ALPACA_TARGET
        - PT_AYA_TARGET
    )

    if pt_metamath_target <= 0:
        raise RuntimeError(
            "Invalid PT MetaMath quota."
        )

    pt_metamath = select_rows(
        rows_by_subsource(
            ultra_pt,
            "metamathqa_ptbr",
        ),
        pt_metamath_target,
        "PT / MetaMathQA PT-BR",
    )

    portuguese = (
        pt_oasst
        + pt_ultrachat
        + pt_alpaca
        + pt_aya
        + pt_metamath
    )

    if (
        len(portuguese)
        != LANGUAGE_TARGETS["pt"]
    ):
        raise RuntimeError(
            "PT quota mismatch."
        )

    # --------------------------------------------------------
    # SPANISH
    #
    # Preserve all usable OASST2 ES examples.
    # Complete exactly to 5,400 using ES-UltraChat.
    # --------------------------------------------------------

    es_oasst = sorted(
        oasst_es,
        key=deterministic_key,
    )

    if len(es_oasst) > ES_TARGET:
        es_oasst = es_oasst[
            :ES_TARGET
        ]

    es_ultra_target = (
        ES_TARGET
        - len(es_oasst)
    )

    if es_ultra_target < 0:
        raise RuntimeError(
            "Invalid ES quota."
        )

    es_ultra = select_rows(
        es_ultrachat,
        es_ultra_target,
        "ES / Latam-GPT ES-UltraChat",
    )

    spanish = (
        es_oasst
        + es_ultra
    )

    if (
        len(spanish)
        != LANGUAGE_TARGETS["es"]
    ):
        raise RuntimeError(
            "ES quota mismatch."
        )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    selected = (
        english
        + portuguese
        + spanish
    )

    counts = Counter(
        row["language"]
        for row in selected
    )

    expected = Counter(
        LANGUAGE_TARGETS
    )

    if counts != expected:
        raise RuntimeError(
            "Final language composition "
            f"mismatch: {counts}"
        )

    if len(selected) != TARGET_TOTAL:
        raise RuntimeError(
            "Final total mismatch."
        )

    print()
    print("V4 source composition:")

    source_detail = Counter(
        (
            row["language"],
            row["source"],
            row["subsource"],
        )
        for row in selected
    )

    for key in sorted(
        source_detail
    ):
        language, source, subsource = key

        print(
            f"  {language:2s} | "
            f"{source:28s} | "
            f"{subsource:24s} | "
            f"{source_detail[key]:,}"
        )

    print()
    print("Language totals:")

    for language in (
        "en",
        "pt",
        "es",
    ):
        count = counts[language]

        print(
            f"  {language}: "
            f"{count:,} "
            f"({100 * count / TARGET_TOTAL:.2f}%)"
        )

    return selected


# ============================================================
# UNION-FIND CONTENT GROUPING
# ============================================================


class UnionFind:
    def __init__(self, size):
        self.parent = list(
            range(size)
        )

        self.rank = [
            0
            for _ in range(size)
        ]

    def find(self, value):
        while (
            self.parent[value]
            != value
        ):
            self.parent[value] = (
                self.parent[
                    self.parent[value]
                ]
            )

            value = self.parent[value]

        return value

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):
            root_a, root_b = (
                root_b,
                root_a,
            )

        self.parent[root_b] = root_a

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):
            self.rank[root_a] += 1


def build_content_groups(rows):
    uf = UnionFind(
        len(rows)
    )

    prompt_owner = {}

    for index, row in enumerate(rows):
        prompts = set(
            user_prompts(row)
        )

        for prompt in prompts:
            if not prompt:
                continue

            if prompt in prompt_owner:
                uf.union(
                    index,
                    prompt_owner[prompt],
                )
            else:
                prompt_owner[
                    prompt
                ] = index

    groups = defaultdict(list)

    for index, row in enumerate(rows):
        groups[
            uf.find(index)
        ].append(row)

    groups = list(
        groups.values()
    )

    print()
    print("=" * 80)
    print("CONTENT GROUPING")
    print("=" * 80)

    print(
        f"Conversations: {len(rows):,}"
    )

    print(
        f"Content groups: {len(groups):,}"
    )

    print(
        "Largest group:",
        max(
            len(group)
            for group in groups
        ),
    )

    return groups


# ============================================================
# SPLITTING
#
# Whole groups are assigned to one split.
# We use dominant language only for stratification.
# ============================================================


def split_groups(groups):
    buckets = defaultdict(list)

    for group in groups:
        language_counts = Counter(
            row["language"]
            for row in group
        )

        dominant_language = sorted(
            language_counts.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )[0][0]

        signature = sha256_text(
            "\n".join(
                sorted(
                    row["id"]
                    for row in group
                )
            )
        )

        buckets[
            dominant_language
        ].append(
            (
                signature,
                group,
            )
        )

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for language in (
        "en",
        "pt",
        "es",
    ):
        items = sorted(
            buckets[language],
            key=lambda x: x[0],
        )

        for signature, group in items:
            value = int(
                signature[:12],
                16,
            ) % 100

            if value < 90:
                split = "train"

            elif value < 95:
                split = "validation"

            else:
                split = "test"

            splits[split].extend(
                group
            )

    return splits


# ============================================================
# LEAKAGE AUDIT
# ============================================================


def prompt_set(rows):
    prompts = set()

    for row in rows:
        prompts.update(
            user_prompts(row)
        )

    prompts.discard("")

    return prompts


def audit_leakage(splits):
    prompts = {
        name: prompt_set(rows)
        for name, rows
        in splits.items()
    }

    tv = (
        prompts["train"]
        & prompts["validation"]
    )

    tt = (
        prompts["train"]
        & prompts["test"]
    )

    vt = (
        prompts["validation"]
        & prompts["test"]
    )

    print()
    print("=" * 80)
    print("PROMPT LEAKAGE AUDIT")
    print("=" * 80)

    print(
        "train <-> validation:",
        len(tv),
    )

    print(
        "train <-> test:",
        len(tt),
    )

    print(
        "validation <-> test:",
        len(vt),
    )

    if tv or tt or vt:
        raise RuntimeError(
            "Prompt leakage detected."
        )

    print("Prompt leakage: PASS")


# ============================================================
# SPLIT AUDIT
# ============================================================


def audit_splits(splits):
    print()
    print("=" * 80)
    print("SPLIT AUDIT")
    print("=" * 80)

    total = sum(
        len(rows)
        for rows in splits.values()
    )

    if total != TARGET_TOTAL:
        raise RuntimeError(
            f"Split total mismatch: {total}"
        )

    for split in (
        "train",
        "validation",
        "test",
    ):
        rows = splits[split]

        languages = Counter(
            row["language"]
            for row in rows
        )

        print()
        print(split.upper())

        print(
            f"  Conversations: "
            f"{len(rows):,}"
        )

        for language in (
            "en",
            "pt",
            "es",
        ):
            count = languages[
                language
            ]

            percentage = (
                100.0
                * count
                / len(rows)
            )

            print(
                f"  {language}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

            if count == 0:
                raise RuntimeError(
                    f"{language} absent "
                    f"from {split}"
                )


# ============================================================
# PROVENANCE
# ============================================================


def field_counts(
    rows,
    field,
):
    return dict(
        sorted(
            Counter(
                row.get(
                    field,
                    "unknown",
                )
                for row in rows
            ).items()
        )
    )


def provenance(rows):
    counter = Counter(
        (
            row["language"],
            row["source"],
            row["subsource"],
            row["category"],
        )
        for row in rows
    )

    return [
        {
            "language": language,
            "source": source,
            "subsource": subsource,
            "category": category,
            "count": count,
        }
        for (
            language,
            source,
            subsource,
            category,
        ), count
        in sorted(
            counter.items()
        )
    ]


# ============================================================
# JSONL OUTPUT
# ============================================================


def write_jsonl(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )

            handle.write("\n")


# ============================================================
# MAIN
# ============================================================


def main():
    print()
    print("=" * 80)
    print(
        "MAIA SFT VALIDATION - STAGE 4"
    )
    print(
        "DATASET PREPARATION V4"
    )
    print("=" * 80)

    print()
    print(
        "Target: "
        "EN 7,200 / PT 5,400 / "
        "ES 5,400"
    )

    print(
        "Total: 18,000"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    oasst = load_oasst2()
    dolly = load_dolly()
    ultra_pt = (
        load_ultra_alpaca_ptbr()
    )
    es_ultrachat = (
        load_es_ultrachat()
    )

    # --------------------------------------------------------
    # Deduplicate each source independently
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SOURCE DEDUPLICATION")
    print("=" * 80)

    datasets = {}

    for name, rows in (
        ("oasst2", oasst),
        ("dolly", dolly),
        (
            "ultra-alpaca-ptbr",
            ultra_pt,
        ),
        (
            "latam-gpt-es-ultrachat",
            es_ultrachat,
        ),
    ):
        clean, removed = (
            deduplicate_exact(rows)
        )

        datasets[name] = clean

        print(
            f"{name}: "
            f"{len(clean):,} kept, "
            f"{removed:,} removed"
        )

    # --------------------------------------------------------
    # Fixed quotas
    # --------------------------------------------------------

    selected = (
        build_fixed_composition(
            datasets["oasst2"],
            datasets["dolly"],
            datasets[
                "ultra-alpaca-ptbr"
            ],
            datasets[
                "latam-gpt-es-ultrachat"
            ],
        )
    )

    # --------------------------------------------------------
    # Cross-source exact duplicate audit
    # --------------------------------------------------------

    clean_selected, removed = (
        deduplicate_exact(
            selected
        )
    )

    print()
    print(
        "Cross-source exact duplicates:",
        removed,
    )

    if removed:
        raise RuntimeError(
            "Cross-source duplicates found "
            "after quota selection. "
            "Do not train until quota refill "
            "logic is reviewed."
        )

    selected = clean_selected

    # --------------------------------------------------------
    # Exact composition audit
    # --------------------------------------------------------

    language_counts = Counter(
        row["language"]
        for row in selected
    )

    if (
        language_counts
        != Counter(LANGUAGE_TARGETS)
    ):
        raise RuntimeError(
            "Language target mismatch: "
            f"{dict(language_counts)}"
        )

    if len(selected) != TARGET_TOTAL:
        raise RuntimeError(
            "Dataset total mismatch."
        )

    # --------------------------------------------------------
    # Content-connected groups
    # --------------------------------------------------------

    groups = build_content_groups(
        selected
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    splits = split_groups(
        groups
    )

    # --------------------------------------------------------
    # Mandatory audits
    # --------------------------------------------------------

    audit_leakage(
        splits
    )

    audit_splits(
        splits
    )

    # --------------------------------------------------------
    # Stable output order + hashes
    # --------------------------------------------------------

    hashes = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        rows = sorted(
            splits[split],
            key=lambda row: row["id"],
        )

        splits[split] = rows

        path = (
            OUTPUT_DIR
            / f"{split}.jsonl"
        )

        write_jsonl(
            path,
            rows,
        )

        hashes[split] = (
            file_sha256(path)
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "stage":
            "stage4-full-assistant-control-v4",

        "seed":
            SEED,

        "total":
            TARGET_TOTAL,

        "target_languages":
            LANGUAGE_TARGETS,

        "language_source_policy": {
            "en": {
                "dolly": 5200,
                "oasst2": 2000,
            },

            "pt": {
                "oasst2":
                    "all usable direct OASST2 PT",
                "ultrachat_ptbr":
                    2500,
                "alpaca_ptbr":
                    1400,
                "aya":
                    700,
                "metamathqa_ptbr":
                    "remainder to exactly 5400",
                "codealpaca_ptbr":
                    0,
            },

            "es": {
                "oasst2":
                    "all usable direct OASST2 ES",
                "latam-gpt-es-ultrachat":
                    "remainder to exactly 5400",
            },
        },

        "excluded": {
            "codealpaca_ptbr":
                "excluded from Stage 4 V4 "
                "after manual inspection "
                "showed corrupted translated "
                "programming syntax",
        },

        "split_policy": {
            "train":
                TRAIN_PERCENT,
            "validation":
                VALIDATION_PERCENT,
            "test":
                TEST_PERCENT,
            "anti_leakage":
                "content-connected groups "
                "sharing any normalized user "
                "prompt are indivisible",
        },

        "counts": {
            split:
                len(rows)
            for split, rows
            in splits.items()
        },

        "languages": {
            split:
                field_counts(
                    rows,
                    "language",
                )
            for split, rows
            in splits.items()
        },

        "sources": {
            split:
                field_counts(
                    rows,
                    "source",
                )
            for split, rows
            in splits.items()
        },

        "subsources": {
            split:
                field_counts(
                    rows,
                    "subsource",
                )
            for split, rows
            in splits.items()
        },

        "provenance": {
            split:
                provenance(rows)
            for split, rows
            in splits.items()
        },

        "sha256": hashes,

        "audit": {
            "prompt_leakage":
                "PASS",
            "cross_source_exact_duplicates":
                0,
            "target_total":
                TARGET_TOTAL,
            "target_language_composition":
                LANGUAGE_TARGETS,
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL STAGE 4 V4 DATASET")
    print("=" * 80)

    for split in (
        "train",
        "validation",
        "test",
    ):
        print()
        print(split.upper())

        print(
            "  Conversations:",
            f"{len(splits[split]):,}",
        )

        print(
            "  Languages:",
            field_counts(
                splits[split],
                "language",
            ),
        )

        print(
            "  Sources:",
            field_counts(
                splits[split],
                "source",
            ),
        )

        print(
            "  Subsources:",
            field_counts(
                splits[split],
                "subsource",
            ),
        )

        print(
            "  SHA-256:",
            hashes[split],
        )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    print()
    print("=" * 80)
    print(
        "STAGE 4 V4 DATASET PREPARATION: PASS"
    )
    print("=" * 80)

    print()
    print(
        "TRAINING HAS NOT STARTED."
    )

    print(
        "Review this output before "
        "building the benchmark or training."
    )


if __name__ == "__main__":
    main()