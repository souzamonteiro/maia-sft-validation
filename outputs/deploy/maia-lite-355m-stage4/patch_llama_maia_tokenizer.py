
#!/usr/bin/env python3

# ============================================================
# MAIA LITE 355M
# llama.cpp tokenizer compatibility patch
#
# Maia Lite uses a custom 50,257-token ByteLevel BPE tokenizer.
#
# Its vocabulary and merges are Maia-specific, but its
# pre-tokenization algorithm is the GPT-2 ByteLevel BPE
# pre-tokenizer already implemented by llama.cpp.
#
# This patch maps the Maia tokenizer fingerprint to the
# existing "gpt2" GGUF pre-tokenizer implementation.
#
# It DOES NOT modify:
#   - Maia vocabulary
#   - Maia merges
#   - tokenizer.json
#   - model weights
# ============================================================

from pathlib import Path
import os
import sys
import shutil


MAIA_TOKENIZER_HASH = (
    "a727209002a4bef5c366d1ebedf90abc"
    "1d4c85d919dfa18a2364d27747dcb599"
)

MAIA_PRETOKENIZER = "gpt2"


# ============================================================
# LOCATE llama.cpp
# ============================================================

default_llama_dir = Path(
    "/Volumes/External_SSD/Documentos/Projects/llama.cpp"
)

llama_dir = Path(
    os.environ.get(
        "LLAMA_CPP_DIR",
        default_llama_dir,
    )
).resolve()

base_py = (
    llama_dir
    / "conversion"
    / "base.py"
)


print("=" * 80)
print("MAIA LITE - llama.cpp TOKENIZER PATCH")
print("=" * 80)

print()
print("llama.cpp:")
print(llama_dir)

print()
print("Target:")
print(base_py)

print()
print("Maia tokenizer hash:")
print(MAIA_TOKENIZER_HASH)

print()
print(
    "GGUF pre-tokenizer:",
    MAIA_PRETOKENIZER,
)


# ============================================================
# PREFLIGHT
# ============================================================

if not llama_dir.is_dir():
    raise RuntimeError(
        f"llama.cpp directory not found:\n{llama_dir}"
    )

if not base_py.is_file():
    raise RuntimeError(
        f"conversion/base.py not found:\n{base_py}"
    )


source = base_py.read_text(
    encoding="utf-8"
)


# ============================================================
# VERIFY EXPECTED STRUCTURE
# ============================================================

required_fragments = [
    "def get_vocab_base_pre",
    "chkhsh = sha256",
    "res = None",
    "BPE pre-tokenizer was not recognized",
]

for fragment in required_fragments:

    if fragment not in source:

        raise RuntimeError(
            "Unexpected llama.cpp conversion/base.py structure.\n"
            f"Missing fragment: {fragment!r}\n"
            "Refusing to patch."
        )


# ============================================================
# ALREADY SUPPORTED?
# ============================================================

if MAIA_TOKENIZER_HASH in source:

    print()
    print(
        "Maia tokenizer hash is already present."
    )

    print()
    print(
        "No modification required."
    )

    print()
    print("=" * 80)
    print("PATCH STATUS: ALREADY PRESENT")
    print("=" * 80)

    sys.exit(0)


# ============================================================
# PATCH
# ============================================================

anchor = "        res = None\n"

if anchor not in source:

    raise RuntimeError(
        "Could not locate insertion point "
        "'        res = None'."
    )


patch = (
    anchor
    + "\n"
    + "        # Maia Lite 355M custom ByteLevel BPE tokenizer\n"
    + "        # Vocabulary and merges are Maia-specific.\n"
    + "        # Pre-tokenization is GPT-2 ByteLevel BPE.\n"
    + f'        if chkhsh == "{MAIA_TOKENIZER_HASH}":\n'
    + f'            res = "{MAIA_PRETOKENIZER}"\n'
)


patched_source = source.replace(
    anchor,
    patch,
    1,
)


# ============================================================
# BACKUP
# ============================================================

backup = base_py.with_suffix(
    ".py.maia-backup"
)

if not backup.exists():

    shutil.copy2(
        base_py,
        backup,
    )

    print()
    print("Backup created:")
    print(backup)

else:

    print()
    print("Backup already exists:")
    print(backup)


# ============================================================
# WRITE PATCH
# ============================================================

base_py.write_text(
    patched_source,
    encoding="utf-8",
)


# ============================================================
# VERIFY
# ============================================================

verification = base_py.read_text(
    encoding="utf-8"
)

if MAIA_TOKENIZER_HASH not in verification:

    raise RuntimeError(
        "Patch verification failed."
    )

expected_mapping = (
    f'if chkhsh == "{MAIA_TOKENIZER_HASH}":'
)

if expected_mapping not in verification:

    raise RuntimeError(
        "Hash mapping verification failed."
    )


print()
print("Mapping installed:")
print()
print(
    f"  {MAIA_TOKENIZER_HASH}"
)
print("      ->")
print(
    f"  tokenizer.ggml.pre = "
    f"{MAIA_PRETOKENIZER!r}"
)

print()
print("=" * 80)
print("PATCH COMPLETE: PASS")
print("=" * 80)