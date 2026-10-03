#!/usr/bin/env python3

# ============================================================
# MAIA SFT VALIDATION
# GPT-2 MEDIUM + MPS-SAFE LoRA
# INTERACTIVE CHAT TEST
#
# Loads:
#   - openai-community/gpt2-medium
#   - best Stage 4 MPS-safe LoRA adapter
#
# Uses the EXACT Stage 4 prompt serialization through
# src.common_stage4.assistant_segments().
#
# Generation:
#   - Apple MPS
#   - greedy decoding
#   - max_new_tokens = 192
#   - no sampling
# ============================================================

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    "/Volumes/External_SSD/Documentos/Projects/maia-sft-validation"
)

BEST_ADAPTER_DIR = (
    PROJECT_DIR
    / "outputs"
    / "gpt2-medium-stage4-lora-mps-safe-best"
)

BASE_MODEL_ID = "openai-community/gpt2-medium"

MAX_CONTEXT = 1024
MAX_NEW_TOKENS = 192


# ============================================================
# PROJECT IMPORTS
# ============================================================

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.common_stage4 import SYSTEM, assistant_segments


# ============================================================
# DEVICE
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "Apple MPS is not available. "
        "This script is intended for the Mac mini."
    )

device = torch.device("mps")


# ============================================================
# PREFLIGHT
# ============================================================

print("=" * 80)
print("GPT-2 MEDIUM + MPS-SAFE LoRA - INTERACTIVE CHAT")
print("=" * 80)

if not PROJECT_DIR.exists():
    raise FileNotFoundError(
        f"Project directory not found:\n{PROJECT_DIR}"
    )

if not BEST_ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Best LoRA adapter not found:\n{BEST_ADAPTER_DIR}"
    )

print()
print("Device:        ", device)
print("Base model:    ", BASE_MODEL_ID)
print("LoRA adapter:  ", BEST_ADAPTER_DIR)
print("System:        ", SYSTEM)
print()


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        BEST_ADAPTER_DIR,
        local_files_only=True,
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID
    )

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Vocabulary:    ", len(tokenizer))
print("EOS token ID:  ", tokenizer.eos_token_id)
print("PAD token ID:  ", tokenizer.pad_token_id)


# ============================================================
# BASE MODEL
# ============================================================

print()
print("Loading GPT-2 Medium...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float32,
)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.use_cache = True


# ============================================================
# LoRA ADAPTER
# ============================================================

print("Loading Stage 4 MPS-safe LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    BEST_ADAPTER_DIR,
    is_trainable=False,
)

model = model.to(device)
model.eval()

model.config.use_cache = True
model.config.pad_token_id = tokenizer.pad_token_id

print()
print("Model ready.")
print()


# ============================================================
# EXACT STAGE 4 PREFIX
# ============================================================

def build_stage4_prefix(user_text):
    """
    Build the prompt using the exact Stage 4 serialization.

    The dummy assistant response exists only so that
    assistant_segments() returns the prefix corresponding
    to the assistant turn.

    The dummy response itself is never passed to the model.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {
            "role": "user",
            "content": user_text,
        },
        {
            "role": "assistant",
            "content": "__MAIA_GENERATION_PLACEHOLDER__",
        },
    ]

    segments = list(
        assistant_segments(messages)
    )

    if len(segments) != 1:
        raise RuntimeError(
            "Unexpected Stage 4 serialization result: "
            f"{len(segments)} assistant segments."
        )

    prefix, _ = segments[0]

    return prefix


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def ask(user_text):
    prefix = build_stage4_prefix(user_text)

    encoded = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Reserve room for the generated response.
    max_prompt_tokens = (
        MAX_CONTEXT - MAX_NEW_TOKENS
    )

    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[
            :,
            -max_prompt_tokens:
        ]

        attention_mask = attention_mask[
            :,
            -max_prompt_tokens:
        ]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    prompt_length = input_ids.shape[1]

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    new_tokens = generated[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# SERIALIZATION CHECK
# ============================================================

test_prompt = "What is the capital of France?"

print("=" * 80)
print("STAGE 4 SERIALIZATION CHECK")
print("=" * 80)
print()
print(build_stage4_prefix(test_prompt))
print()


# ============================================================
# SAME TEST SET USED ON A100
# ============================================================

tests = [
    "What is the capital of France?",
    "Explain what a neural network is in two sentences.",
    "Qual é a capital do Brasil?",
    "Explique em poucas palavras o que é uma rede neural.",
    "¿Cuál es la capital de Argentina?",
    "Explica brevemente qué es el aprendizaje automático.",
    "What is 17 + 28?",
    "Write a Python function that returns the square of a number.",
]

print("=" * 80)
print("QUICK TEST")
print("=" * 80)

for question in tests:

    print()
    print("USER:")
    print(question)

    answer = ask(question)

    print()
    print("ASSISTANT:")
    print(answer)

    print()
    print("-" * 80)


# ============================================================
# INTERACTIVE MODE
# ============================================================

print()
print("=" * 80)
print("INTERACTIVE MODE")
print("=" * 80)
print()
print("Type a question and press Enter.")
print("Type 'exit' to finish.")
print()

while True:

    try:
        user_text = input("You: ").strip()

    except (EOFError, KeyboardInterrupt):
        print()
        print("Chat finished.")
        break

    if not user_text:
        continue

    if user_text.lower() in {
        "exit",
        "quit",
        "sair",
    }:
        print("Chat finished.")
        break

    try:
        response = ask(user_text)

        print()
        print("Assistant:")
        print(response)
        print()

    except Exception as exc:
        print()
        print(
            "Generation error:",
            repr(exc),
        )
        print()