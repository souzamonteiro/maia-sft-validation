from pathlib import Path
import json, torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "openai-community/gpt2-medium"
MAX_LENGTH = 256
SEED = 42

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def prompt_for(instruction):
    return f"### User:\n{instruction}\n\n### Assistant:\n"

def get_tokenizer(model_id=MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def encode_response_only(tok, instruction, response, max_length=MAX_LENGTH):
    p = tok.encode(prompt_for(instruction), add_special_tokens=False)
    r = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    ids = p + r
    if len(ids) > max_length:
        raise ValueError(f"Example too long: {len(ids)} > {max_length}")
    pad = max_length - len(ids)
    return {
        "input_ids": torch.tensor(ids + [tok.pad_token_id] * pad, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
        "labels": torch.tensor([-100] * len(p) + r + [-100] * pad, dtype=torch.long),
    }
