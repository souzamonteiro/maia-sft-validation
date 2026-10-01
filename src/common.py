import json, random, os, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)

def get_tokenizer(model_id):
    tok=AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    tok.padding_side="right"
    return tok

def format_prompt(ex):
    return f"Instruction: {ex['instruction']}\nResponse:"

def normalize(s):
    return " ".join(s.strip().lower().split())

def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")
