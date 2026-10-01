import json,random,torch
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def load_json(p): return json.loads(Path(p).read_text())
def load_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def set_seed(s): random.seed(s);torch.manual_seed(s)
def get_tokenizer(m):
 t=AutoTokenizer.from_pretrained(m)
 if t.pad_token is None:t.pad_token=t.eos_token
 t.padding_side="right";return t
def format_prompt(x): return f"Instruction: {x['instruction']}\nResponse:"
def normalize(s): return " ".join(s.strip().lower().split())
def device():
 if torch.backends.mps.is_available():return torch.device("mps")
 if torch.cuda.is_available():return torch.device("cuda")
 return torch.device("cpu")
