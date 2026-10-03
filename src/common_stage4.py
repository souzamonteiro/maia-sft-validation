import hashlib,json,random,re
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; CFG=json.loads((ROOT/"configs/stage4.json").read_text()); DATA=ROOT/"data/stage4"; OUT=ROOT/"outputs"; SEED=CFG["seed"]
SYSTEM="You are a helpful, careful multilingual assistant. Answer in the user's language unless asked otherwise."
def seed_all(): random.seed(SEED); torch.manual_seed(SEED)
def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def read_jsonl(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write_jsonl(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
def normalize(s): return re.sub(r"\s+"," ",s.strip()).casefold()
def render_prefix(ms): return "".join(f"{m['role'].capitalize()}: {m['content'].strip()}\n" for m in ms)
def assistant_segments(ms):
    out=[]; hist=[]
    for m in ms:
        if m["role"]=="assistant": out.append((render_prefix(hist)+"Assistant: ",m["content"].strip()))
        hist.append(m)
    return out
