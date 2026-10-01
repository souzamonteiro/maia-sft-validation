import argparse, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.common import ROOT, MODEL_ID, load_jsonl, prompt_for

def normalize(s):
    return " ".join(s.strip().split()).lower()

@torch.no_grad()
def run(model_path, split, out_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    rows = load_jsonl(ROOT / "data" / "stage1" / f"{split}.jsonl")
    results, exact, contained, eos = [], 0, 0, 0

    for r in rows:
        enc = tok(prompt_for(r["instruction"]), return_tensors="pt").to(device)
        out = model.generate(
            **enc, max_new_tokens=40, do_sample=False,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id
        )
        new = out[0, enc["input_ids"].shape[1]:]
        eos += int(tok.eos_token_id in new.tolist())
        pred = tok.decode(new, skip_special_tokens=True).strip()
        target = r["response"].strip()
        em = normalize(pred) == normalize(target)
        tc = normalize(target) in normalize(pred)
        exact += int(em); contained += int(tc)
        results.append({**r, "prediction":pred, "exact_match":em, "target_contained":tc})

    metrics = {
        "model":str(model_path), "split":split, "examples":len(rows),
        "exact_match":exact, "exact_match_rate":exact/len(rows),
        "target_contained":contained, "target_contained_rate":contained/len(rows),
        "eos_count":eos, "eos_rate":eos/len(rows)
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({"metrics":metrics, "results":results}, indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.model, a.split, a.out)
