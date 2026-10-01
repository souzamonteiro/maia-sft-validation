import argparse, json
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM
from src.common import ROOT, load_json, load_jsonl, get_tokenizer, format_prompt, normalize, device

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--output",required=True)
args=ap.parse_args()
cfg=load_json(ROOT/"configs/stage2.json")
data=load_jsonl(ROOT/"data/stage2/test.jsonl")
tok=get_tokenizer(args.model)
model=AutoModelForCausalLM.from_pretrained(args.model).to(device()).eval()
results=[]
for ex in data:
    prompt=format_prompt(ex)
    enc=tok(prompt,return_tensors="pt").to(device())
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=24,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    gen=out[0,enc["input_ids"].shape[1]:]
    eos=tok.eos_token_id in gen.tolist()
    pred=tok.decode(gen,skip_special_tokens=True).strip()
    exact=normalize(pred)==normalize(ex["response"])
    contained=normalize(ex["response"]) in normalize(pred)
    results.append({**ex,"prediction":pred,"exact_match":exact,"target_contained":contained,"eos":eos})

def agg(items):
    n=len(items)
    return {"examples":n,"exact_match":sum(x["exact_match"] for x in items),
            "exact_match_rate":sum(x["exact_match"] for x in items)/n,
            "target_contained":sum(x["target_contained"] for x in items),
            "target_contained_rate":sum(x["target_contained"] for x in items)/n,
            "eos_count":sum(x["eos"] for x in items),
            "eos_rate":sum(x["eos"] for x in items)/n}
metrics={"model":args.model,"split":"test","overall":agg(results),"by_language":{},"by_family":{}}
for lang in ("en","pt","es"):
    metrics["by_language"][lang]=agg([x for x in results if x["language"]==lang])
for fam in sorted({x["family"] for x in results}):
    metrics["by_family"][fam]=agg([x for x in results if x["family"]==fam])
payload={"metrics":metrics,"results":results}
(ROOT/args.output).parent.mkdir(parents=True,exist_ok=True)
(ROOT/args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metrics,ensure_ascii=False,indent=2))
