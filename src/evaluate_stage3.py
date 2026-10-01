import argparse,json,torch
from transformers import AutoModelForCausalLM
from src.common import ROOT,load_jsonl,get_tokenizer,format_prompt,normalize,device
p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--split",required=True);p.add_argument("--output",required=True);a=p.parse_args()
data=load_jsonl(ROOT/f"data/stage3/{a.split}.jsonl");tok=get_tokenizer(a.model);dev=device();m=AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval();res=[]
for x in data:
 e=tok(format_prompt(x),return_tensors="pt").to(dev)
 with torch.no_grad():o=m.generate(**e,max_new_tokens=24,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
 g=o[0,e["input_ids"].shape[1]:];pred=tok.decode(g,skip_special_tokens=True).strip();res.append({**x,"prediction":pred,"exact_match":normalize(pred)==normalize(x["response"]),"eos":tok.eos_token_id in g.tolist()})
def ag(q):
 n=len(q);return {"examples":n,"exact_match":sum(x["exact_match"] for x in q),"exact_match_rate":sum(x["exact_match"] for x in q)/n,"eos_count":sum(x["eos"] for x in q),"eos_rate":sum(x["eos"] for x in q)/n}
met={"model":a.model,"split":a.split,"overall":ag(res),"by_family":{},"by_variant":{}}
for f in sorted({x["family"] for x in res}):met["by_family"][f]=ag([x for x in res if x["family"]==f])
for v in sorted({x["variant"] for x in res}):met["by_variant"][v]=ag([x for x in res if x["variant"]==v])
(ROOT/a.output).write_text(json.dumps({"metrics":met,"results":res},indent=2));print(json.dumps(met,indent=2))
