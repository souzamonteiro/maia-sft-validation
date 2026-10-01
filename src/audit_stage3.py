from collections import Counter
from src.common import ROOT,load_json,load_jsonl,get_tokenizer
from src.dataset import ResponseOnlyDataset
c=load_json(ROOT/"configs/stage3.json");ns=("train","validation","test_core","test_heldout");s={n:load_jsonl(ROOT/f"data/stage3/{n}.jsonl") for n in ns};exp={"train":800,"validation":100,"test_core":100,"test_heldout":200}
print("="*80);print("STAGE 3 - ROBUST GENERALIZATION AUDIT");print("="*80)
for n,a in s.items():
 print(n,len(a),dict(Counter(x["family"] for x in a)),dict(Counter(x["variant"] for x in a)));assert len(a)==exp[n];assert len({x["id"] for x in a})==len(a);assert len({x["instruction"] for x in a})==len(a)
for i,a in enumerate(ns):
 for b in ns[i+1:]:
  assert not ({x["id"] for x in s[a]}&{x["id"] for x in s[b]});assert not ({x["instruction"] for x in s[a]}&{x["instruction"] for x in s[b]})
assert {x["variant"] for x in s["train"]}.isdisjoint({x["variant"] for x in s["test_heldout"]})
for n in ns:
 q=[x for x in s[n] if x["family"]=="comparison"];z=Counter(x["response"] for x in q);assert abs(z["true."]-z["false."])<=1
t=get_tokenizer(c["model_id"]);d=ResponseOnlyDataset(s["train"],t,c["max_length"])
for i in range(100):assert (d[i]["labels"]!=-100).sum().item()>0 and (d[i]["labels"]==-100).sum().item()>0
print("Held-out variant separation: PASS");print("Response-only labels: PASS");print("Leakage audit: PASS");print("AUDIT: PASS")
