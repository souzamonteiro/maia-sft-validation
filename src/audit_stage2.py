from collections import Counter
from src.common import ROOT, load_json, load_jsonl, get_tokenizer
from src.dataset import ResponseOnlyDataset

cfg=load_json(ROOT/"configs/stage2.json")
splits={s:load_jsonl(ROOT/f"data/stage2/{s}.jsonl") for s in ("train","validation","test")}
print("="*80); print("STAGE 2 - MULTILINGUAL DATA / LEAKAGE / MASK AUDIT"); print("="*80)
expected={"train":1200,"validation":150,"test":150}
for s,arr in splits.items():
    print(f"{s:10}: {len(arr):4d} | languages={dict(Counter(x['language'] for x in arr))} | families={dict(Counter(x['family'] for x in arr))}")
    assert len(arr)==expected[s]
    assert len({x["id"] for x in arr})==len(arr)
    assert len({x["instruction"] for x in arr})==len(arr)

for a,b in (("train","validation"),("train","test"),("validation","test")):
    ids={x["id"] for x in splits[a]} & {x["id"] for x in splits[b]}
    ins={x["instruction"] for x in splits[a]} & {x["instruction"] for x in splits[b]}
    pairs={(x["instruction"],x["response"]) for x in splits[a]} & {(x["instruction"],x["response"]) for x in splits[b]}
    print(f"{a} <-> {b}: ID={len(ids)}, instruction={len(ins)}, full-pair={len(pairs)}")
    assert not ids and not ins and not pairs

tok=get_tokenizer(cfg["model_id"])
ds=ResponseOnlyDataset(splits["train"],tok,cfg["max_length"])
for i in range(min(100,len(ds))):
    item=ds[i]
    supervised=(item["labels"]!=-100).sum().item()
    assert supervised>0
print("Response-only labels: PASS")
print("Leakage audit:        PASS")
print("AUDIT:                PASS")
