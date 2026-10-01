from src.common import ROOT,load_jsonl,get_tokenizer
from src.dataset import ResponseOnlyDataset
def test_stage3():
 tr=load_jsonl(ROOT/"data/stage3/train.jsonl");va=load_jsonl(ROOT/"data/stage3/validation.jsonl");tc=load_jsonl(ROOT/"data/stage3/test_core.jsonl");th=load_jsonl(ROOT/"data/stage3/test_heldout.jsonl");assert (len(tr),len(va),len(tc),len(th))==(800,100,100,200);assert {x["variant"] for x in tr}.isdisjoint({x["variant"] for x in th});tok=get_tokenizer("openai-community/gpt2-medium");d=ResponseOnlyDataset(tr[:5],tok,256);assert all((x["labels"]!=-100).sum().item()>0 for x in d)
