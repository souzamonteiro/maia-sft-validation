from src.common import ROOT, load_jsonl, get_tokenizer
from src.dataset import ResponseOnlyDataset
def test_stage2_masking():
    xs=load_jsonl(ROOT/"data/stage2/train.jsonl")
    tok=get_tokenizer("openai-community/gpt2-medium")
    ds=ResponseOnlyDataset(xs[:5],tok,256)
    for x in ds:
        assert (x["labels"]!=-100).sum().item()>0
        assert (x["labels"]==-100).sum().item()>0
