import torch
from torch.utils.data import Dataset
from .common import format_prompt

class ResponseOnlyDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples=examples; self.tok=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex=self.examples[idx]
        prompt=format_prompt(ex)
        p=self.tok(prompt, add_special_tokens=False)["input_ids"]
        r=self.tok(" "+ex["response"]+self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids=(p+r)[:self.max_length]
        labels=([-100]*len(p)+r)[:self.max_length]
        attn=[1]*len(ids)
        pad=self.max_length-len(ids)
        ids += [self.tok.pad_token_id]*pad
        labels += [-100]*pad
        attn += [0]*pad
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(attn),
                "labels":torch.tensor(labels)}
