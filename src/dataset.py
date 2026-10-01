import torch
from torch.utils.data import Dataset
from .common import format_prompt
class ResponseOnlyDataset(Dataset):
 def __init__(self,x,tok,maxlen):self.x=x;self.t=tok;self.m=maxlen
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  x=self.x[i];p=self.t(format_prompt(x),add_special_tokens=False)["input_ids"];r=self.t(" "+x["response"]+self.t.eos_token,add_special_tokens=False)["input_ids"]
  ids=(p+r)[:self.m];lab=([-100]*len(p)+r)[:self.m];att=[1]*len(ids);pad=self.m-len(ids)
  ids += [self.t.pad_token_id]*pad;lab += [-100]*pad;att += [0]*pad
  return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(att),"labels":torch.tensor(lab)}
