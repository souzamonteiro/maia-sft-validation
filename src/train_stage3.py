import json,math,shutil,torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,get_cosine_schedule_with_warmup
from src.common import ROOT,load_json,load_jsonl,get_tokenizer,set_seed,device
from src.dataset import ResponseOnlyDataset
c=load_json(ROOT/"configs/stage3.json");set_seed(c["seed"]);dev=device();print("Selected device:",dev);tok=get_tokenizer(c["model_id"]);tr=load_jsonl(ROOT/"data/stage3/train.jsonl");va=load_jsonl(ROOT/"data/stage3/validation.jsonl");td=ResponseOnlyDataset(tr,tok,c["max_length"]);vd=ResponseOnlyDataset(va,tok,c["max_length"]);g=torch.Generator().manual_seed(c["seed"]);dl=DataLoader(td,batch_size=1,shuffle=True,generator=g);vl=DataLoader(vd,batch_size=1);m=AutoModelForCausalLM.from_pretrained(c["model_id"]).to(dev);m.config.use_cache=False;o=torch.optim.AdamW(m.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]);upe=math.ceil(len(dl)/c["gradient_accumulation_steps"]);total=upe*c["epochs"];sch=get_cosine_schedule_with_warmup(o,max(1,int(total*c["warmup_ratio"])),total)
@torch.no_grad()
def ev():
 m.eval();s=0
 for b in vl:b={k:v.to(dev) for k,v in b.items()};s+=m(**b).loss.item()
 m.train();return s/len(vl)
initial=ev();print(f"Initial validation loss: {initial:.6f}");best=1e99;hist=[];u=0;bd=ROOT/"outputs/gpt2-medium-stage3-best";fd=ROOT/"outputs/gpt2-medium-stage3-final";o.zero_grad(set_to_none=True)
for ep in range(1,c["epochs"]+1):
 for st,b in enumerate(dl,1):
  b={k:v.to(dev) for k,v in b.items()};raw=m(**b).loss;(raw/c["gradient_accumulation_steps"]).backward()
  if st%c["gradient_accumulation_steps"]==0:
   torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();sch.step();o.zero_grad(set_to_none=True);u+=1
   if u==1 or u%10==0:print(f"epoch={ep:02d} update={u:04d}/{total} loss={raw.item():.6f} lr={sch.get_last_lr()[0]:.2e}")
 v=ev();hist.append({"epoch":ep,"update":u,"validation_loss":v});print(f"EPOCH {ep} validation_loss={v:.6f}")
 if v<best:
  best=v
  if bd.exists():shutil.rmtree(bd)
  m.save_pretrained(bd);tok.save_pretrained(bd);print(f"BEST CHECKPOINT UPDATED: epoch={ep} validation_loss={v:.6f}")
if fd.exists():shutil.rmtree(fd)
m.save_pretrained(fd);tok.save_pretrained(fd);q={"initial_validation_loss":initial,"best_validation_loss":best,"history":hist,"optimizer_updates":u};(ROOT/"outputs/stage3-training-metrics.json").write_text(json.dumps(q,indent=2));print(json.dumps(q,indent=2))
