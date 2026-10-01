import json, math, shutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from src.common import ROOT, load_json, load_jsonl, get_tokenizer, set_seed, device
from src.dataset import ResponseOnlyDataset

cfg=load_json(ROOT/"configs/stage2.json"); set_seed(cfg["seed"]); dev=device()
print("Selected device:",dev)
tok=get_tokenizer(cfg["model_id"])
train_ex=load_jsonl(ROOT/"data/stage2/train.jsonl"); val_ex=load_jsonl(ROOT/"data/stage2/validation.jsonl")
train_ds=ResponseOnlyDataset(train_ex,tok,cfg["max_length"]); val_ds=ResponseOnlyDataset(val_ex,tok,cfg["max_length"])
gen=torch.Generator().manual_seed(cfg["seed"])
loader=DataLoader(train_ds,batch_size=cfg["micro_batch_size"],shuffle=True,generator=gen)
vloader=DataLoader(val_ds,batch_size=1,shuffle=False)
model=AutoModelForCausalLM.from_pretrained(cfg["model_id"]).to(dev)
model.config.use_cache=False
opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
updates_per_epoch=math.ceil(len(loader)/cfg["gradient_accumulation_steps"])
total=updates_per_epoch*cfg["epochs"]; warm=max(1,int(total*cfg["warmup_ratio"]))
sched=get_cosine_schedule_with_warmup(opt,warm,total)

@torch.no_grad()
def val_loss():
    model.eval(); total_loss=0.0; n=0
    for b in vloader:
        b={k:v.to(dev) for k,v in b.items()}
        loss=model(**b).loss
        total_loss+=loss.item(); n+=1
    model.train(); return total_loss/n

initial=val_loss(); print(f"Initial validation loss: {initial:.6f}")
best=float("inf"); history=[]; update=0
best_dir=ROOT/"outputs/gpt2-medium-stage2-best"; final_dir=ROOT/"outputs/gpt2-medium-stage2-final"
opt.zero_grad(set_to_none=True)
for epoch in range(1,cfg["epochs"]+1):
    for step,b in enumerate(loader,1):
        b={k:v.to(dev) for k,v in b.items()}
        loss=model(**b).loss/cfg["gradient_accumulation_steps"]
        loss.backward()
        if step%cfg["gradient_accumulation_steps"]==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); update+=1
            if update==1 or update%10==0:
                print(f"epoch={epoch:02d} update={update:04d}/{total} loss={loss.item()*cfg['gradient_accumulation_steps']:.6f} lr={sched.get_last_lr()[0]:.2e}")
    vl=val_loss(); history.append({"epoch":epoch,"update":update,"validation_loss":vl})
    print(f"EPOCH {epoch} validation_loss={vl:.6f}")
    if vl < best:
        best=vl
        if best_dir.exists(): shutil.rmtree(best_dir)
        model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
        print(f"BEST CHECKPOINT UPDATED: epoch={epoch} validation_loss={vl:.6f}")

if final_dir.exists(): shutil.rmtree(final_dir)
model.save_pretrained(final_dir); tok.save_pretrained(final_dir)
metrics={"initial_validation_loss":initial,"best_validation_loss":best,"history":history,"optimizer_updates":update}
(ROOT/"outputs").mkdir(exist_ok=True)
(ROOT/"outputs/stage2-training-metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
