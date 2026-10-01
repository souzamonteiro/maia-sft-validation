import json, math, random, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from src.common import ROOT, MODEL_ID, SEED, get_tokenizer, load_jsonl
from src.dataset import SFTDataset

def validation_loss(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            values.append(model(**batch).loss.item())
    model.train()
    return sum(values) / len(values)

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Selected device:", device)

    cfg = json.loads((ROOT/"configs/stage1.json").read_text())
    tok = get_tokenizer()
    train_rows = load_jsonl(ROOT/"data/stage1/train.jsonl")
    val_rows = load_jsonl(ROOT/"data/stage1/validation.jsonl")
    train_ds = SFTDataset(train_rows, tok, cfg["max_length"])
    val_ds = SFTDataset(val_rows, tok, cfg["max_length"])

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    accum = cfg["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader)/accum)
    total_updates = updates_per_epoch * cfg["epochs"]
    warmup = max(1, int(total_updates * cfg["warmup_ratio"]))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    initial_val = validation_loss(model, val_loader, device)
    print(f"Initial validation loss: {initial_val:.6f}")

    history = []
    update = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        count = 0

        for i, batch in enumerate(train_loader, 1):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss
            (loss / accum).backward()
            running += loss.item()
            count += 1

            if i % accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 10 == 0:
                    print(
                        f"epoch={epoch:02d} update={update:04d}/{total_updates} "
                        f"loss={running/count:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                running = 0.0
                count = 0

        val = validation_loss(model, val_loader, device)
        history.append({"epoch":epoch, "update":update, "validation_loss":val})
        print(f"EPOCH {epoch} validation_loss={val:.6f}")

    out = ROOT/"outputs/gpt2-medium-stage1"
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    metrics = {
        "stage":"stage1-generalization-control-v2",
        "model":MODEL_ID,
        "parameters":sum(p.numel() for p in model.parameters()),
        "train_examples":len(train_rows),
        "validation_examples":len(val_rows),
        "epochs":cfg["epochs"],
        "optimizer_updates":update,
        "initial_validation_loss":initial_val,
        "final_validation_loss":history[-1]["validation_loss"],
        "history":history
    }
    (ROOT/"outputs/stage1-training-metrics.json").write_text(
        json.dumps(metrics, indent=2)+"\n"
    )

if __name__ == "__main__":
    main()
