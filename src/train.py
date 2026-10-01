import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import DEFAULT_MAX_LENGTH, DEFAULT_SEED, MODEL_ID, get_device, print_environment, read_jsonl, set_seed
from .dataset import ResponseOnlyDataset


def main():
    parser = argparse.ArgumentParser(description="Controlled full-parameter GPT-2 Medium SFT sanity test")
    parser.add_argument("--data", default="data/micro_train.jsonl")
    parser.add_argument("--output", default="outputs/gpt2-medium-micro-sft")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    print_environment()
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    model.train()

    rows = read_jsonl(args.data)
    dataset = ResponseOnlyDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    first_loss = None
    last_loss = None
    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    print(f"Examples: {len(dataset)}")
    print(f"Sequence length: {args.max_length}")
    print(f"Optimizer updates planned: {total_updates}")

    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for micro_step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            if not torch.isfinite(raw_loss):
                raise RuntimeError(f"Non-finite loss: {raw_loss.item()}")
            (raw_loss / args.grad_accum).backward()
            running += raw_loss.detach().float().item()
            count += 1

            should_update = micro_step % args.grad_accum == 0 or micro_step == len(loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = running / count
                if first_loss is None:
                    first_loss = avg_loss
                last_loss = avg_loss
                if global_step == 1 or global_step % args.log_every == 0:
                    print(f"epoch={epoch + 1:02d} update={global_step:04d} loss={avg_loss:.6f}")
                running = 0.0
                count = 0

        if device.type == "mps":
            torch.mps.empty_cache()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Initial update loss: {first_loss:.6f}")
    print(f"Final update loss:   {last_loss:.6f}")
    print(f"Loss ratio:          {last_loss / first_loss:.4f}")
    print(f"Saved to:            {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
