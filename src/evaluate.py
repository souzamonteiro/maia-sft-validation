import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import get_device, read_jsonl
from .generate import generate


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/gpt2-medium-micro-sft")
    parser.add_argument("--data", default="data/micro_train.jsonl")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    rows = read_jsonl(args.data)[: args.limit]

    exact = 0
    contains = 0
    results = []
    for row in rows:
        prediction = generate(model, tokenizer, row["instruction"], device)
        target = row["response"].strip()
        p = normalize(prediction)
        t = normalize(target)
        exact += int(p == t)
        contains += int(t in p)
        results.append({"instruction": row["instruction"], "target": target, "prediction": prediction})

    n = len(rows)
    print(json.dumps({
        "examples": n,
        "exact_match": exact,
        "exact_match_rate": exact / n,
        "target_contained": contains,
        "target_contained_rate": contains / n,
    }, indent=2))
    print("\nFirst five generations:")
    for row in results[:5]:
        print("-" * 80)
        print("Instruction:", row["instruction"])
        print("Target:     ", row["target"])
        print("Prediction: ", row["prediction"])


if __name__ == "__main__":
    main()
