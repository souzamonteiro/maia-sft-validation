import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import get_device, print_environment
from .dataset import PROMPT_TEMPLATE


def generate(model, tokenizer, instruction, device, max_new_tokens=64):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/gpt2-medium-micro-sft")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    print_environment()
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    print(generate(model, tokenizer, args.prompt, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
