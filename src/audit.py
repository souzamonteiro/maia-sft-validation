import argparse

from transformers import AutoTokenizer

from .common import DEFAULT_MAX_LENGTH, MODEL_ID, read_jsonl
from .dataset import PROMPT_TEMPLATE, encode_response_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/micro_train.jsonl")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.data)

    print("=" * 80)
    print("RESPONSE-ONLY LABEL AUDIT")
    print("=" * 80)
    for i, row in enumerate(rows[: args.examples]):
        ex = encode_response_only(tokenizer, row["instruction"], row["response"], args.max_length)
        prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"].strip())
        active = ex.labels[ex.labels != -100].tolist()
        ignored = int((ex.labels == -100).sum())
        print(f"\nExample {i}")
        print(f"Prompt tokens: {ex.prompt_length}")
        print(f"Response tokens incl. EOS: {ex.response_length}")
        print(f"Ignored/padded labels: {ignored}/{args.max_length}")
        print("PROMPT:")
        print(prompt)
        print("TARGET RESPONSE:")
        print(tokenizer.decode(active, skip_special_tokens=False))
        assert all(x == -100 for x in ex.labels[: ex.prompt_length].tolist())
        assert active[-1] == tokenizer.eos_token_id
    print("\nAUDIT: PASS")


if __name__ == "__main__":
    main()
