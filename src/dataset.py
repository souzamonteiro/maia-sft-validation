from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

PROMPT_TEMPLATE = "### User:\n{instruction}\n\n### Assistant:\n"


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_length: int
    response_length: int


def encode_response_only(tokenizer, instruction: str, response: str, max_length: int) -> EncodedExample:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction.strip())
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]

    if len(prompt_ids) >= max_length:
        raise ValueError("Prompt alone exceeds max_length")

    available = max_length - len(prompt_ids)
    response_ids = response_ids[:available]
    if not response_ids:
        raise ValueError("No room left for response tokens")
    response_ids[-1] = tokenizer.eos_token_id

    ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids.copy()
    attention = [1] * len(ids)

    pad_count = max_length - len(ids)
    ids += [tokenizer.pad_token_id] * pad_count
    labels += [-100] * pad_count
    attention += [0] * pad_count

    return EncodedExample(
        input_ids=torch.tensor(ids, dtype=torch.long),
        attention_mask=torch.tensor(attention, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        prompt_length=len(prompt_ids),
        response_length=len(response_ids),
    )


class ResponseOnlyDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.examples = [
            encode_response_only(tokenizer, row["instruction"], row["response"], max_length)
            for row in rows
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }
