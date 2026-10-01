from torch.utils.data import Dataset
from .common import encode_response_only

class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = [
            encode_response_only(tokenizer, r["instruction"], r["response"], max_length)
            for r in rows
        ]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]
