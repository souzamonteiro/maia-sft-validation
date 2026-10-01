from transformers import AutoTokenizer

from src.common import MODEL_ID
from src.dataset import encode_response_only


def test_response_only_masking():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    ex = encode_response_only(tokenizer, "Return code MAIA-0007.", "MAIA-0007", 64)
    assert (ex.labels[: ex.prompt_length] == -100).all()
    active = ex.labels[ex.labels != -100]
    assert active[-1].item() == tokenizer.eos_token_id
    decoded = tokenizer.decode(active.tolist(), skip_special_tokens=True)
    assert "MAIA-0007" in decoded
