from src.common import get_tokenizer, encode_response_only

def test_response_only_masking():
    tok = get_tokenizer()
    item = encode_response_only(tok, "Compute exactly: 2 + 2.", "4.")
    labels = item["labels"].tolist()
    supervised = [x for x in labels if x != -100]
    assert supervised
    assert supervised[-1] == tok.eos_token_id
    first = next(i for i,x in enumerate(labels) if x != -100)
    assert all(x == -100 for x in labels[:first])
