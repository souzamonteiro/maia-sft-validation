from collections import Counter
from src.common import ROOT, get_tokenizer, load_jsonl, encode_response_only

EXPECTED = {"train": 800, "validation": 100, "test": 100}

def main():
    tok = get_tokenizer()
    print("=" * 80)
    print("STAGE 1 V2 - DATASET / RESPONSE-ONLY / LEAKAGE AUDIT")
    print("=" * 80)

    splits = {}
    for name, expected in EXPECTED.items():
        rows = load_jsonl(ROOT / "data" / "stage1" / f"{name}.jsonl")
        assert len(rows) == expected
        assert len({r["id"] for r in rows}) == len(rows)
        assert len({r["instruction"] for r in rows}) == len(rows)
        splits[name] = rows
        print(f"{name:10s}: {len(rows):4d} | categories={dict(Counter(r['category'] for r in rows))}")
        for r in rows:
            item = encode_response_only(tok, r["instruction"], r["response"])
            assert int((item["labels"] != -100).sum()) > 0

    for a, b in [("train","validation"),("train","test"),("validation","test")]:
        ids_a = {r["id"] for r in splits[a]}
        ids_b = {r["id"] for r in splits[b]}
        ins_a = {r["instruction"] for r in splits[a]}
        ins_b = {r["instruction"] for r in splits[b]}
        pairs_a = {(r["instruction"], r["response"]) for r in splits[a]}
        pairs_b = {(r["instruction"], r["response"]) for r in splits[b]}
        assert ids_a.isdisjoint(ids_b)
        assert ins_a.isdisjoint(ins_b)
        assert pairs_a.isdisjoint(pairs_b)
        print(f"{a} <-> {b}: ID=0, instruction=0, full-pair=0")

    print("Response-only labels: PASS")
    print("Leakage audit:        PASS")
    print("AUDIT:                PASS")

if __name__ == "__main__":
    main()
