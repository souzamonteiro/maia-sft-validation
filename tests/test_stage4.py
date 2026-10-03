from src.common_stage4 import normalize,assistant_segments
def test_normalize():assert normalize(" Hello   WORLD ")=="hello world"
def test_segments():
    m=[{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]
    assert [x[1] for x in assistant_segments(m)]==["a","a2"]
