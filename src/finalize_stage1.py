import json
from src.common import ROOT

training = json.loads((ROOT/"outputs/stage1-training-metrics.json").read_text())
baseline = json.loads((ROOT/"outputs/stage1-baseline-test.json").read_text())["metrics"]
post = json.loads((ROOT/"outputs/stage1-post-test.json").read_text())["metrics"]

metrics = {
    **training,
    "baseline_test":baseline,
    "post_sft_test":post,
    "exact_match_gain":post["exact_match_rate"] - baseline["exact_match_rate"],
    "target_contained_gain":post["target_contained_rate"] - baseline["target_contained_rate"],
    "result":"PASS" if post["exact_match_rate"] > baseline["exact_match_rate"] else "REVIEW"
}
(ROOT/"outputs/stage1-metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print(json.dumps(metrics, indent=2))
