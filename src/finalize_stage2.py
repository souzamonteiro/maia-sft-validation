import json
from src.common import ROOT, load_json
cfg=load_json(ROOT/"configs/stage2.json")
train=load_json(ROOT/"outputs/stage2-training-metrics.json")
base=load_json(ROOT/"outputs/stage2-baseline-test.json")["metrics"]
post=load_json(ROOT/"outputs/stage2-post-test.json")["metrics"]
metrics={
 "stage":cfg["stage"],"model":cfg["model_id"],"train_examples":1200,"validation_examples":150,
 "test_examples":150,"epochs":cfg["epochs"],"optimizer_updates":train["optimizer_updates"],
 "initial_validation_loss":train["initial_validation_loss"],"best_validation_loss":train["best_validation_loss"],
 "history":train["history"],"baseline_test":base,"post_sft_test":post,
 "exact_match_gain":post["overall"]["exact_match_rate"]-base["overall"]["exact_match_rate"],
 "result":"PASS" if post["overall"]["exact_match_rate"]>base["overall"]["exact_match_rate"] else "FAIL"
}
(ROOT/"outputs/stage2-metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(metrics,ensure_ascii=False,indent=2))
