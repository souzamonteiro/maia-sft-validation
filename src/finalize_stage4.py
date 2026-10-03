import json
from src.common_stage4 import *
def main():
    t=json.loads((OUT/"stage4-training-metrics.json").read_text());b=json.loads((OUT/"stage4-baseline-metrics.json").read_text());p=json.loads((OUT/"stage4-post-metrics.json").read_text())
    m={"stage":CFG["stage"],"training":t,"baseline":b,"post":p,"objective_exact_match_gain":p["objective_exact_match_rate"]-b["objective_exact_match_rate"],"eos_gain":p["eos_rate"]-b["eos_rate"],"result":"PASS" if p["objective_exact_match_rate"]>b["objective_exact_match_rate"] else "REVIEW","note":"Review 120 qualitative assistant generations before deployment."};(OUT/"stage4-metrics.json").write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=="__main__":main()
