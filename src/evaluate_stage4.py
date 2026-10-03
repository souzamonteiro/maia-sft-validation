import argparse,json,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.common_stage4 import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);ap.add_argument("--name",required=True);a=ap.parse_args();dev=device()
    tok=AutoTokenizer.from_pretrained(a.model);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    rows=read_jsonl(DATA/"assistant_benchmark.jsonl");res=[];exact=obj=eos=0
    for i,r in enumerate(rows,1):
        p=f"System: {SYSTEM}\nUser: {r['prompt']}\nAssistant: ";x=tok(p,return_tensors="pt").to(dev)
        with torch.no_grad():y=model.generate(**x,max_new_tokens=96,do_sample=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        new=y[0,x["input_ids"].shape[1]:].tolist();had=tok.eos_token_id in new;eos+=had
        if had:new=new[:new.index(tok.eos_token_id)]
        pred=tok.decode(new,skip_special_tokens=True).strip();ok=None
        if r["expected"] is not None:obj+=1;ok=normalize(pred)==normalize(r["expected"]);exact+=ok
        res.append({**r,"prediction":pred,"exact_match":ok,"eos":bool(had)})
        if i%25==0:print(i,"/",len(rows))
    m={"model":a.model,"examples":len(rows),"objective_examples":obj,"objective_exact_match":exact,"objective_exact_match_rate":exact/max(1,obj),"eos_count":eos,"eos_rate":eos/len(rows)}
    OUT.mkdir(exist_ok=True);(OUT/f"stage4-{a.name}-metrics.json").write_text(json.dumps(m,indent=2));write_jsonl(OUT/f"stage4-{a.name}-results.jsonl",res);print(m)
if __name__=="__main__":main()
