"""One fresh process per V8 cell/system; neural inference is executed, policy rows are frozen."""
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import joblib,numpy as np
from experiments.qa_chunk_selection.answerability import AnswerabilityModel
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.train import score_fn_for
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS
from untie.config import PipelineConfig
from untie.models import ModelFactory,profile_for_language
from .io_guard import read_jsonl

def sync(torch,device):
    if torch is not None and device is not None:torch.cuda.synchronize(device)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--cell",required=True);p.add_argument("--system",required=True);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--v5-run",type=Path,required=True)
    p.add_argument("--device",default="auto");p.add_argument("--sample-manifest",type=Path,required=True);p.add_argument("--warmup",type=int,default=10);p.add_argument("--repeats",type=int,default=3);a=p.parse_args(argv)
    spec=CELLS[a.cell]
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
    device=choose_device(a.device)[0]
    try:
        import torch
        cuda=torch.device(device) if str(device).startswith("cuda") and torch.cuda.is_available() else None
        if cuda is not None:torch.cuda.reset_peak_memory_stats(cuda)
    except ImportError:torch=None;cuda=None
    v5seed=a.v5_run/a.cell/"outer_fold_0/seed_42";ranker=AnswerabilityModel(str(v5seed/"eval/outer_model/model"),device,512);factory=ModelFactory(PipelineConfig(profile=profile_for_language(spec.language),device=device));answerer=factory.answerer
    wanted=json.loads(a.sample_manifest.read_text())["cells"][a.cell]["question_ids"];items={str(x.question_id):x for x in records_from_jsonl(spec.records)};records=[items[q] for q in wanted]
    qmap=json.loads((v5seed/"eval/question_policy.json").read_text());q5=json.loads(spec.questions_q5.read_text()) if spec.questions_q5 else {}
    branch="hrr" if a.system.startswith("H") else "v44_af";policy=None;columns=None
    if a.system not in {"H0","H1","F0"}:
        root=(a.run_root/"branch_hrr/models/local"/a.cell/"outer_fold_0/seed_42"/a.system) if branch=="hrr" else (a.run_root/"branch_v44_af/canonical/models/local"/a.cell/"outer_fold_0/seed_42"/a.system)
        policy={"utility":joblib.load(root/"utility_model.joblib"),"harm":joblib.load(root/"harm_model.joblib")};meta=json.loads((root/"selection.json").read_text());columns=meta["feature_columns"]
    if branch=="hrr":feature_path=a.run_root/"cache/canonical_features"/a.cell/"outer_fold_0/seed_42/eval_hrr_targets.jsonl"
    else:feature_path=a.run_root/"cache/af_features/local"/a.cell/a.cell/"outer_fold_0/seed_42.jsonl"
    cached=read_jsonl(feature_path) if feature_path.is_file() else [];byq={}
    for row in cached:byq.setdefault(str(row["question_id"]),[]).append(row)
    def request(item):
        runtime=to_runtime(item);scores={int(c.chunk_id):float(v) for c,v in zip(runtime.chunks,score_fn_for(ranker)(runtime))};top=sorted(scores,key=lambda cid:(-scores[cid],cid))[:min(spec.k,len(scores))]
        variant=qmap.get(item.aspect);question=item.question if variant is None else q5[item.aspect][int(variant)];ids=top[:1] if a.system=="H0" else top
        answers={cid:str(answerer(question=question,context=item.chunk_by_id(cid).text).get("answer") or "") for cid in ids}
        qarel=a.system not in {"H0","F0"}
        if qarel:ranker.predict([(question,answers[cid]) for cid in ids])
        if policy and byq.get(str(item.question_id)):
            import pandas as pd
            frame=pd.DataFrame(byq[str(item.question_id)])[columns];policy["utility"].predict(frame);policy["harm"].predict_proba(frame)
        return len(ids),len(ids) if qarel else 0
    warm=(records*((a.warmup+len(records)-1)//max(len(records),1)))[:a.warmup]
    for item in warm:request(item)
    lat=[];qa=[];qar=[];reps=[]
    for rep in range(a.repeats):
        current=[]
        for item in records:
            sync(torch,cuda);start=time.perf_counter();q,r=request(item);sync(torch,cuda);elapsed=time.perf_counter()-start;lat.append(elapsed);current.append(elapsed);qa.append(q);qar.append(r)
        reps.append({"repeat":rep,"median_seconds":float(np.median(current)),"questions":len(current)})
    params=sum(x.numel() for x in ranker.model.parameters())+sum(x.numel() for x in factory.qa_model.parameters());disk=sum(x.stat().st_size for root in (v5seed/"eval/outer_model/model",spec.qa_model) for x in root.rglob("*") if x.is_file())
    result={"cell":a.cell,"system":a.system,"branch":branch,"questions":len(records),"warmup":a.warmup,"repeats":a.repeats,"sample_question_ids":wanted,
      "median_latency_seconds":float(np.median(lat)),"p95_latency_seconds":float(np.percentile(lat,95)),"throughput_questions_per_second":float(len(lat)/sum(lat)),
      "mean_qa_forwards":float(np.mean(qa)),"max_qa_forwards":int(max(qa,default=1)),"mean_q_a_ranker_forwards":float(np.mean(qar)),
      "peak_gpu_vram_bytes":int(torch.cuda.max_memory_allocated(cuda)) if cuda is not None else None,"peak_cpu_rss_kb":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
      "loaded_parameter_count":int(params),"weights_disk_bytes":int(disk),"device":device,"fresh_process":True,"warmup_excluded":True,
      "neural_inference_fresh":True,"policy_feature_values_from_frozen_canonical_cache":bool(policy),"repeat_summary":reps}
    print(json.dumps(result,allow_nan=False))
if __name__=="__main__":main()
