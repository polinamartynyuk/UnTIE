"""Fresh-process neural timing for one frozen V7 LOCAL policy."""
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
    v5seed=a.v5_run/a.cell/"outer_fold_0/seed_42";ranker=AnswerabilityModel(str(v5seed/"eval/outer_model/model"),device,512)
    factory=ModelFactory(PipelineConfig(profile=profile_for_language(spec.language),device=device));answerer=factory.answerer
    pred_path=a.run_root/"predictions/local"/a.cell/a.cell/"outer_fold_0/seed_42"/f"{a.system}.jsonl";predictions={r["runtime"]["question_id"]:r["runtime"] for r in read_jsonl(pred_path)}
    policy={}
    if a.system!="S0":
        model_root=a.run_root/"models/local"/a.cell/"outer_fold_0/seed_42"/a.system
        for name in ("utility","harm","admission"):policy[name]=joblib.load(model_root/f"{name}_model.joblib")
        policy["selection"]=json.loads((model_root/"selection.json").read_text())["selection"]
    wanted=json.loads(a.sample_manifest.read_text())["cells"][a.cell]["question_ids"]
    items={str(x.question_id):x for x in records_from_jsonl(spec.records)};records=[items[q] for q in wanted]
    qmap=json.loads((v5seed/"eval/question_policy.json").read_text());q5=json.loads(spec.questions_q5.read_text()) if spec.questions_q5 else {}
    def request(item):
        runtime=to_runtime(item);scores={int(c.chunk_id):float(v) for c,v in zip(runtime.chunks,score_fn_for(ranker)(runtime))};top=sorted(scores,key=lambda cid:(-scores[cid],cid))[:min(spec.k,len(scores))]
        variant=qmap.get(item.aspect);question=item.question if variant is None else q5[item.aspect][int(variant)];frozen=predictions[str(item.question_id)]
        eval_ids=[] if a.system=="S0" else [int(x["chunk_id"]) for x in frozen["evaluated_candidates"]];ids=[top[0],*eval_ids]
        answers={cid:str(answerer(question=question,context=item.chunk_by_id(cid).text).get("answer") or "") for cid in ids}
        if a.system!="S0":
            ranker.predict([(question,answers[cid]) for cid in ids])
            # Include actual sklearn inference in the timing path; neural features are frozen from the exact fold policy.
            cached=[]
            for row in frozen["evaluated_candidates"]:
                cached.append({"language":frozen["language"],"delta_b0_norm":0.0,"delta_qarel_raw":row["q_a_delta"],"pairlex":row["pairlex"],"candidate_rank_norm":0.0,
                 "effective_k":frozen["effective_k"],"gap12_norm":0.0,"b0_entropy_norm":0.0,"answer_valid_i":int(bool(row["answer"])),"answer_valid_0":int(bool(frozen["baseline_answer"])),
                 "both_empty":int(not row["answer"] and not frozen["baseline_answer"]),"qarel_x_margin":0.0,"pairlex_x_margin":0.0,"qarel_x_pairlex":row["q_a_delta"]*row["pairlex"],"b0_x_margin":0.0,
                 "b0_score_norm":0.0,"candidate_count":len(runtime.chunks),"gap13_norm":0.0,"top1_z":0.0})
            if cached:
                import pandas as pd
                frame=pd.DataFrame(cached);policy["utility"].predict(frame);policy["harm"].predict_proba(frame);policy["admission"].predict_proba(frame)
        return len(ids),0 if a.system=="S0" else len(ids)
    warm=(records*((a.warmup+len(records)-1)//max(len(records),1)))[:a.warmup]
    for item in warm:request(item)
    lat=[];qa=[];qar=[];repeat_summary=[]
    for repeat in range(a.repeats):
        current=[]
        for item in records:
            sync(torch,cuda);start=time.perf_counter();q,r=request(item);sync(torch,cuda);elapsed=time.perf_counter()-start;lat.append(elapsed);current.append(elapsed);qa.append(q);qar.append(r)
        repeat_summary.append({"repeat":repeat,"median_seconds":float(np.median(current)),"questions":len(current)})
    if a.system=="S3" and max(qa,default=1)>3:raise RuntimeError("IMPLEMENTATION BUG: S3 exceeded three QA forwards")
    params=sum(x.numel() for x in ranker.model.parameters())+sum(x.numel() for x in factory.qa_model.parameters());disk=sum(x.stat().st_size for root in (v5seed/"eval/outer_model/model",spec.qa_model) for x in root.rglob("*") if x.is_file())
    selection=policy.get("selection",{})
    result={"cell":a.cell,"system":a.system,"mode":"local","fold":0,"seed":42,"questions":len(records),"warmup":a.warmup,"repeats":a.repeats,"sample_question_ids":wanted,
      "alpha":selection.get("alpha"),"C_harm":selection.get("C_harm"),"tau_admit":selection.get("tau_admit"),"tau_utility":selection.get("tau_utility"),"tau_harm":selection.get("tau_harm"),
      "median_latency_seconds":float(np.median(lat)),"p95_latency_seconds":float(np.percentile(lat,95)),"p99_latency_seconds":float(np.percentile(lat,99)),"throughput_questions_per_second":float(len(lat)/sum(lat)),
      "mean_qa_forwards":float(np.mean(qa)),"max_qa_forwards":int(max(qa,default=1)),"mean_q_a_ranker_forwards":float(np.mean(qar)),"peak_gpu_vram_bytes":int(torch.cuda.max_memory_allocated(cuda)) if cuda is not None else None,
      "peak_cpu_rss_kb":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"loaded_parameter_count":int(params),"weights_disk_bytes":int(disk),"device":device,"fresh_process":True,"warmup_excluded":True,"cache_used_for_neural_inference":False,
      "policy_replayed_from_frozen_prediction_manifest":True,"repeat_summary":repeat_summary}
    print(json.dumps(result,allow_nan=False))
if __name__=="__main__":main()
