"""One-system fresh-process end-to-end benchmark worker."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.qa_chunk_selection.answerability import AnswerabilityModel
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.train import score_fn_for
from experiments.qa_chunk_selection.v3.split import records_for_role
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language

from .contract import CELLS
from .core import feature_rows, gate_row, runtime_choice
from .historical import tier



def evaluation_records(spec, *, run=0, limit=None):
    records=records_from_jsonl(spec.records)
    splits=json.loads(spec.splits.read_text())
    historical={str(row["question_id"]) for row in (json.loads(line) for line in spec.historical_b0.read_text().splitlines() if line.strip())}
    records=[item for item in records if str(item.question_id) in historical]
    selected=list(records_for_role(records,splits,run=run,role="eval"))
    return selected if limit is None else selected[:int(limit)]

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--cell",required=True); p.add_argument("--system",required=True)
    p.add_argument("--run-root",type=Path,required=True); p.add_argument("--device",default="auto"); p.add_argument("--questions",type=int,default=20)
    args=p.parse_args(argv); spec=CELLS[args.cell]
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
    device=choose_device(args.device)[0]; seed_root=args.run_root/args.cell/"outer_fold_0/seed_42"
    try:
        import torch
        cuda_device=torch.device(device) if str(device).startswith("cuda") and torch.cuda.is_available() else None
        if cuda_device is not None:torch.cuda.reset_peak_memory_stats(cuda_device)
    except ImportError:
        torch=None; cuda_device=None
    ranker=AnswerabilityModel(str(seed_root/"eval/outer_model/model"),device,512)
    factory=ModelFactory(PipelineConfig(profile=profile_for_language(spec.language),device=device)); answerer=factory.answerer
    gate=joblib.load(seed_root/"gate/model.joblib"); residual=joblib.load(seed_root/"residual/A1/model.joblib")
    threshold=json.loads((seed_root/"gate/model.json").read_text())["thresholds"]["2.0"]
    qmap=json.loads((seed_root/"eval/question_policy.json").read_text())
    questions=json.loads(spec.questions_q5.read_text()) if spec.questions_q5 else {}
    records=evaluation_records(spec,run=0,limit=args.questions)
    latencies=[]; qa_counts=[]; rank_counts=[]
    for item in records:
        started=time.perf_counter(); runtime=to_runtime(item); scores_list=score_fn_for(ranker)(runtime)
        scores={int(c.chunk_id):float(v) for c,v in zip(runtime.chunks,scores_list)}
        ranked=sorted(scores,key=lambda cid:(-scores[cid],cid)); top=ranked[:min(spec.k,len(ranked))]
        variant=qmap.get(item.aspect); question=item.question if variant is None else questions[item.aspect][int(variant)]
        # Gate is evaluated before QA.
        dummy_answers={cid:"" for cid in top}; dummy_rel={cid:0. for cid in top}; tiers={cid:tier(item.chunk_by_id(cid)) for cid in top}
        pre=feature_rows(item.question_id,item.document_id,item.aspect,42,top,scores,dummy_answers,dummy_rel,tiers,len(item.chunks))
        gate_p=float(gate.predict_proba(pd.DataFrame([gate_row(pre)]))[0,1])
        route=args.system=="V5-Q" or (args.system=="V5-QG" and len(top)>1 and gate_p>=threshold)
        chosen=top[0]; qa_count=1; extra=0
        if route:
            answers={cid:str(answerer(question=question,context=item.chunk_by_id(cid).text).get("answer") or "") for cid in top}
            relevance_values=ranker.predict([(question,answers[cid]) for cid in top]); relevance=dict(zip(top,relevance_values))
            rows=feature_rows(item.question_id,item.document_id,item.aspect,42,top,scores,answers,relevance,tiers,len(item.chunks))
            selected,_=runtime_choice(rows,residual,"A1"); chosen=int(selected["chunk_id"]); qa_count=len(top); extra=len(top)
        else:
            answerer(question=question,context=item.chunk_by_id(chosen).text); extra=0
        if cuda_device is not None:torch.cuda.synchronize(cuda_device)
        latencies.append(time.perf_counter()-started); qa_counts.append(qa_count); rank_counts.append(extra)
    params=sum(p.numel() for p in ranker.model.parameters())+sum(p.numel() for p in factory.qa_model.parameters())
    disk=sum(p.stat().st_size for root in (seed_root/"eval/outer_model/model",spec.qa_model) for p in root.rglob("*") if p.is_file())
    peak_gpu=int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else None
    payload={"cell":args.cell,"system":args.system,"questions":len(records),"loaded_parameter_count":int(params),"weights_disk_bytes":int(disk),
        "peak_gpu_vram_bytes":peak_gpu,"peak_cpu_rss_kb":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "median_latency_seconds":float(np.median(latencies)) if latencies else 0.,"p95_latency_seconds":float(np.percentile(latencies,95)) if latencies else 0.,
        "throughput_questions_per_second":float(len(latencies)/sum(latencies)) if sum(latencies)>0 else 0.,
        "mean_qa_forwards":float(np.mean(qa_counts)) if qa_counts else 0.,"mean_extra_ranker_forwards":float(np.mean(rank_counts)) if rank_counts else 0.,
        "device":device,"fresh_process":True,"cache_used_for_timing":False}
    print(json.dumps(payload,allow_nan=False))


if __name__=="__main__":main()

