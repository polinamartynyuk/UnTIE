"""Fresh-process warm-model resource benchmark for one V6 cell/system."""
from __future__ import annotations
import argparse, json, resource, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from experiments.qa_chunk_selection.answerability import AnswerabilityModel
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v3.train import score_fn_for
from experiments.qa_chunk_selection.v4.answers import answers_match, pair_token_f1
from experiments.qa_chunk_selection.v4.clusters import shortlist_source_clusters
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language


def synchronize(torch, device):
    if torch is not None and device is not None: torch.cuda.synchronize(device)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--cell',required=True); p.add_argument('--system',required=True)
    p.add_argument('--mode',required=True); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--v5-run',type=Path,required=True)
    p.add_argument('--device',default='auto'); p.add_argument('--questions',type=int,default=100); p.add_argument('--warmup',type=int,default=10); p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args(argv); spec=CELLS[a.cell]
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
    device=choose_device(a.device)[0]
    try:
        import torch
        cuda=torch.device(device) if str(device).startswith('cuda') and torch.cuda.is_available() else None
        if cuda is not None: torch.cuda.reset_peak_memory_stats(cuda)
    except ImportError: torch=None; cuda=None
    v5seed=a.v5_run/a.cell/'outer_fold_0/seed_42'
    ranker=AnswerabilityModel(str(v5seed/'eval/outer_model/model'),device,512)
    factory=ModelFactory(PipelineConfig(profile=profile_for_language(spec.language),device=device)); answerer=factory.answerer
    scope='shared' if a.mode=='pooled' else a.cell
    pred_path=a.run_root/'predictions'/a.mode/scope/a.cell/'outer_fold_0/seed_42'/f'{a.system}.jsonl'
    predictions={r['runtime']['question_id']:r['runtime'] for r in (json.loads(x) for x in pred_path.read_text().splitlines() if x.strip())}
    utility=None; gate=None
    if a.system!='S0':
        model_root=a.run_root/'meta_models'/a.mode/scope/'outer_fold_0/seed_42'/a.system
        utility=joblib.load(model_root/'utility_model.joblib')
        if (model_root/'gate_model.joblib').is_file(): gate=joblib.load(model_root/'gate_model.joblib')
    cached_alt={}
    alt_path=a.run_root/'cache/meta_features'/a.cell/'outer_fold_0/seed_42/eval_utility_targets.jsonl'
    if alt_path.is_file():
        for row in (json.loads(x) for x in alt_path.read_text().splitlines() if x.strip()): cached_alt.setdefault(row['question_id'],[]).append(row)
    cached_gate={}
    gate_path=a.run_root/'cache/meta_features'/a.cell/'outer_fold_0/seed_42/eval_gate_targets.jsonl'
    if gate_path.is_file(): cached_gate={r['question_id']:r for r in (json.loads(x) for x in gate_path.read_text().splitlines() if x.strip())}
    splits=json.loads(spec.splits.read_text()); historical=set(predictions)
    records=[r for r in records_from_jsonl(spec.records) if str(r.question_id) in historical]
    records=list(records_for_role(records,splits,run=0,role='eval'))
    records=sorted(records,key=lambda x:str(x.question_id))[:min(a.questions,len(records))]
    qmap=json.loads((v5seed/'eval/question_policy.json').read_text()); q5=json.loads(spec.questions_q5.read_text()) if spec.questions_q5 else {}

    def request(item,timed=True):
        runtime=to_runtime(item); values=score_fn_for(ranker)(runtime); scores={int(c.chunk_id):float(v) for c,v in zip(runtime.chunks,values)}
        top=sorted(scores,key=lambda cid:(-scores[cid],cid))[:min(spec.k,len(scores))]
        variant=qmap.get(item.aspect); question=item.question if variant is None else q5[item.aspect][int(variant)]
        route=a.system=='S2' or bool(predictions[str(item.question_id)].get('gate_decision'))
        if gate is not None and str(item.question_id) in cached_gate: gate.predict_proba(pd.DataFrame([cached_gate[str(item.question_id)]]))
        qa_count=1; qar_count=0
        if route:
            answers={cid:str(answerer(question=question,context=item.chunk_by_id(cid).text).get('answer') or '') for cid in top}
            ranker.predict([(question,answers[cid]) for cid in top]); qa_count=len(top); qar_count=len(top)
            if a.system=='S4':
                for cid in top:
                    others=[answers[x] for x in top if x!=cid]
                    _=[pair_token_f1(answers[cid],other) for other in others]
                    same=[x for x in top if answers_match(answers[cid],answers[x])]
                    shortlist_source_clusters(item,same or [cid],min_overlap=.6)
            if utility is not None and str(item.question_id) in cached_alt: utility.predict_proba(pd.DataFrame(cached_alt[str(item.question_id)]))
        else: answerer(question=question,context=item.chunk_by_id(top[0]).text)
        return qa_count,qar_count

    for item in (records*((a.warmup+len(records)-1)//max(len(records),1)))[:a.warmup]: request(item,False)
    latencies=[]; qa=[]; qar=[]; repeat_summary=[]
    for repeat in range(a.repeats):
        current=[]
        for item in records:
            synchronize(torch,cuda); started=time.perf_counter(); q,r=request(item); synchronize(torch,cuda)
            value=time.perf_counter()-started; latencies.append(value); current.append(value); qa.append(q); qar.append(r)
        repeat_summary.append({'repeat':repeat,'median_seconds':float(np.median(current)),'questions':len(current)})
    params=sum(x.numel() for x in ranker.model.parameters())+sum(x.numel() for x in factory.qa_model.parameters())
    disk=sum(x.stat().st_size for root in (v5seed/'eval/outer_model/model',spec.qa_model) for x in root.rglob('*') if x.is_file())
    result={'cell':a.cell,'system':a.system,'mode':a.mode,'questions':len(records),'warmup':a.warmup,'repeats':a.repeats,
            'sample_question_ids':[str(x.question_id) for x in records], 'median_latency_seconds':float(np.median(latencies)),
            'p95_latency_seconds':float(np.percentile(latencies,95)),'throughput_questions_per_second':float(len(latencies)/sum(latencies)),
            'mean_qa_forwards':float(np.mean(qa)),'mean_q_a_ranker_forwards':float(np.mean(qar)),'mean_semantic_forwards':0.0,
            'peak_gpu_vram_bytes':int(torch.cuda.max_memory_allocated(cuda)) if cuda is not None else None,
            'peak_cpu_rss_kb':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'loaded_parameter_count':int(params),
            'weights_disk_bytes':int(disk),'device':device,'fresh_process':True,'warmup_excluded':True,
            'cache_used_for_neural_inference':False,'repeat_summary':repeat_summary}
    print(json.dumps(result,allow_nan=False))
if __name__=='__main__': main()
