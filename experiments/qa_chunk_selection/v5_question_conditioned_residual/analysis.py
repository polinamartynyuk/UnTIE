from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from .contract import CELLS
from .io_guard import read_jsonl, write_json


def summary(rows):
    metrics=("token_f1","mrr","hit@1","hit@k","hn@1","hn@k","amb@1","gsrn@1","qa_forwards","extra_ranker_forwards")
    value={name:float(np.mean([float(r.get(name,0) or 0) for r in rows])) for name in metrics}
    value["questions"]=len({r["question_id"] for r in rows}); value["rows"]=len(rows)
    value["routing_rate"]=float(np.mean([bool(r.get("routed")) for r in rows]))
    return value


def paired_bootstrap(base,other,n_boot,seed=42):
    by=defaultdict(lambda:defaultdict(list))
    for label,rows in (("base",base),("other",other)):
        for row in rows:by[(row["document_id"],row["question_id"])][label].append(float(row["token_f1"]))
    docs=defaultdict(list)
    for (doc,qid),parts in by.items():
        if parts["base"] and parts["other"]:docs[doc].append(np.mean(parts["other"])-np.mean(parts["base"]))
    ids=sorted(docs); observed=float(np.mean([x for d in ids for x in docs[d]])) if ids else 0.0
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(n_boot):
        draw=rng.choice(ids,size=len(ids),replace=True) if ids else []
        values=[x for d in draw for x in docs[d]]; samples.append(float(np.mean(values)) if values else 0.0)
    return {"observed_delta":observed,"ci_low":float(np.percentile(samples,2.5)),"ci_high":float(np.percentile(samples,97.5)),
            "bootstrap_unit":"document cluster after seed mean","documents":len(ids)}


def switch_metrics(base,rows):
    index={(r["question_id"],r["seed"]):r for r in base}; counts=defaultdict(int); delta=[]
    for row in rows:
        old=index[row["question_id"],row["seed"]]
        if not row["changed_top1"]:continue
        d=float(row["token_f1"])-float(old["token_f1"]); delta.append(d)
        counts["good_switch" if d>0 else "bad_switch" if d<0 else "neutral_switch"]+=1
        left="positive" if old["selected_tier"]=="sufficient_positive" else "non-positive"
        right="positive" if row["selected_tier"]=="sufficient_positive" else "non-positive"
        counts[f"{left}_to_{right}"]+=1
    return {"changed_top1":sum(counts[x] for x in ("good_switch","bad_switch","neutral_switch")),
            **counts,"mean_f1_delta_changed":float(np.mean(delta)) if delta else 0.0}


def gate_metrics(rows):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y=[]; p=[]
    for row in rows:
        # Evidence target can be reconstructed from B0 and top-K flags saved in prediction.
        y.append(int(not row["hit@1"] and row["hit@k"])); p.append(float(row["gate_probability"]))
    pred=[x>=0.5 for x in p]; positives=sum(y)
    return {"auroc":float(roc_auc_score(y,p)) if len(set(y))>1 else None,
            "pr_auc":float(average_precision_score(y,p)) if positives else None,
            "precision_at_0.5":sum(a and b for a,b in zip(pred,y))/max(sum(pred),1),
            "recall_at_0.5":sum(a and b for a,b in zip(pred,y))/max(positives,1),
            "actionable_positive_prevalence":positives/max(len(y),1)}


def analyze(config,output,progress):
    progress.stage("analysis","metrics, clustered bootstrap, switches and global report")
    primary={}; comparisons={}; switches={}; gates={}; all_rows={}; references={}; ablations={}
    for cell in config.cells:
        systems={}
        for path in sorted((output/cell/"oof").glob("*/predictions.jsonl")):
            systems[path.parent.name]=read_jsonl(path)
        if not systems:raise FileNotFoundError(f"no completed OOF systems for {cell}")
        all_rows[cell]=systems; primary[cell]={name:summary(rows) for name,rows in systems.items()}
        base=systems["B0"]
        comparisons[cell]={name:paired_bootstrap(base,rows,config.n_bootstrap) for name,rows in systems.items() if name!="B0"}
        switches[cell]={name:switch_metrics(base,rows) for name,rows in systems.items() if name!="B0"}
        gates[cell]=gate_metrics(systems["V5-Q-A1"])
        ablations[cell]={"A1_minus_A0":paired_bootstrap(systems["V5-Q-A0"],systems["V5-Q-A1"],config.n_bootstrap),
                         "A2_minus_A1":paired_bootstrap(systems["V5-Q-A1"],systems["V5-Q-A2"],config.n_bootstrap)}
        spec=CELLS[cell]; ref_rows=read_jsonl(spec.historical_reference)
        references[cell]={spec.historical_system:{"token_f1":float(np.mean([float(x.get("token_f1",0) or 0) for x in ref_rows])),"rows":len(ref_rows)}}
        if spec.bge_reference and spec.bge_reference.is_file():
            bge=read_jsonl(spec.bge_reference)
            references[cell]["BGE"]={"token_f1":float(np.mean([float(x.get("token_f1",0) or 0) for x in bge])),"rows":len(bge)}
    write_json(output/"analysis/primary_metrics.json",primary)
    write_json(output/"analysis/paired_bootstrap.json",comparisons)
    write_json(output/"analysis/switch_metrics.json",switches)
    write_json(output/"analysis/gate_metrics.json",gates)
    write_json(output/"analysis/feature_ablation.json",ablations)
    write_json(output/"analysis/historical_references.json",references)
    aspect_metrics={}; seed_stability={}; fold_stability={}; candidate_counts={}; missed={}; budget_curves={}
    for cell,systems in all_rows.items():
        aspect_metrics[cell]={}; seed_stability[cell]={}; fold_stability[cell]={}; candidate_counts[cell]={}
        for name,rows in systems.items():
            aspects=sorted({str(r["aspect"]) for r in rows})
            aspect_metrics[cell][name]={aspect:summary([r for r in rows if str(r["aspect"])==aspect]) for aspect in aspects}
            seed_stability[cell][name]={str(seed):summary([r for r in rows if int(r["seed"])==seed]) for seed in sorted({int(r["seed"]) for r in rows})}
            fold_stability[cell][name]={str(fold):summary([r for r in rows if int(r["outer_fold"])==fold]) for fold in sorted({int(r["outer_fold"]) for r in rows})}
            candidate_counts[cell][name]={str(k):summary([r for r in rows if int(r.get("effective_k",1))==k]) for k in sorted({int(r.get("effective_k",1)) for r in rows})}
        b0=systems["B0"]
        missed[cell]={"count":sum(int(not r["routed"] and not r["hit@1"] and r["hit@k"]) for r in systems["V5-QG-A1-B2.00"]),
                      "topk_positive_recall":float(np.mean([r["hit@k"] for r in b0])),
                      "b0_residual_error_count":sum(int(not r["hit@1"] and r["hit@k"]) for r in b0)}
        budget_curves[cell]=[{"system":name,**summary(rows)} for name,rows in systems.items() if name=="B0" or name=="V5-Q-A1" or name.startswith("V5-QG-A1-")]
    write_json(output/"analysis/aspect_metrics.json",aspect_metrics)
    write_json(output/"analysis/seed_stability.json",seed_stability)
    write_json(output/"analysis/fold_stability.json",fold_stability)
    write_json(output/"analysis/candidate_count_analysis.json",candidate_counts)
    write_json(output/"analysis/gate_missed_opportunities.json",missed)
    write_json(output/"analysis/budget_curves.json",budget_curves)
    write_json(output/"analysis/evidence_transitions.json",switches)
    resource_path=output/"resource_benchmark/summary.json"
    write_json(output/"analysis/resource_tradeoff.json",json.loads(resource_path.read_text()) if resource_path.is_file() else {"status":"unavailable"})
    global_summary={cell:{"corpus":next(iter(systems.values()))[0]["corpus"],
        "b0_f1":primary[cell]["B0"]["token_f1"],
        "v5_q_delta":comparisons[cell]["V5-Q-A1"]["observed_delta"],
        "v5_qg_delta":comparisons[cell]["V5-QG-A1-B2.00"]["observed_delta"],
        "routing_rate":primary[cell]["V5-QG-A1-B2.00"]["routing_rate"],
        "qa_forwards":primary[cell]["V5-QG-A1-B2.00"]["qa_forwards"]} for cell,systems in all_rows.items()}
    write_json(output/"analysis/global_summary.json",global_summary)
    lines=["# V5-A multilingual question-conditioned residual reranking","",
           "Development characterization on reused corpora; SciER 384/50 and 128/24 are two non-independent cells of one corpus.","",
           "| Cell | B0 Token F1 | V5-Q Δ | 95% CI | V5-QG Δ | QA forwards | Routing |","|---|---:|---:|---:|---:|---:|---:|"]
    for cell in config.cells:
        q=comparisons[cell]["V5-Q-A1"]; qg=comparisons[cell]["V5-QG-A1-B2.00"]; m=primary[cell]
        lines.append(f"| {cell} | {m['B0']['token_f1']:.4f} | {q['observed_delta']:+.4f} | [{q['ci_low']:+.4f}, {q['ci_high']:+.4f}] | {qg['observed_delta']:+.4f} | {m['V5-QG-A1-B2.00']['qa_forwards']:.3f} | {m['V5-QG-A1-B2.00']['routing_rate']:.1%} |")
    lines += ["","Primary V5 is A1. A0 is the mandatory mechanism ablation. Gate thresholds were fitted on META OOF from the resource budget only; outer EVAL was not used for tuning."]
    lines += ["", "Historical V4.x/V4.5 systems and RU BGE are preserved in analysis/historical_references.json; they are references rather than retuned V5 candidates."]
    report=output/"reports/results_v5.md"; report.parent.mkdir(parents=True,exist_ok=True); report.write_text("\n".join(lines)+"\n",encoding="utf-8")

