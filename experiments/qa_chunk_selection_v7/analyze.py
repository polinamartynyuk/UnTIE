"""V7 quality, risk, activity, stability and development decision analysis."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.qa_chunk_selection.progress import ProgressLogger
from .contract import FAMILIES
from .io_guard import read_jsonl,write_json,write_text

def _load(root):
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for path in Path(root).glob("predictions/*/*/*/outer_fold_*/seed_*/*.jsonl"):
        mode,scope,cell=path.relative_to(Path(root)/"predictions").parts[:3]
        data[mode][cell][path.stem].extend(read_jsonl(path))
    return data
def _summary(rows):
    mean=lambda ns,key:float(np.mean([r[ns][key] for r in rows])) if rows else 0.0
    conditional=[r["evaluation"]["f1_given_selected_positive"] for r in rows if r["evaluation"].get("f1_given_selected_positive") is not None]
    return {"rows":len(rows),"questions":len({r["runtime"]["question_id"] for r in rows}),"token_f1":mean("evaluation","token_f1"),
      "hit@1":mean("evaluation","hit@1"),"hit@k":mean("evaluation","hit@k"),"hn@1":mean("evaluation","hn@1"),"hn@k":mean("evaluation","hn@k"),
      "f1_given_selected_positive":float(np.mean(conditional)) if conditional else None,"amb@1":mean("evaluation","amb@1"),"gsrn@1":mean("evaluation","gsrn@1"),
      "mrr_selected_first":mean("evaluation","mrr_selected_first"),
      "switch_rate":mean("evaluation","changed_top1"),"qa_forwards":mean("runtime","qa_forwards"),
      "max_qa_forwards":max((r["runtime"]["qa_forwards"] for r in rows),default=1),"q_a_ranker_forwards":mean("runtime","q_a_ranker_forwards")}
def _paired(base,other,bootstrap,seed=42):
    left={(r["runtime"]["question_id"],r["runtime"]["seed"]):r for r in base};per_question=defaultdict(list);question_doc={}
    for r in other:
        key=(r["runtime"]["question_id"],r["runtime"]["seed"])
        if key in left:
            per_question[key[0]].append(r["evaluation"]["token_f1"]-left[key]["evaluation"]["token_f1"]);question_doc[key[0]]=r["runtime"]["canonical_document_id"]
    by_doc=defaultdict(list)
    for qid,values in per_question.items():by_doc[question_doc[qid]].append(float(np.mean(values)))
    docs=sorted(by_doc);observed=float(np.mean([v for d in docs for v in by_doc[d]])) if docs else 0.0;rng=np.random.default_rng(seed);samples=[]
    for _ in range(bootstrap):
        draw=rng.choice(docs,len(docs),replace=True) if docs else []
        samples.append(float(np.mean([v for d in draw for v in by_doc[d]])) if len(draw) else 0.0)
    return {"delta_token_f1":observed,"ci_low":float(np.percentile(samples,2.5)),"ci_high":float(np.percentile(samples,97.5)),
            "cluster_unit":"canonical_document_id","seed_handling":"average within question before bootstrap","documents":len(docs),"questions":len(per_question)}
def _activity(rows):
    byq=defaultdict(list)
    for r in rows:byq[r["runtime"]["question_id"]].append(int(r["evaluation"]["changed_top1"]))
    counts=[sum(x) for x in byq.values()];n=len(counts);threshold=max(3,math.ceil(0.01*n))
    stable=sum(x>=2 for x in counts)
    return {"questions":n,"required_stably_switched":threshold,"stably_switched":stable,"non_degenerate":stable>=threshold,
            "any_seed_switch_rate":sum(x>=1 for x in counts)/n if n else 0,"two_of_three_switch_rate":stable/n if n else 0,
            "all_three_switch_rate":sum(x>=3 for x in counts)/n if n else 0}
def _switch(rows):
    changed=[r for r in rows if r["evaluation"]["changed_top1"]];u=[r["evaluation"]["actual_switch_utility"] for r in changed]
    return {"good":sum(x>0 for x in u),"bad":sum(x<0 for x in u),"neutral":sum(x==0 for x in u),
            "mean_utility":float(np.mean(u)) if u else 0,"median_utility":float(np.median(u)) if u else 0,
            "large_gain":sum(x>=5 for x in u),"large_harm":sum(x<=-5 for x in u)}
def _model_diagnostics(root):
    utility=[];harm=[];admission=[]
    for p in Path(root).glob("models/*/*/outer_fold_*/seed_*/*/selection.json"):
        d=json.loads(p.read_text());s=d["selection"];base={"path":str(p.relative_to(root)),"mode":d["mode"],"scope":d["scope"],"system":d["system"],"fold":d["fold"],"seed":d["seed"]}
        utility.append(base|{"alpha":s["alpha"],"oof_mae":s["ridge_oof_mae"],"diagnostics":s.get("utility_diagnostics",{}),"trials":s["ridge_trials"]})
        harm.append(base|{"C":s["C_harm"],"oof_log_loss":s["harm_oof_log_loss"],"diagnostics":s.get("harm_diagnostics",{}),"trials":s["harm_trials"]})
        admission.append(base|{"tau_admit":s.get("tau_admit"),"fold_policy":s["fold_policy"],"diagnostics":s.get("admission_diagnostics",{}),"selection_source":s["selection_source"]})
    return utility,harm,admission
def _secondary_answer_metrics(data):
    from untie.extraction_metrics import char_f1,rouge_l_f1
    from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import rouge_l_f1 as unicode_rouge
    from experiments.qa_chunk_selection.v4_4_ru_ablation.lemma import RussianLemmatizer
    lemmatizer=RussianLemmatizer(smoke=False);cache={};result={}
    for mode,cells in data.items():
      result[mode]={}
      for cell,systems in cells.items():
       result[mode][cell]={}
       for system,rows in systems.items():
        values=[]
        for row in rows:
            answer=str(row["runtime"].get("final_answer") or "");gold=tuple(str(x) for x in row["evaluation"].get("gold_answers",[]));key=(row["runtime"]["language"],answer,gold)
            if key not in cache:
                cache[key]={"char_f1":max((char_f1(answer,x) for x in gold),default=0.0),"rouge_l_f1":max((rouge_l_f1(answer,x) for x in gold),default=0.0),
                  "unicode_rouge_l_f1":max((unicode_rouge(answer,x) for x in gold),default=0.0),"exact_match":float(any(answer.strip().casefold()==x.strip().casefold() for x in gold)),
                  "lemma_f1":lemmatizer.gold_f1(answer,gold) if row["runtime"]["language"]=="ru" else None}
            values.append(cache[key])
        result[mode][cell][system]={name:(float(np.mean([x[name] for x in values if x[name] is not None])) if any(x[name] is not None for x in values) else None) for name in ("char_f1","rouge_l_f1","unicode_rouge_l_f1","exact_match","lemma_f1")}
        result[mode][cell][system]["bertscore_status"]="not_available_in_historical_contract"
    return result

def _historical_references(root):
    from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
    manifest=json.loads((Path(root)/"manifest.json").read_text());v6=Path(manifest["config"]["v6_run"]);metrics=json.loads((v6/"analysis/primary_metrics.json").read_text())
    refs={}
    for cell in metrics.get("local",{}):
        refs[cell]={"V6_LOCAL_S4":metrics["local"][cell].get("S4"),"V6_LOCAL_S4_lex":metrics["local"][cell].get("S4-lex"),
                    "V6_POOLED_S4_no_aspect":metrics.get("pooled",{}).get(cell,{}).get("S4-no-aspect")}
    bge_root=PROJECT_ROOT/"artifacts/qa_chunk_selection_v4_5_ru/run_20260915_155234"
    for cell in ("russerrc","scimdix"):
        path=bge_root/cell/"oof/S4_bge_b0/metrics.json"
        if path.is_file():
            value=json.loads(path.read_text());refs.setdefault(cell,{})["V4_5_BGE_S4"]={k:value.get(k) for k in ("token_f1","hit@1","hn@1","mrr") if value.get(k) is not None and np.isfinite(value.get(k))}
    return {"selection_role":"frozen references only; excluded from V7 model/policy selection","references":refs}

def _pairlex_audit(root):
    from scipy.stats import spearmanr
    report={}
    cache=Path(root)/"cache/meta_features"
    for cell_dir in sorted(cache.glob("*")):
        if not cell_dir.is_dir():continue
        rows=[]
        for path in cell_dir.glob("outer_fold_*/seed_*/eval_candidate_targets.jsonl"):rows.extend(read_jsonl(path))
        if not rows:continue
        left=np.asarray([r["pairlex"] for r in rows],float);right=np.asarray([r.get("historical_full_lex_norm",0) for r in rows],float)
        corr=float(spearmanr(left,right).statistic) if np.std(left)>1e-12 and np.std(right)>1e-12 else None
        groups=defaultdict(list)
        for r in rows:groups[(r["outer_fold"],r["seed"],r["question_id"])].append(r)
        agreement=[]
        for values in groups.values():
            pair_best=min(values,key=lambda x:(-x["pairlex"],x["b0_rank"]))["chunk_id"]
            full_best=min(values,key=lambda x:(-x.get("historical_full_lex_norm",0),x["b0_rank"]))["chunk_id"]
            agreement.append(pair_best==full_best)
        report[cell_dir.name]={"rows":len(rows),"questions_fold_seed":len(groups),"spearman_correlation":corr,"top_rank_agreement":float(np.mean(agreement)) if agreement else None,
                               "interpretation":"PairLEX is a progressive proxy, not historical full LEX"}
    return report

def _fallback_analysis(data):
    result={}
    for mode,cells in data.items():
      result[mode]={}
      for cell,systems in cells.items():
       result[mode][cell]={}
       for system,rows in systems.items():
        folds={(r["runtime"]["outer_fold"],r["runtime"]["seed"],r["runtime"].get("fold_policy","B0")) for r in rows}
        fallback_folds=sum(policy=="B0_FALLBACK" for _fold,_seed,policy in folds)
        fallback_rows=sum(r["runtime"].get("fold_policy")=="B0_FALLBACK" for r in rows)
        result[mode][cell][system]={"outer_fold_seed_units":len(folds),"fallback_fold_seed_units":fallback_folds,
          "fallback_question_rate":fallback_rows/len(rows) if rows else 0.0,
          "fallback_is_not_success_evidence":True}
    return result

def _risk_diagnostics(data):
    result={}
    for mode,cells in data.items():
      result[mode]={}
      for cell,systems in cells.items():
       result[mode][cell]={}
       for system,rows in systems.items():
        if system=="S0":continue
        accepted=[]
        for row in rows:
            utilities=row["evaluation"].get("candidate_utilities",{})
            for candidate in row["runtime"].get("evaluated_candidates",[]):
                if candidate.get("accepted"):
                    accepted.append({"predicted_mu":candidate["mu"],"predicted_p_harm":candidate["p_harm"],"actual_utility":utilities.get(str(candidate["chunk_id"]),0.0)})
        bins=[]
        for low,high in zip((0,.2,.4,.6,.8),(.2,.4,.6,.8,1.000001)):
            values=[x for x in accepted if low<=x["predicted_p_harm"]<high]
            bins.append({"low":low,"high":high,"n":len(values),"actual_harm_rate":float(np.mean([x["actual_utility"]<0 for x in values])) if values else None,
                         "mean_predicted_harm":float(np.mean([x["predicted_p_harm"] for x in values])) if values else None})
        result[mode][cell][system]={"accepted_switches":len(accepted),"calibration_by_p_harm":bins,"rows":accepted}
    return result

def analyze_run(root,bootstrap=5000,progress=None):
    root=Path(root);progress=progress or ProgressLogger(root/"progress.jsonl");progress.stage("analysis","quality, bootstrap, risk, activity, stability and freeze decision")
    data=_load(root);primary={};paired={};switches={};activity={};fold_stability={};seed_stability={};aspect={};k_analysis={}
    for mode,cells in data.items():
      primary[mode]={};paired[mode]={};switches[mode]={};activity[mode]={};fold_stability[mode]={};seed_stability[mode]={};aspect[mode]={};k_analysis[mode]={}
      for cell,systems in cells.items():
        primary[mode][cell]={s:_summary(rows) for s,rows in systems.items()};paired[mode][cell]={s:_paired(systems["S0"],rows,bootstrap) for s,rows in systems.items() if s!="S0"}
        switches[mode][cell]={s:_switch(rows) for s,rows in systems.items() if s!="S0"};activity[mode][cell]={s:_activity(rows) for s,rows in systems.items() if s!="S0"}
        fold_stability[mode][cell]={s:{str(f):_summary([r for r in rows if r["runtime"]["outer_fold"]==f]) for f in sorted({r["runtime"]["outer_fold"] for r in rows})} for s,rows in systems.items()}
        seed_stability[mode][cell]={s:{str(seed):_summary([r for r in rows if r["runtime"]["seed"]==seed]) for seed in sorted({r["runtime"]["seed"] for r in rows})} for s,rows in systems.items()}
        aspect[mode][cell]={s:{a:_summary([r for r in rows if r["runtime"]["canonical_aspect"]==a]) for a in sorted({r["runtime"]["canonical_aspect"] for r in rows})} for s,rows in systems.items()}
        k_analysis[mode][cell]={s:{str(k):_summary([r for r in rows if r["runtime"]["effective_k"]==k]) for k in sorted({r["runtime"]["effective_k"] for r in rows})} for s,rows in systems.items()}
    for name,value in (("primary_metrics",primary),("bootstrap",paired),("switch_analysis",switches),("activity_analysis",activity),("fold_stability",fold_stability),("seed_stability",seed_stability),("aspect_analysis",aspect),("k_analysis",k_analysis)):write_json(root/"analysis"/f"{name}.json",value)
    u,h,a=_model_diagnostics(root);write_json(root/"analysis/utility_diagnostics.json",u);write_json(root/"analysis/harm_diagnostics.json",h);write_json(root/"analysis/admission_diagnostics.json",a)
    write_json(root/"analysis/risk_diagnostics.json",_risk_diagnostics(data))
    write_json(root/"analysis/fallback_analysis.json",_fallback_analysis(data))
    write_json(root/"analysis/pairlex_audit.json",_pairlex_audit(root))
    historical=_historical_references(root);write_json(root/"analysis/historical_references.json",historical)
    write_json(root/"analysis/secondary_answer_metrics.json",_secondary_answer_metrics(data))
    global_summary={}
    for mode in ("local","pooled"):
      if mode not in paired:continue
      deltas={c:paired[mode][c]["S3"]["delta_token_f1"] for c in paired[mode]};scier_values=[v for c,v in deltas.items() if c.startswith("scier_")]
      scier=float(np.mean(scier_values)) if scier_values else None
      family=[v for c,v in deltas.items() if not c.startswith("scier_")]+([scier] if scier is not None else [])
      evidence=all(100*(primary[mode][c]["S3"]["hit@1"]-primary[mode][c]["S0"]["hit@1"])>=-0.5-1e-12 and 100*(primary[mode][c]["S3"]["hn@1"]-primary[mode][c]["S0"]["hn@1"])<=1+1e-12 for c in deltas)
      nondeg=all(activity[mode][c]["S3"]["non_degenerate"] for c in deltas)
      lcb_development=all(paired[mode][c]["S3"]["ci_low"]>-0.5 for c in deltas);strict=all(paired[mode][c]["S3"]["ci_low"]>0 for c in deltas)
      point_positive=all(v>0 for v in deltas.values());family_delta=float(np.mean(family)) if family else 0.0
      summary={"system":"S3","cell_deltas":deltas,"scier_family_delta":scier,"global_family_equal_delta":family_delta,
        "point_positive_all":point_positive,"lcb_above_minus_0_5_all":lcb_development,"strict_lcb_positive_all":strict,
        "evidence_safe_all":evidence,"non_degenerate_all":nondeg}
      if mode=="local":summary["quality_gate"]=point_positive and lcb_development and evidence and nondeg
      else:summary["pooled_promising"]=family_delta>0 and all(v>=-0.5 for v in deltas.values()) and evidence
      global_summary[mode]=summary
    write_json(root/"analysis/global_summary.json",global_summary)
    status="DEVELOPMENT_QUALITY_PASS_PENDING_RESOURCE" if global_summary.get("local",{}).get("quality_gate") else "V7_UNIVERSAL_CANDIDATE_NOT_SUPPORTED"
    write_json(root/"analysis/development_decision.json",{"primary_mode":"local","primary_system":"S3","status":status,"diagnostics_ineligible":True,"estimate":"DEVELOPMENT"})
    lines=["# V7 / RAPR — development results","","Outer CV is development evidence; S3 LOCAL is the only primary candidate.","","| Mode | Cell | System | Token F1 | Δ vs S0 | 95% CI | QA mean/max | Switch |","|---|---|---|---:|---:|---:|---:|---:|"]
    for mode in sorted(primary):
      for cell in sorted(primary[mode]):
       for system in sorted(primary[mode][cell]):
        m=primary[mode][cell][system];p=paired[mode][cell].get(system,{"delta_token_f1":0,"ci_low":0,"ci_high":0})
        lines.append(f"| {mode} | {cell} | {system} | {m['token_f1']:.4f} | {p['delta_token_f1']:+.4f} | [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] | {m['qa_forwards']:.3f}/{m['max_qa_forwards']} | {m['switch_rate']:.2%} |")
    lines += ["", "## Frozen historical references (not used for V7 selection)", "", "| Cell | Reference | Token F1 |", "|---|---|---:|"]
    for cell,values in sorted(historical["references"].items()):
        for name,value in values.items():
            if value and value.get("token_f1") is not None:lines.append(f"| {cell} | {name} | {value['token_f1']:.4f} |")
    write_text(root/"reports/results_v7.md","\n".join(lines)+"\n");write_text(root/"reports/reproducibility.md","# V7 reproducibility\n\nSee frozen_contract.yaml, manifest.json, audit/, models/, predictions/, analysis/, and resource_benchmark/. Historical V5/V6 inputs are read-only and hash-checked before and after the run.\n")
    progress.log("analysis complete",modes=sorted(primary));return primary

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=5000);a=p.parse_args(argv);analyze_run(a.run_dir,a.bootstrap);return 0
if __name__=="__main__":raise SystemExit(main())
