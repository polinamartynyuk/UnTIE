"""Quality, attribution, stability and canonical branch selection for V8."""
from __future__ import annotations
from collections import defaultdict
import argparse,json,math
from pathlib import Path
import numpy as np
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection_v7.analyze import _activity,_paired,_secondary_answer_metrics,_summary,_switch
from .contract import PRIMARY_AF,PRIMARY_HRR
from .io_guard import read_jsonl,write_json,write_text

def _load(root,branch):
    base=Path(root)/("branch_hrr/predictions" if branch=="hrr" else "branch_v44_af/canonical/predictions")
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for path in base.glob("*/*/*/outer_fold_*/seed_*/*.jsonl"):
        mode,scope,cell=path.relative_to(base).parts[:3];data[mode][cell][path.stem].extend(read_jsonl(path))
    return data

def _analyze_branch(data,base_system,bootstrap):
    primary={};paired={};activity={};switches={};folds={};seeds={};aspects={};ks={}
    for mode,cells in data.items():
      primary[mode]={};paired[mode]={};activity[mode]={};switches[mode]={};folds[mode]={};seeds[mode]={};aspects[mode]={};ks[mode]={}
      for cell,systems in cells.items():
        primary[mode][cell]={s:_summary(rows) for s,rows in systems.items()}
        paired[mode][cell]={s:_paired(systems[base_system],rows,bootstrap) for s,rows in systems.items() if s!=base_system}
        activity[mode][cell]={s:_activity(rows) for s,rows in systems.items() if s!=base_system};switches[mode][cell]={s:_switch(rows) for s,rows in systems.items() if s!=base_system}
        folds[mode][cell]={s:{str(f):_summary([r for r in rows if r["runtime"]["outer_fold"]==f]) for f in sorted({r["runtime"]["outer_fold"] for r in rows})} for s,rows in systems.items()}
        seeds[mode][cell]={s:{str(v):_summary([r for r in rows if r["runtime"]["seed"]==v]) for v in sorted({r["runtime"]["seed"] for r in rows})} for s,rows in systems.items()}
        aspects[mode][cell]={s:{a:_summary([r for r in rows if r["runtime"]["canonical_aspect"]==a]) for a in sorted({r["runtime"]["canonical_aspect"] for r in rows})} for s,rows in systems.items()}
        ks[mode][cell]={s:{str(k):_summary([r for r in rows if r["runtime"]["effective_k"]==k]) for k in sorted({r["runtime"]["effective_k"] for r in rows})} for s,rows in systems.items()}
    return {"primary":primary,"paired":paired,"activity":activity,"switches":switches,"fold_stability":folds,"seed_stability":seeds,"aspect":aspects,"k":ks}

def _global(result,primary_system,base_system):
    mode="local";paired=result["paired"].get(mode,{});metrics=result["primary"].get(mode,{})
    if not paired:return {"complete":False}
    deltas={c:paired[c][primary_system]["delta_token_f1"] for c in paired};scier_values=[v for c,v in deltas.items() if c.startswith("scier_")];scier=float(np.mean(scier_values)) if scier_values else None
    families=[v for c,v in deltas.items() if not c.startswith("scier_")]+([scier] if scier is not None else [])
    complete_protocol=len(deltas)==5;point=complete_protocol and all(v>0 for v in deltas.values());lcb=complete_protocol and all(paired[c][primary_system]["ci_low"]>-.5 for c in deltas);strict=complete_protocol and all(paired[c][primary_system]["ci_low"]>0 for c in deltas)
    evidence=all(100*(metrics[c][primary_system]["hit@1"]-metrics[c][base_system]["hit@1"])>=-.5-1e-12 and 100*(metrics[c][primary_system]["hn@1"]-metrics[c][base_system]["hn@1"])<=1+1e-12 for c in deltas)
    nondeg=all(result["activity"][mode][c][primary_system]["non_degenerate"] for c in deltas)
    return {"complete":complete_protocol,"evaluated_cells":sorted(deltas),"system":primary_system,"baseline":base_system,"cell_deltas":deltas,"scier_family_delta":scier,"global_family_equal_delta":float(np.mean(families)) if families else 0.0,
      "point_positive_all":point,"lcb_above_minus_0_5_all":lcb,"strict_lcb_positive_all":strict,"evidence_safe_all":evidence,"non_degenerate_all":nondeg,
      "strong_universal_criterion":point and lcb and evidence,"confirmation_ready":point and lcb and evidence and nondeg}

def _coefficients(root):
    rows=[]
    paths=list(Path(root).glob("branch_hrr/models/*/*/outer_fold_*/seed_*/*/selection.json"))+list(Path(root).glob("branch_v44_af/canonical/models/*/*/outer_fold_*/seed_*/*/selection.json"))
    for path in paths:
        data=json.loads(path.read_text());selection=data["selection"]
        for name,value in selection.get("standardized_coefficients",{}).items():rows.append({"path":str(path.relative_to(root)),"system":path.parent.name,"feature":name,"value":value})
    summary={}
    for feature in sorted({x["feature"] for x in rows}):
        values=[x["value"] for x in rows if x["feature"]==feature];summary[feature]={"median":float(np.median(values)),"q25":float(np.percentile(values,25)),"q75":float(np.percentile(values,75)),"positive_rate":float(np.mean(np.asarray(values)>0)),"n":len(values)}
    return {"rows":rows,"summary":summary,"raw_coefficients_are_standardized":True}

def _historical_records():
    return {"scirex":{"system":"V4.4 SEM-FIX","token_f1":37.5108},"scier_384":{"system":"V4.4 LEX","token_f1":60.8250},
      "scier_128":{"system":"V4.4 SEM","token_f1":62.5857},"russerrc":{"system":"V4.5 BGE B0","token_f1":69.9535},"scimdix":{"system":"V4.5 BGE B0","token_f1":61.7486},
      "warning":"Historical references use different protocols and are excluded from canonical branch selection."}

def analyze_run(root,bootstrap=5000,progress=None):
    root=Path(root);progress=progress or ProgressLogger(root/"progress.jsonl");progress.stage("analysis","paired bootstrap, attribution, coefficients and frozen branch decision")
    hdata=_load(root,"hrr");fdata=_load(root,"v44_af");h=_analyze_branch(hdata,"H0",bootstrap);f=_analyze_branch(fdata,"F0",bootstrap)
    for branch,result in (("branch_hrr",h),("branch_v44_af",f)):
        for name,value in result.items():write_json(root/branch/"analysis"/f"{name}.json",value)
    hg=_global(h,PRIMARY_HRR,"H0");fg=_global(f,PRIMARY_AF,"F0");write_json(root/"analysis/global_summary.json",{"branch_hrr":hg,"branch_v44_af":fg})
    write_json(root/"analysis/coefficient_diagnostics.json",_coefficients(root));write_json(root/"analysis/historical_references.json",_historical_records())
    write_json(root/"analysis/secondary_answer_metrics_hrr.json",_secondary_answer_metrics(hdata));write_json(root/"analysis/secondary_answer_metrics_v44_af.json",_secondary_answer_metrics(fdata))
    passing=[name for name,value in (("V8-HRR",hg),("V8-V44-AF",fg)) if value.get("strong_universal_criterion")]
    selected=None
    if len(passing)==1:selected=passing[0]
    elif len(passing)==2:
        hm=np.mean([h["primary"]["local"][c][PRIMARY_HRR]["token_f1"] for c in h["primary"].get("local",{})]);fm=np.mean([f["primary"]["local"][c][PRIMARY_AF]["token_f1"] for c in f["primary"].get("local",{})]);selected="V8-HRR" if hm>fm+.2 else ("V8-V44-AF" if fm>hm+.2 else "RESOURCE_TIEBREAK_PENDING")
    status="V8_UNIVERSAL_CANDIDATE_NOT_SUPPORTED" if not passing else ("RESOURCE_BENCHMARK_PENDING" if selected=="RESOURCE_TIEBREAK_PENDING" else "DEVELOPMENT_CANDIDATE_SELECTED")
    decision={"status":status,"passing_branches":passing,"selected":selected,"comparison_protocol":"canonical Layer C only","layer_p_and_r_ineligible":True,"estimate":"DEVELOPMENT"};write_json(root/"analysis/development_decision.json",decision)
    def report(branch,result,base,primary):
        lines=[f"# {branch} results","","Canonical protocol only; LOCAL is primary.","","| Mode | Cell | System | F1 | Δ | 95% CI | Switch | QA |","|---|---|---|---:|---:|---|---:|---:|"]
        for mode,cells in sorted(result["primary"].items()):
          for cell,systems in sorted(cells.items()):
           for system,m in sorted(systems.items()):
            p=result["paired"][mode][cell].get(system,{"delta_token_f1":0,"ci_low":0,"ci_high":0})
            lines.append(f"| {mode} | {cell} | {system} | {m['token_f1']:.4f} | {p['delta_token_f1']:+.4f} | [{p['ci_low']:+.4f}; {p['ci_high']:+.4f}] | {m['switch_rate']:.2%} | {m['qa_forwards']:.3f} |")
        return "\n".join(lines)+"\n"
    write_text(root/"reports/v8_hrr_results.md",report("V8-HRR",h,"H0",PRIMARY_HRR));write_text(root/"reports/v8_v44_af_results.md",report("V8-V44-AF",f,"F0",PRIMARY_AF))
    write_text(root/"reports/v8_branch_comparison.md","# V8 canonical branch comparison\n\n"+json.dumps(decision,ensure_ascii=False,indent=2)+"\n")
    write_text(root/"reports/reproducibility.md","# V8 reproducibility\n\nSee contracts/, audit/, cache/, models/, branch_hrr/, branch_v44_af/, analysis/, and resource_benchmark/. Historical V4.4–V7 inputs are read-only and hash-checked before and after the run.\n")
    progress.log("analysis complete",status=status);return decision

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=5000);a=p.parse_args(argv);analyze_run(a.run_dir,a.bootstrap);return 0
if __name__=="__main__":raise SystemExit(main())
