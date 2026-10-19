"""Fresh-process resource benchmark for canonical V8 systems."""
from __future__ import annotations
from collections import defaultdict
import json,platform,subprocess,sys
import numpy as np
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .contract import PRIMARY_AF,PRIMARY_HRR
from .io_guard import read_jsonl,write_json

def _sample_manifest(output,cells,cap):
    result={"contract":"LOCAL outer-fold-0 seed-42; round-robin by aspect and K; warmup excluded","cells":{}}
    for cell in cells:
        path=output/"branch_hrr/predictions/local"/cell/cell/"outer_fold_0/seed_42/H0.jsonl";rows=read_jsonl(path);strata=defaultdict(list)
        for r in rows:strata[(r["runtime"]["canonical_aspect"],int(r["runtime"]["effective_k"]))].append(r["runtime"]["question_id"])
        for values in strata.values():values.sort()
        chosen=[]
        while len(chosen)<min(cap,len(rows)) and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(chosen)<min(cap,len(rows)):chosen.append(strata[key].pop(0))
        result["cells"][cell]={"question_ids":chosen,"questions":len(chosen)}
    write_json(output/"resource_benchmark/sample_manifest.json",result);return result

def run_benchmarks(config,output,progress):
    progress.stage("benchmark","fresh-process B0, both V8 primaries and V44 anchor; Qwen/BGE exact-runner audit")
    sample=_sample_manifest(output,config.datasets,config.benchmark_questions);systems=("H0","H1",PRIMARY_HRR,"F0",PRIMARY_AF);rows=[]
    for cell in config.datasets:
      for system in systems:
        progress.log("resource worker",cell=cell,system=system)
        cmd=[sys.executable,"-m","experiments.qa_chunk_selection_v8.benchmark_worker","--cell",cell,"--system",system,"--run-root",str(output),"--v5-run",str(config.v5_run),
             "--device",config.device,"--sample-manifest",str(output/"resource_benchmark/sample_manifest.json"),"--warmup","10","--repeats","3"]
        done=subprocess.run(cmd,cwd=PROJECT_ROOT,text=True,capture_output=True)
        if done.returncode:raise RuntimeError(f"resource worker failed {cell}/{system}: {(done.stderr or done.stdout)[-8000:]}")
        line=next((x for x in reversed(done.stdout.splitlines()) if x.strip().startswith("{")),None)
        if line is None:raise RuntimeError(f"no benchmark JSON for {cell}/{system}")
        row=json.loads(line);rows.append(row);write_json(output/f"resource_benchmark/raw/{cell}_{system}.json",row)
    external={"Qwen":{"status":"BLOCKED_NO_EXACT_FRESH_RUNNER","reason":"No historical Qwen runtime implementation/model identity is present in the repository; historical cross-hardware numbers are not substituted."},
      "BGE_RU":{"status":"CONTEXT_ONLY","reason":"Historical BGE quality/resources retained as context; canonical V8 primary keeps exact V7 ranker and does not silently replace it."}}
    write_json(output/"resource_benchmark/summary.json",{"runs":rows,"sample_manifest":sample,"systems":list(systems),"external_references":external,
      "interpretation":"Fresh canonical comparisons are valid; dissertation Qwen resource claim remains blocked until an exact runner/model is supplied."})
    decision_path=output/"analysis/development_decision.json"
    if decision_path.is_file():
        decision=json.loads(decision_path.read_text())
        if decision.get("selected")=="RESOURCE_TIEBREAK_PENDING":
            h=float(np.mean([x["median_latency_seconds"] for x in rows if x["system"]==PRIMARY_HRR]));f=float(np.mean([x["median_latency_seconds"] for x in rows if x["system"]==PRIMARY_AF]))
            decision["resource_tiebreak"]={"hrr_mean_median_seconds":h,"v44_af_mean_median_seconds":f};decision["selected"]="V8-HRR" if h<f else "V8-V44-AF";decision["status"]="DEVELOPMENT_CANDIDATE_SELECTED"
        decision["qwen_resource_claim_status"]="BLOCKED_NO_EXACT_FRESH_RUNNER";write_json(decision_path,decision)
    try:
        import torch
        tv=torch.__version__;cuda=torch.version.cuda
    except ImportError:tv=None;cuda=None
    write_json(output/"resource_benchmark/environment.json",{"python":sys.version,"platform":platform.platform(),"torch":tv,"cuda":cuda,"executable":sys.executable})

