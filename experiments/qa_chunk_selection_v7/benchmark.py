from __future__ import annotations
import json,platform,subprocess,sys
from collections import defaultdict
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .io_guard import read_jsonl,write_json

def _sample_manifest(output,cells,cap):
    result={"contract":"outer-fold-0 LOCAL seed-42; capped at 100; round-robin stratified by aspect and effective_k","cells":{}}
    for cell in cells:
        path=output/"predictions/local"/cell/cell/"outer_fold_0/seed_42/S0.jsonl";rows=read_jsonl(path);strata=defaultdict(list)
        for r in rows:
            key=(r["runtime"]["canonical_aspect"],int(r["runtime"]["effective_k"]));strata[key].append(r["runtime"]["question_id"])
        for values in strata.values():values.sort()
        chosen=[]
        while len(chosen)<min(cap,len(rows)) and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(chosen)<min(cap,len(rows)):chosen.append(strata[key].pop(0))
        result["cells"][cell]={"question_ids":chosen,"questions":len(chosen),"strata":sorted({f"{a}|K={k}" for a,k in strata})}
    write_json(output/"resource_benchmark/sample_manifest.json",result);return result

def run_benchmarks(config,output,progress):
    progress.stage("benchmark","fresh-process LOCAL fold-0 seed-42 timing with fixed stratified samples")
    sample=_sample_manifest(output,config.datasets,config.benchmark_questions);rows=[];systems=["S0","S1","S2","S3"]
    for cell in config.datasets:
      for system in systems:
        progress.log("resource worker",cell=cell,system=system,mode="local",fold=0,seed=42)
        cmd=[sys.executable,"-m","experiments.qa_chunk_selection_v7.benchmark_worker","--cell",cell,"--system",system,"--run-root",str(output),"--v5-run",str(config.v5_run),
             "--device",config.device,"--sample-manifest",str(output/"resource_benchmark/sample_manifest.json"),"--warmup","10","--repeats","3"]
        done=subprocess.run(cmd,cwd=PROJECT_ROOT,text=True,capture_output=True)
        if done.returncode:raise RuntimeError(f"resource worker failed {cell}/{system}: {(done.stderr or done.stdout)[-8000:]}")
        line=next((x for x in reversed(done.stdout.splitlines()) if x.strip().startswith("{")),None)
        if line is None:raise RuntimeError(f"no benchmark JSON for {cell}/{system}")
        row=json.loads(line);rows.append(row);write_json(output/f"resource_benchmark/raw/{cell}_{system}.json",row)
    write_json(output/"resource_benchmark/summary.json",{"runs":rows,"sample_manifest":sample,"mode":"local","fold":0,"seed":42,"systems":systems,
               "hard_max_qa":3,"latency_targets":{"median_ratio":1.25,"p95_ratio":2.5}})
    indexed={(r["cell"],r["system"]):r for r in rows};cells={}
    for cell in config.datasets:
        base=indexed[(cell,"S0")];value=indexed[(cell,"S3")]
        cells[cell]={"median_ratio":value["median_latency_seconds"]/base["median_latency_seconds"],"p95_ratio":value["p95_latency_seconds"]/base["p95_latency_seconds"],
                     "mean_qa_forwards":value["mean_qa_forwards"],"max_qa_forwards":value["max_qa_forwards"]}
        cells[cell]["pass"]=cells[cell]["max_qa_forwards"]<=3 and cells[cell]["median_ratio"]<=1.25 and cells[cell]["p95_ratio"]<=2.5
    resource_pass=all(v["pass"] for v in cells.values());resource={"cells":cells,"resource_gate":resource_pass,"hard_max_qa_pass":all(v["max_qa_forwards"]<=3 for v in cells.values())}
    write_json(output/"analysis/resource_summary.json",resource)
    global_path=output/"analysis/global_summary.json";quality=json.loads(global_path.read_text()).get("local",{}).get("quality_gate",False) if global_path.is_file() else False
    status="GO_TO_FREEZE" if quality and resource_pass else ("QUALITY_ONLY_CANDIDATE" if quality else "V7_UNIVERSAL_CANDIDATE_NOT_SUPPORTED")
    write_json(output/"analysis/development_decision.json",{"primary_mode":"local","primary_system":"S3","quality_gate":quality,"resource_gate":resource_pass,"status":status,"diagnostics_ineligible":True,"estimate":"DEVELOPMENT"})
    try:
        import torch
        torch_version=torch.__version__;cuda=torch.version.cuda
    except ImportError:torch_version=None;cuda=None
    write_json(output/"resource_benchmark/environment.json",{"python":sys.version,"platform":platform.platform(),"torch":torch_version,"cuda":cuda,"executable":sys.executable})
