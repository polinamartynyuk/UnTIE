"""V7 read-only historical audit over immutable V5/V6 inputs."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.audit import (baseline_manifest,candidate_and_lineage_audit,historical_hashes as v5_hashes)
from .contract import CELLS,FAMILIES,V5_RUN,V6_RUN
from .io_guard import sha256,write_json,write_text

def historical_hashes(v5_run=V5_RUN,v6_run=V6_RUN,cells=CELLS):
    values={f"v5:{k}":v for k,v in v5_hashes(v5_run,cells).items()}
    critical=[v6_run/"COMPLETED.json",v6_run/"frozen_contract.yaml",v6_run/"analysis/primary_metrics.json",
              v6_run/"analysis/selected_development_candidate.json",v6_run/"analysis/universality_summary.json",
              v6_run/"analysis/switch_analysis.json",v6_run/"audit/regression_summary.json"]
    source=Path(__file__).resolve().parents[1]/"qa_chunk_selection_v6"
    critical += sorted(p for p in source.rglob("*.py") if "__pycache__" not in p.parts)
    missing=[str(p) for p in critical if not p.is_file()]
    if missing:raise FileNotFoundError(f"historical V6 inputs missing: {missing}")
    for p in critical: values[f"v6:{p.relative_to(p.parents[2] if 'artifacts' in p.parts else source.parent)}"]={"path":str(p.resolve()),"sha256":sha256(p),"size":p.stat().st_size}
    return dict(sorted(values.items()))

def git_snapshot(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name,cmd in {"status.txt":["git","status","--porcelain"],"head.txt":["git","rev-parse","HEAD"],"diff.patch":["git","diff","--no-ext-diff"]}.items():
        done=subprocess.run(cmd,text=True,capture_output=True,check=True);write_text(destination/name,done.stdout)

def run_audit(output,v5_run=V5_RUN,v6_run=V6_RUN,cells=CELLS):
    output=Path(output);hashes=historical_hashes(v5_run,v6_run,cells);candidates,lineage,overlap=candidate_and_lineage_audit(cells)
    write_json(output/"historical_hashes_before.json",hashes);write_json(output/"canonical_baselines.json",baseline_manifest(v5_run,cells))
    write_json(output/"candidate_universe.json",candidates);write_json(output/"scier_lineage.json",lineage);write_json(output/"cross_corpus_overlap.json",overlap)
    write_json(output/"k_manifest.json",{c:V5_CELLS[c].k for c in cells});write_json(output/"split_hashes.json",{c:sha256(V5_CELLS[c].splits) for c in cells})
    write_json(output/"upstream_provenance.json",{c:"dataset_trained; reused exact V5 META_FEATURE_OOF" for c in cells})
    for c in cells:
        model=v5_run/"inputs/model_manifests"/f"{c}_models.json";question=v5_run/"inputs/question_manifests"/f"{c}.json"
        write_json(output/"model_manifests"/f"{c}.json",{"source":str(model.resolve()),"sha256":sha256(model),"payload":json.loads(model.read_text())})
        write_json(output/"question_policy"/f"{c}.json",{"source":str(question.resolve()),"sha256":sha256(question),"payload":json.loads(question.read_text())})
    return {"hashes":hashes,"candidate_universe":candidates,"lineage":lineage,"overlap":overlap}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path);p.add_argument("--read-only-history",action="store_true");p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args(argv)
    if not a.read_only_history:raise ValueError("--read-only-history is mandatory")
    run_audit(a.run_dir/"audit");return 0
if __name__=="__main__":raise SystemExit(main())
