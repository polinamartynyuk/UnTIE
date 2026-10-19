"""Read-only history, feature provenance and repository-state audit for V8."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
from experiments.qa_chunk_selection_v7.audit import historical_hashes as v7_upstream_hashes
from experiments.qa_chunk_selection_v6.audit import baseline_manifest,candidate_and_lineage_audit
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from .contract import CELLS,V4_4_RUN,V4_4_EXTERNAL_RUN,V4_5_RUN,V5_RUN,V6_RUN,V7_RUN,SEM_COMMON,SUPPORT_COMMON
from .io_guard import sha256,write_json,write_text

def git_snapshot(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name,cmd in {"status.txt":["git","status","--porcelain"],"head.txt":["git","rev-parse","HEAD"],"diff.patch":["git","diff","--no-ext-diff"]}.items():
        done=subprocess.run(cmd,text=True,capture_output=True,check=True);write_text(destination/name,done.stdout)

def _critical_files(v4_4,v4_4_external,v4_5,v7):
    files=[v4_4/"status.json",v4_4/"contract.json",v4_4/"reference/b0_live/metrics.json",v4_4/"reference/v6_k5_oof/metrics.json",
      v4_4/"lex-fix/metrics.json",v4_4/"sem-fix/metrics.json",v4_4_external/"COMPLETED.json",v4_5/"COMPLETED.json",
      v4_5/"analysis/primary_metrics.json",v7/"COMPLETED.json",v7/"frozen_contract.yaml",v7/"analysis/primary_metrics.json"]
    for cell in ("scier_384","scier_128"):
        for system in ("b0","v6","lex","sem"):files += [v4_4_external/cell/"results"/system/"metrics.json",v4_4_external/cell/"results"/system/"predictions_scored.jsonl"]
    files += [v4_4/"reference/b0_live/predictions_scored.jsonl",v4_4/"reference/v6_k5_oof/predictions_scored.jsonl",
              v4_4/"lex-fix/predictions_scored.jsonl",v4_4/"sem-fix/predictions_scored.jsonl"]
    source=Path(__file__).resolve().parents[1]/"qa_chunk_selection_v7";files += sorted(p for p in source.rglob("*.py") if "__pycache__" not in p.parts)
    return files

def historical_hashes(v4_4=V4_4_RUN,v4_4_external=V4_4_EXTERNAL_RUN,v4_5=V4_5_RUN,v5=V5_RUN,v6=V6_RUN,v7=V7_RUN,cells=CELLS):
    values={f"upstream:{k}":v for k,v in v7_upstream_hashes(v5,v6,cells).items()}
    files=_critical_files(v4_4,v4_4_external,v4_5,v7);missing=[str(p) for p in files if not p.is_file()]
    if missing:raise FileNotFoundError(f"historical critical files missing: {missing}")
    for p in files:values[f"critical:{p.resolve()}"]={"path":str(p.resolve()),"sha256":sha256(p),"size":p.stat().st_size}
    return dict(sorted(values.items()))

def run_audit(output,config):
    output=Path(output);hashes=historical_hashes(config.v4_4_run,config.v4_4_external_run,config.v4_5_run,config.v5_run,config.v6_run,config.v7_run,config.datasets)
    write_json(output/"hashes_before.json",hashes)
    write_json(output/"feature_recovery.json",{
      "B0":{"status":"RECOVERED","source":"V5 canonical b0_score_raw/b0_score_norm"},
      "Q-A":{"status":"RECOVERED","source":"V5 q_answer_ranker_raw; exact fold ranker state"},
      "reader_agreement":{"status":"RECOVERED","source":"v4.v6_reader.agreement_features exact_freq+majority","normalization":"raw difference"},
      "FullLEX":{"status":"RECOVERED","source":"v4.v6_reader mean_token_f1/100","normalization":"within-question z"},
      "PairLEX":{"status":"RECOVERED","source":"V7 TokenF1(Ai,A_B0)/100","reference":"B0 answer"},
      "support":{"status":"RECOVERED","source":"v4.v6_reader source_clusters","globally_common":SUPPORT_COMMON},
      "SEM":{"status":"UNRECOVERED_CANONICAL","globally_common":SEM_COMMON,"reason":"V5/V7 canonical rows store sem_available=0 and sem_norm=0 on all cells; disabled before EVAL"}})
    write_json(output/"v44_common_core.json",{"same_schema_all_cells":True,"components":["B0","reader agreement","support"],"support_global_eligibility":SUPPORT_COMMON,
      "parameter_family":"alpha*B0+beta*(exact_freq+mean_token_f1+majority)+gamma*source_clusters","grid":[0,.25,.5,1,2],"hardcoded_language_weights":False})
    write_json(output/"history_write_guard.json",{"active":True,"historical_roots_read_only":True,"v8_namespace_only":True})
    candidates,lineage,overlap=candidate_and_lineage_audit(config.datasets)
    write_json(output/"candidate_universe.json",candidates);write_json(output/"scier_lineage.json",lineage);write_json(output/"cross_corpus_overlap.json",overlap)
    write_json(output/"canonical_baselines.json",baseline_manifest(config.v5_run,config.datasets));write_json(output/"k_manifest.json",{c:V5_CELLS[c].k for c in config.datasets})
    write_json(output/"split_hashes.json",{c:sha256(V5_CELLS[c].splits) for c in config.datasets})
    write_json(output/"upstream_provenance.json",{c:"dataset-trained; exact V5 META_FEATURE_OOF and outer refit" for c in config.datasets})
    for cell in config.datasets:
        source=config.v5_run/"inputs/model_manifests"/f"{cell}_models.json";write_json(output/"model_manifests"/f"{cell}.json",{"source":str(source.resolve()),"sha256":sha256(source),"payload":json.loads(source.read_text())})
        question=config.v5_run/"inputs/question_manifests"/f"{cell}.json";write_json(output/"question_policy"/f"{cell}.json",{"source":str(question.resolve()),"sha256":sha256(question),"payload":json.loads(question.read_text())})
    return hashes
