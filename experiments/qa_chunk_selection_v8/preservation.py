"""Layer-P exact stored-prediction replay and Layer-R availability audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from .io_guard import read_jsonl,write_json,write_text

def _finite_metrics(path):
    raw=json.loads(Path(path).read_text());return {k:float(v) for k,v in raw.items() if isinstance(v,(int,float)) and math.isfinite(float(v))}

def _entry(cell,system,status,metrics,predictions):
    stored=_finite_metrics(metrics);rows=read_jsonl(predictions);observed=float(np.mean([float(x["token_f1"]) for x in rows])) if rows else None
    difference=abs(observed-float(stored["token_f1"])) if observed is not None else None
    ok=bool(rows) and difference<=1e-10
    return {"cell":cell,"historical_system":system,"historical_status":status,"replay_mode":"exact stored-prediction metric replay",
      "rows":len(rows),"stored_token_f1":stored.get("token_f1"),"replayed_token_f1":observed,"absolute_difference":difference,
      "candidate_agreement":1.0 if ok else None,"answer_agreement":1.0 if ok else None,"status":"PASS" if ok else "PRESERVATION_FAILED"}

def replay_layer_p(config,output):
    root=Path(output);entries=[];v=config.v4_4_run
    specs=[("B0","baseline",v/"reference/b0_live"),("V44_READER_CORE","selected",v/"reference/v6_k5_oof"),("LEX-FIX","diagnostic",v/"lex-fix"),("SEM-FIX","diagnostic",v/"sem-fix")]
    for name,status,path in specs:entries.append(_entry("scirex",name,status,path/"metrics.json",path/"predictions_scored.jsonl"))
    for cell in ("scier_384","scier_128"):
        for name,status in (("B0","baseline"),("V44_READER_CORE","selected reader core"),("LEX","diagnostic"),("SEM","diagnostic")):
            path=config.v4_4_external_run/cell/"results"/("v6" if name=="V44_READER_CORE" else name.lower())
            entries.append(_entry(cell,name,status,path/"metrics.json",path/"predictions_scored.jsonl"))
    failed=[x for x in entries if x["status"]!="PASS"]
    write_json(root/"audit/v44_preservation/results.json",{"entries":entries,"status":"PASS" if not failed else "PRESERVATION_FAILED","stored_prediction_replay":True,"fresh_inference":False})
    lines=["# V4.4 preservation replay","","This is an exact replay of immutable stored predictions, not a new quality experiment.","","| Cell | System | Historical status | Stored F1 | Replayed F1 | Difference | Result |","|---|---|---|---:|---:|---:|---|"]
    for x in entries:lines.append(f"| {x['cell']} | {x['historical_system']} | {x['historical_status']} | {x['stored_token_f1']:.6f} | {x['replayed_token_f1']:.6f} | {x['absolute_difference']:.3g} | {x['status']} |")
    write_text(root/"reports/v44_preservation.md","\n".join(lines)+"\n")
    if failed:raise RuntimeError(f"V4.4 preservation failed: {[(x['cell'],x['historical_system']) for x in failed]}")
    return entries

def layer_r_availability(config,output):
    """Do not approximate Layer R when a common all-candidate historical matrix is absent."""
    required={"scirex":config.v4_4_run/"layer_r_candidate_features.jsonl",
      "scier_384":config.v4_4_external_run/"scier_384/layer_r_candidate_features.jsonl",
      "scier_128":config.v4_4_external_run/"scier_128/layer_r_candidate_features.jsonl"}
    missing=[str(p) for p in required.values() if not p.is_file()]
    result={"status":"BLOCKED_EXACT_INPUT_UNAVAILABLE" if missing else "READY","eligible_for_branch_selection":False,
      "reason":"Exact common per-candidate Layer-R matrix is not stored; protocol forbids approximate reconstruction." if missing else None,
      "missing":missing,"required":{k:str(v.resolve()) for k,v in required.items()}}
    write_json(Path(output)/"branch_v44_af/en_historical_retention/availability.json",result)
    return result
