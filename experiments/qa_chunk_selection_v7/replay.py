"""Replay immutable B0/V5 and verify frozen V6 reference predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from experiments.qa_chunk_selection_v6.replay import replay as replay_v5
from experiments.qa_chunk_selection_v6.io_guard import read_jsonl as read_historical_jsonl
from .contract import CELLS,V5_RUN,V6_RUN

def replay(v5_run=V5_RUN,v6_run=V6_RUN,cells=CELLS,strict=True):
    report={"b0_v5":replay_v5(v5_run,cells,strict=strict),"v6":{}}
    primary=json.loads((v6_run/"analysis/primary_metrics.json").read_text())
    for cell in cells:
        entry={}
        for system in ("S4","S4-lex"):
            paths=sorted((v6_run/"predictions/local"/cell/cell).glob(f"outer_fold_*/seed_*/{system}.jsonl"))
            rows=[row for p in paths for row in read_historical_jsonl(p)]
            observed=float(np.mean([r["evaluation"]["token_f1"] for r in rows])) if rows else None
            expected=primary["local"][cell][system]["token_f1"]
            ok=bool(rows) and abs(observed-expected)<1e-10 and all(r["runtime"]["selected_chunk_id"] in {x["chunk_id"] for x in r["runtime"]["b0_top_k"]} for r in rows)
            entry[system]={"status":"PASS" if ok else "FAIL","rows":len(rows),"recomputed_token_f1":observed,"stored_token_f1":expected}
            if strict and not ok:raise RuntimeError(f"V6_REPLAY_FAILED {cell}/{system}")
        report["v6"][cell]=entry
    return report
