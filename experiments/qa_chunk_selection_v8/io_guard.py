"""Atomic V8-only writes and explicit V1-V7 read-only boundaries."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path
from typing import Iterable
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

V8_ARTIFACT_ROOT=(PROJECT_ROOT/"artifacts/qa_chunk_selection_v8_dual").resolve()
V8_LOG_ROOT=(PROJECT_ROOT/"logs/qa_chunk_selection_v8").resolve()
HISTORICAL_PREFIXES=tuple((PROJECT_ROOT/x).resolve() for x in (
 "artifacts/qa_chunk_selection_v3","artifacts/qa_chunk_selection_v4_3","artifacts/qa_chunk_selection_v4_4",
 "artifacts/qa_chunk_selection_v4_4_external_20260913_213335","artifacts/qa_chunk_selection_v4_4_ru_20260912_191024",
 "artifacts/qa_chunk_selection_v4_5_ru","artifacts/qa_chunk_selection_v5","artifacts/qa_chunk_selection_v6_universal",
 "artifacts/qa_chunk_selection_v7_rapr","experiments/qa_chunk_selection/v4_4","experiments/qa_chunk_selection/v4_4_external",
 "experiments/qa_chunk_selection/v4_4_ru","experiments/qa_chunk_selection/v4_5_ru_compact_refinement",
 "experiments/qa_chunk_selection/v5_question_conditioned_residual","experiments/qa_chunk_selection_v6",
 "experiments/qa_chunk_selection_v7","configs/qa_chunk_selection/v6_universal","configs/qa_chunk_selection/v7_rapr"))

def _inside(path,root):
    try: Path(path).resolve().relative_to(Path(root).resolve()); return True
    except ValueError: return False

def assert_v8_write(path: Path) -> Path:
    p=Path(path).resolve()
    if any(_inside(p,x) for x in HISTORICAL_PREFIXES): raise RuntimeError(f"historical path is read-only: {p}")
    if not (_inside(p,V8_ARTIFACT_ROOT) or _inside(p,V8_LOG_ROOT)): raise RuntimeError(f"V8 output must stay in its namespace: {p}")
    return p

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def validate_finite(value,at="root"):
    if isinstance(value,float) and not math.isfinite(value): raise ValueError(f"non-finite value at {at}: {value}")
    if isinstance(value,dict):
        for k,v in value.items(): validate_finite(v,f"{at}.{k}")
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value): validate_finite(v,f"{at}[{i}]")

def _atomic(path,payload):
    path=assert_v8_write(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f: f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def write_json(path,value): validate_finite(value);_atomic(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
def write_jsonl(path,rows:Iterable[dict]): rows=list(rows);validate_finite(rows);_atomic(path,"".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in rows).encode())
def write_text(path,text): _atomic(path,str(text).encode())
def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def refuse_overwrite(path,resume=False):
    p=assert_v8_write(path)
    if p.exists() and any(p.iterdir()) and not resume: raise FileExistsError(f"non-empty V8 run exists; use a new run directory or --resume: {p}")
    p.mkdir(parents=True,exist_ok=True);return p
def remove_file(path):
    p=assert_v8_write(path)
    if p.is_file():p.unlink()
