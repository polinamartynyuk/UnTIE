"""Atomic V9-only writes with V1-V8 treated as immutable history."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

V9_ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts/qa_chunk_selection_v9_h2_acs").resolve()
V9_LOG_ROOT = (PROJECT_ROOT / "logs/qa_chunk_selection_v9").resolve()
HISTORICAL_PREFIXES = tuple((PROJECT_ROOT / x).resolve() for x in (
    "artifacts/qa_chunk_selection_v3",
    "artifacts/qa_chunk_selection_v4_3",
    "artifacts/qa_chunk_selection_v4_4",
    "artifacts/qa_chunk_selection_v4_4_external_20260913_213335",
    "artifacts/qa_chunk_selection_v4_5_ru",
    "artifacts/qa_chunk_selection_v5",
    "artifacts/qa_chunk_selection_v6_universal",
    "artifacts/qa_chunk_selection_v7_rapr",
    "artifacts/qa_chunk_selection_v8_dual",
    "experiments/qa_chunk_selection/v4_4",
    "experiments/qa_chunk_selection/v4_4_external",
    "experiments/qa_chunk_selection/v4_4_ru",
    "experiments/qa_chunk_selection/v4_5_ru_compact_refinement",
    "experiments/qa_chunk_selection/v5_question_conditioned_residual",
    "experiments/qa_chunk_selection_v6",
    "experiments/qa_chunk_selection_v7",
    "experiments/qa_chunk_selection_v8",
    "configs/qa_chunk_selection/v6_universal",
    "configs/qa_chunk_selection/v7_rapr",
    "configs/qa_chunk_selection/v8_hrr",
))


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def assert_v9_write(path: Path) -> Path:
    target = Path(path).resolve()
    if any(_inside(target, root) for root in HISTORICAL_PREFIXES):
        raise RuntimeError(f"historical path is read-only: {target}")
    if not (_inside(target, V9_ARTIFACT_ROOT) or _inside(target, V9_LOG_ROOT)):
        raise RuntimeError(f"V9 output must stay in its namespace: {target}")
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_finite(value, at: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {at}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{at}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite(child, f"{at}[{index}]")


def _atomic(path: Path, payload: bytes) -> None:
    target = assert_v9_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def write_json(path: Path, value) -> None:
    validate_finite(value)
    _atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    validate_finite(rows)
    payload = "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows)
    _atomic(path, payload.encode())


def write_text(path: Path, value: str) -> None:
    _atomic(path, str(value).encode())


def write_joblib(path: Path, value) -> None:
    """Atomically serialize a model while enforcing the V9-only write boundary."""
    import joblib

    target = assert_v9_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def refuse_overwrite(path: Path, resume: bool = False) -> Path:
    target = assert_v9_write(path)
    if target.exists() and any(target.iterdir()) and not resume:
        raise FileExistsError(f"non-empty V9 run exists; use a new run directory or --resume: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def remove_file(path: Path) -> None:
    target = assert_v9_write(path)
    if target.is_file():
        target.unlink()

