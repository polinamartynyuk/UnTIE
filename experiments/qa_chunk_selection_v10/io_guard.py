from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts/qa_chunk_selection_v10_srr").resolve()
LOG_ROOT = (PROJECT_ROOT / "logs/qa_chunk_selection_v10").resolve()
HISTORICAL_PREFIXES = tuple((PROJECT_ROOT / x).resolve() for x in (
    "artifacts/qa_chunk_selection_v3", "artifacts/qa_chunk_selection_v4_3",
    "artifacts/qa_chunk_selection_v4_4", "artifacts/qa_chunk_selection_v4_4_external_20260913_213335",
    "artifacts/qa_chunk_selection_v4_5_ru", "artifacts/qa_chunk_selection_v5",
    "artifacts/qa_chunk_selection_v6_universal", "artifacts/qa_chunk_selection_v7_rapr",
    "artifacts/qa_chunk_selection_v8_dual", "artifacts/qa_chunk_selection_v9_h2_acs",
    "experiments/qa_chunk_selection_v8", "experiments/qa_chunk_selection_v9",
    "configs/qa_chunk_selection/v8_hrr", "configs/qa_chunk_selection/v9_h2_acs",
))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_v10_write(path: Path) -> Path:
    target = Path(path).resolve()
    if any(_inside(target, root) for root in HISTORICAL_PREFIXES):
        raise RuntimeError(f"historical path is read-only: {target}")
    if not (_inside(target, ARTIFACT_ROOT) or _inside(target, LOG_ROOT)):
        raise RuntimeError(f"V10 output must stay in its namespace: {target}")
    return target


def validate_finite(value, at="root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {at}: {value}")
    if isinstance(value, dict):
        for key, child in value.items(): validate_finite(child, f"{at}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value): validate_finite(child, f"{at}[{index}]")


def _atomic(path: Path, payload: bytes) -> None:
    target = assert_v10_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, target)


def write_json(path: Path, value) -> None:
    validate_finite(value)
    _atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows); validate_finite(rows)
    _atomic(path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows).encode())


def write_text(path: Path, value: str) -> None: _atomic(path, str(value).encode())


def write_joblib(path: Path, value) -> None:
    import joblib
    target = assert_v10_write(path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary); os.replace(temporary, target)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def refuse_overwrite(path: Path, resume=False) -> Path:
    target = assert_v10_write(path)
    if target.exists() and any(target.iterdir()) and not resume:
        raise FileExistsError(f"non-empty V10 run exists; use a new run directory or --resume: {target}")
    if (target / "COMPLETED.json").is_file():
        raise FileExistsError(f"completed V10 run is immutable: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def remove_file(path: Path) -> None:
    target = assert_v10_write(path)
    if target.is_file(): target.unlink()

