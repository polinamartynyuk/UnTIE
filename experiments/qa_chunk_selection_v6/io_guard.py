"""Atomic V6 writes and an explicit read-only boundary around historical work."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

V6_ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts/qa_chunk_selection_v6_universal").resolve()
V6_LOG_ROOT = (PROJECT_ROOT / "logs/qa_chunk_selection_v6").resolve()
HISTORICAL_PREFIXES = tuple(
    (PROJECT_ROOT / value).resolve() for value in (
        "artifacts/qa_chunk_selection_v3",
        "artifacts/qa_chunk_selection_v4_3",
        "artifacts/qa_chunk_selection_v4_4",
        "artifacts/qa_chunk_selection_v4_4_external_20260913_213335",
        "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431",
        "artifacts/qa_chunk_selection_v4_5_ru",
        "artifacts/qa_chunk_selection_v5",
        "experiments/qa_chunk_selection/v4_4",
        "experiments/qa_chunk_selection/v4_4_external",
        "experiments/qa_chunk_selection/v4_4_ru",
        "experiments/qa_chunk_selection/v4_5_ru_compact_refinement",
        "experiments/qa_chunk_selection/v5_question_conditioned_residual",
    )
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_v6_write(path: Path) -> Path:
    resolved = Path(path).resolve()
    if any(_inside(resolved, root) for root in HISTORICAL_PREFIXES):
        raise PermissionError(f"historical path is read-only: {resolved}")
    if not (_inside(resolved, V6_ARTIFACT_ROOT) or _inside(resolved, V6_LOG_ROOT)):
        raise PermissionError(f"V6 output must stay in its namespace: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_finite(value, at="root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {at}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{at}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite(child, f"{at}[{index}]")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = assert_v6_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, value) -> None:
    validate_finite(value)
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    validate_finite(rows)
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    _atomic_bytes(path, text.encode())


def write_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def refuse_overwrite(path: Path, *, resume: bool = False) -> Path:
    path = assert_v6_write(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"non-empty V6 run exists; use a new run directory or --resume: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

