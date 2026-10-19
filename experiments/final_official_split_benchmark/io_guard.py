"""Small, dependency-free persistence and immutability helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def validate_finite(value: Any, at: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {at}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{at}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite(child, f"{at}[{index}]")


def _atomic(path: Path, payload: bytes, *, immutable: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        old = path.read_bytes()
        if old != payload:
            raise FileExistsError(f"immutable artifact already exists with different content: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    validate_finite(value)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _atomic(path, payload, immutable=immutable)


def write_jsonl(path: Path, rows: Iterable[dict], *, immutable: bool = False) -> None:
    values = list(rows)
    validate_finite(values)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in values).encode()
    _atomic(path, payload, immutable=immutable)


def write_text(path: Path, value: str, *, immutable: bool = False) -> None:
    _atomic(path, value.encode(), immutable=immutable)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def refuse_overwrite(run_dir: Path, resume: bool) -> None:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"run directory exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def assert_same_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} changed: {observed} != {expected}")
