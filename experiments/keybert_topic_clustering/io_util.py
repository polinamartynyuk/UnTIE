from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTECTED_ROOTS = (
    "artifacts/hierarchical_topic_optimized",
)


def assert_isolated_output(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    root = project_root.resolve()
    for relative in PROTECTED_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents or resolved == protected:
            raise ValueError(
                "keybert experiment refuses to write into a production evaluation "
                f"directory: {resolved}"
            )
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        else:
            raise ValueError(
                "keybert experiment refuses to write into a production evaluation "
                f"directory: {resolved}"
            )
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_timing(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))
