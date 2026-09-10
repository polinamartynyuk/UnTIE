"""Write isolated preprocessing tables without touching the source dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


TABLE_NAMES = (
    "documents",
    "chunks",
    "aspect_instances",
    "gold_mentions",
    "source_metadata",
    "enrichment",
    "qa_input",
)


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".jsonl")
        frame.to_json(fallback, orient="records", lines=True, force_ascii=False)
        return fallback


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_tables(
    output_dir: Path,
    *,
    documents: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    aspect_instances: Sequence[Mapping[str, Any]],
    gold_mentions: Sequence[Mapping[str, Any]],
    source_metadata: Sequence[Mapping[str, Any]],
    enrichment: Sequence[Mapping[str, Any]] | None = None,
    qa_input: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "documents": pd.DataFrame(list(documents)),
        "chunks": pd.DataFrame(list(chunks)),
        "aspect_instances": pd.DataFrame(list(aspect_instances)),
        "gold_mentions": pd.DataFrame(list(gold_mentions)),
        "source_metadata": pd.DataFrame(list(source_metadata)),
        "enrichment": pd.DataFrame(
            list(enrichment or []),
            columns=[
                "document_id",
                "aspect",
                "raw_gold_answer",
                "candidate_value",
                "candidate_type",
                "source_type",
                "source_reference",
                "matching_method",
                "confidence",
                "used_for_gold",
            ],
        ),
        "qa_input": pd.DataFrame(list(qa_input or [])),
    }
    written: dict[str, str] = {}
    for name, frame in frames.items():
        written[name] = str(_write_parquet(frame, output_dir / f"{name}.parquet"))
    return written
