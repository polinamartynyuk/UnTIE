"""Read frozen SciREX preprocessing tables. Never write back to them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import ALLOWED_ASPECTS, ALLOWED_STATUS


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    try:
        items = list(value)
    except TypeError:
        return [str(value)]
    return [str(item) for item in items if item is not None and str(item).strip()]


def as_int_list(value: Any) -> list[int]:
    return [int(item) for item in as_list(value)]


def load_preprocessing(input_dir: Path) -> dict[str, pd.DataFrame]:
    required = (
        "documents",
        "chunks",
        "aspect_instances",
        "gold_mentions",
        "source_metadata",
        "qa_input",
    )
    tables: dict[str, pd.DataFrame] = {}
    for name in required:
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing preprocessing table: {path}")
        tables[name] = pd.read_parquet(path)
    return tables


def load_document_id_list(path: Path) -> tuple[str, ...]:
    """One document_id per line. Blank lines and # comments are ignored."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text not in seen:
            seen.add(text)
            ids.append(text)
    return tuple(ids)


def filter_instances_by_document_ids(
    instances: pd.DataFrame,
    document_ids: Sequence[str],
) -> pd.DataFrame:
    if not document_ids:
        return instances.reset_index(drop=True)
    wanted = {str(item) for item in document_ids}
    return instances[instances["document_id"].astype(str).isin(wanted)].reset_index(drop=True)


def eligible_instances(instances: pd.DataFrame, *, split: str | None = None) -> pd.DataFrame:
    frame = instances[
        instances["aspect"].isin(ALLOWED_ASPECTS)
        & instances["status"].isin(ALLOWED_STATUS)
    ].copy()
    if "score" in frame["aspect"].tolist():
        raise ValueError("score aspect leaked into Phase 1 instances")
    if split:
        frame = frame[frame["split"] == split]
    return frame.reset_index(drop=True)


def document_chunks(chunks: pd.DataFrame, document_id: str) -> pd.DataFrame:
    frame = chunks[chunks["document_id"] == document_id].copy()
    return frame.sort_values("chunk_id").reset_index(drop=True)


def eligible_chunks(chunks: pd.DataFrame, document_id: str) -> pd.DataFrame:
    frame = document_chunks(chunks, document_id)
    if "eligible_for_qa" in frame.columns:
        frame = frame[frame["eligible_for_qa"].astype(bool)]
    return frame.reset_index(drop=True)


def select_debug_document_ids(
    instances: pd.DataFrame,
    *,
    n_docs: int,
    seed: int = 42,
) -> list[str]:
    """Deterministic TRAIN sample; prefer multi-answer instances."""
    frame = eligible_instances(instances)
    scored: dict[str, tuple[int, str]] = {}
    for _, row in frame.iterrows():
        doc_id = str(row["document_id"])
        n_extractable = len(as_list(row["extractable_gold_answers"]))
        scored[doc_id] = (max(scored.get(doc_id, (0, doc_id))[0], n_extractable), doc_id)
    ranked = sorted(scored.values(), key=lambda item: (-item[0], item[1]))
    chosen = [doc_id for _, doc_id in ranked[:n_docs]]
    if len(chosen) < n_docs:
        raise ValueError(f"need {n_docs} debug documents, found {len(chosen)}")
    del seed
    return chosen


def instance_gold_supporting(row: pd.Series) -> set[int]:
    return set(as_int_list(row.get("gold_supporting_chunk_ids")))


def qa_instance_id(document_id: str, aspect: str) -> str:
    return f"{document_id}::{aspect}"
