"""Load frozen SciREX chunks + Phase 1 Q5 labels. Never rewrite sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from experiments.phase1_scirex.tables import as_list, load_preprocessing
from untie.aspect_qa.config import ASPECT_QUESTIONS

from .contract import LABEL_MAP
from .progress import ProgressLogger
from .schema import ChunkRecord, GoldSpan, InstanceRecord, as_list as schema_as_list


def _as_int_list(value: Any) -> list[int]:
    return [int(item) for item in schema_as_list(value)]


def map_primary_class(primary_class: str) -> str:
    mapped = LABEL_MAP.get(str(primary_class))
    if mapped is None:
        raise ValueError(f"unmapped Phase 1 class: {primary_class}")
    return mapped


def _mention_index(mentions: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if mentions.empty:
        return index
    for row in mentions.itertuples(index=False):
        if hasattr(row, "in_bibliography") and bool(getattr(row, "in_bibliography")):
            continue
        key = (str(row.document_id), str(row.aspect))
        index.setdefault(key, []).append(
            {
                "char_start": int(row.char_start),
                "char_end": int(row.char_end),
                "text": str(getattr(row, "matched_text", "") or getattr(row, "canonical_answer", "")),
                "chunk_ids": _as_int_list(getattr(row, "chunk_ids")),
            }
        )
    return index


def _gold_spans_for_instance(
    mention_rows: Sequence[dict[str, Any]],
    chunks: Sequence[ChunkRecord],
) -> tuple[GoldSpan, ...]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    spans: list[GoldSpan] = []
    for row in mention_rows:
        doc_start = int(row["char_start"])
        doc_end = int(row["char_end"])
        text = str(row["text"])
        for chunk_id in row["chunk_ids"]:
            chunk = by_id.get(int(chunk_id))
            if chunk is None:
                continue
            rel_start = max(0, doc_start - chunk.char_start)
            rel_end = min(len(chunk.text), max(rel_start, doc_end - chunk.char_start))
            spans.append(
                GoldSpan(
                    chunk_id=int(chunk_id),
                    start_char=int(rel_start),
                    end_char=int(rel_end),
                    text=text,
                    document_start_char=doc_start,
                    document_end_char=doc_end,
                )
            )
    return tuple(spans)


def _neighbor_ids(chunk_ids: Sequence[int], position: int) -> tuple[int | None, int | None]:
    prev_id = int(chunk_ids[position - 1]) if position > 0 else None
    next_id = int(chunk_ids[position + 1]) if position + 1 < len(chunk_ids) else None
    return prev_id, next_id


def load_split_records(
    label_dir: Path,
    preprocess_dir: Path,
    *,
    split_name: str,
    progress: ProgressLogger | None = None,
) -> tuple[InstanceRecord, ...]:
    label_dir = Path(label_dir)
    chunk_path = label_dir / "qa_chunk_dataset.parquet"
    instance_path = label_dir / "qa_instances.parquet"
    if not chunk_path.exists() or not instance_path.exists():
        raise FileNotFoundError(
            f"missing Phase 1 Q5 labels under {label_dir}. "
            "Expected qa_chunk_dataset.parquet and qa_instances.parquet."
        )
    if progress:
        progress.log(f"reading labels from {label_dir}")
    chunks_frame = pd.read_parquet(chunk_path)
    instances_frame = pd.read_parquet(instance_path)
    tables = load_preprocessing(Path(preprocess_dir))
    mentions = _mention_index(tables["gold_mentions"])
    grouped_chunks: dict[tuple[str, str], pd.DataFrame] = {
        (str(doc_id), str(aspect)): frame.sort_values("chunk_id")
        for (doc_id, aspect), frame in chunks_frame.groupby(["document_id", "aspect"], sort=False)
    }
    records: list[InstanceRecord] = []
    total = len(instances_frame)
    for _, row in instances_frame.iterrows():
        document_id = str(row["document_id"])
        aspect = str(row["aspect"])
        question = str(row.get("canonical_question") or ASPECT_QUESTIONS[aspect])
        if question != ASPECT_QUESTIONS[aspect]:
            raise ValueError(f"canonical question drifted for {document_id}::{aspect}")
        frame = grouped_chunks[(document_id, aspect)]
        ordered_ids = [int(value) for value in frame["chunk_id"].tolist()]
        built: list[ChunkRecord] = []
        for position, (_, chunk_row) in enumerate(frame.iterrows()):
            prev_id, next_id = _neighbor_ids(ordered_ids, position)
            section_title = str(chunk_row.get("section_title") or "")
            section_type = str(chunk_row.get("section_type") or "")
            section_id = section_type or section_title or f"pos:{position}"
            primary = str(chunk_row["primary_class"])
            built.append(
                ChunkRecord(
                    chunk_id=int(chunk_row["chunk_id"]),
                    text=str(chunk_row["chunk_text"]),
                    label=map_primary_class(primary),
                    primary_class=primary,
                    position=position,
                    section_id=section_id,
                    section_title=section_title,
                    char_start=int(chunk_row["char_start"]),
                    char_end=int(chunk_row["char_end"]),
                    token_start=int(chunk_row["token_start"]),
                    token_end=int(chunk_row["token_end"]),
                    gold_supporting=bool(chunk_row["gold_supporting"]),
                    previous_chunk_id=prev_id,
                    next_chunk_id=next_id,
                )
            )
        gold_answers = tuple(as_list(row.get("extractable_gold_answers")))
        record = InstanceRecord(
            document_id=document_id,
            question_id=str(row.get("qa_instance_id") or f"{document_id}::{aspect}"),
            aspect=aspect,
            question=question,
            gold_answers=gold_answers,
            split=split_name,
            chunks=tuple(built),
            gold_spans=_gold_spans_for_instance(
                mentions.get((document_id, aspect), ()),
                built,
            ),
            instance_status=str(row.get("instance_status") or "ok"),
            source_split=str(row.get("split") or split_name),
        )
        records.append(record)
        if progress:
            progress.count(len(records), total, item=f"{split_name} instances")
    if progress:
        progress.log(
            f"loaded {split_name}: instances={len(records)} "
            f"docs={len({item.document_id for item in records})}"
        )
    return tuple(records)


def select_documents(
    records: Sequence[InstanceRecord],
    *,
    limit_docs: int = 0,
    seed: int = 42,
) -> tuple[InstanceRecord, ...]:
    if limit_docs <= 0:
        return tuple(records)
    scored: dict[str, tuple[int, str]] = {}
    for item in records:
        n_pos = len(item.positives)
        prev = scored.get(item.document_id, (0, item.document_id))
        scored[item.document_id] = (max(prev[0], n_pos), item.document_id)
    ranked = sorted(scored.values(), key=lambda item: (-item[0], item[1]))
    chosen = {doc_id for _, doc_id in ranked[: int(limit_docs)]}
    del seed
    return tuple(item for item in records if item.document_id in chosen)


def filter_answerable(
    records: Sequence[InstanceRecord],
    *,
    include_unanswerable: bool,
) -> tuple[InstanceRecord, ...]:
    if include_unanswerable:
        return tuple(records)
    return tuple(item for item in records if item.positives)
