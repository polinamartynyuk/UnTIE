"""Document-level splits and frozen candidate sets. Gold is not used at inference."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Sequence

from .schema import ChunkRecord, InstanceRecord


def document_split_payload(records: Sequence[InstanceRecord]) -> dict[str, list[str]]:
    by_split: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for item in records:
        key = (item.split, item.document_id)
        if key in seen:
            continue
        seen.add(key)
        by_split[item.split].append(item.document_id)
    for split in by_split:
        by_split[split] = sorted(by_split[split])
    overlap = set(by_split.get("train", [])) & set(by_split.get("dev", []))
    if overlap:
        raise ValueError(f"document leakage across train/dev: {sorted(overlap)[:5]}")
    return {split: ids for split, ids in by_split.items()}


def _stable_int(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def controlled_chunks(
    instance: InstanceRecord,
    *,
    n_easy: int,
    seed: int,
) -> tuple[ChunkRecord, ...]:
    positives = list(instance.positives)
    hard = list(instance.hard_negatives)
    easy = sorted(instance.easy_negatives, key=lambda item: item.chunk_id)
    if n_easy > 0 and len(easy) > n_easy:
        ranked = sorted(
            easy,
            key=lambda item: (
                _stable_int(seed, instance.question_id, item.chunk_id),
                item.chunk_id,
            ),
        )
        easy = ranked[:n_easy]
    selected = positives + hard + easy
    selected.sort(key=lambda item: item.position)
    return tuple(selected)


def freeze_candidate_sets(
    records: Sequence[InstanceRecord],
    *,
    mode: str,
    n_easy: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in records:
        if mode == "all":
            chunks = item.chunks
        elif mode == "controlled":
            chunks = controlled_chunks(item, n_easy=n_easy, seed=seed)
        else:
            raise ValueError(f"unknown candidate mode: {mode}")
        rows.append(
            {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "mode": mode,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "n_positive": sum(chunk.label == "positive" for chunk in chunks),
                "n_hard_negative": sum(chunk.label == "hard_negative" for chunk in chunks),
                "n_easy_negative": sum(chunk.label == "easy_negative" for chunk in chunks),
                "n_ambiguous": sum(chunk.label == "ambiguous" for chunk in chunks),
            }
        )
    return rows


def apply_candidate_set(
    instance: InstanceRecord,
    chunk_ids: Sequence[int],
) -> InstanceRecord:
    wanted = {int(item) for item in chunk_ids}
    chunks = tuple(chunk for chunk in instance.chunks if chunk.chunk_id in wanted)
    spans = tuple(span for span in instance.gold_spans if span.chunk_id in wanted)
    return InstanceRecord(
        document_id=instance.document_id,
        question_id=instance.question_id,
        aspect=instance.aspect,
        question=instance.question,
        gold_answers=instance.gold_answers,
        split=instance.split,
        chunks=chunks,
        gold_spans=spans,
        instance_status=instance.instance_status,
        source_split=instance.source_split,
    )
