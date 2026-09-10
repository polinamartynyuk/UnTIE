"""Load V2 snapshot read-only and reconstruct InstanceRecord objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord

from .contract import LOCKED_SPLITS, UNLOCKED_V2_SPLITS
from .io_util import V2_OUTPUT, load_json, write_json, write_jsonl


def records_from_jsonl(path: Path) -> tuple[InstanceRecord, ...]:
    rows: list[InstanceRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        rows.append(instance_from_dict(payload))
    return tuple(rows)


def instance_from_dict(payload: dict[str, Any]) -> InstanceRecord:
    chunks = tuple(ChunkRecord(**item) for item in payload["chunks"])
    spans = tuple(GoldSpan(**item) for item in payload.get("gold_spans") or ())
    return InstanceRecord(
        document_id=payload["document_id"],
        question_id=payload["question_id"],
        aspect=payload["aspect"],
        question=payload["question"],
        gold_answers=tuple(payload["gold_answers"]),
        split=payload["split"],
        chunks=chunks,
        gold_spans=spans,
        instance_status=payload.get("instance_status", "ok"),
        source_split=payload.get("source_split", payload["split"]),
    )


def snapshot_v2_data(v2_dir: Path, dest_dir: Path) -> dict[str, Any]:
    """Copy V2 data files into V3 without mutating V2."""
    src = Path(v2_dir) / "data"
    dataset = src / "dataset.jsonl"
    if not dataset.exists():
        raise FileNotFoundError(
            f"V2 dataset missing at {dataset}. Run V2 first or use --smoke."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("dataset.jsonl", "hn_quality_flags.jsonl", "splits_v2.json", "data_summary.json", "candidate_sets.jsonl"):
        path = src / name
        if not path.exists():
            continue
        target = dest_dir / name
        target.write_bytes(path.read_bytes())
        copied.append(name)
    if str(Path(v2_dir).resolve()) == str(V2_OUTPUT.resolve()) and dest_dir.resolve() == src.resolve():
        raise ValueError("refusing to snapshot V2 data onto itself")
    write_json(
        dest_dir / "v2_snapshot_meta.json",
        {
            "source": str(src),
            "copied": copied,
            "read_only_source": True,
        },
    )
    return {"copied": copied, "source": str(src)}


def split_unlocked(records: Sequence[InstanceRecord]) -> tuple[InstanceRecord, ...]:
    return tuple(item for item in records if item.split in UNLOCKED_V2_SPLITS)


def split_locked(records: Sequence[InstanceRecord]) -> tuple[InstanceRecord, ...]:
    return tuple(item for item in records if item.split in LOCKED_SPLITS)


def by_document(records: Sequence[InstanceRecord]) -> dict[str, list[InstanceRecord]]:
    grouped: dict[str, list[InstanceRecord]] = {}
    for item in records:
        grouped.setdefault(item.document_id, []).append(item)
    return grouped


def synthetic_records(*, n_docs: int = 10, seed: int = 42) -> tuple[InstanceRecord, ...]:
    """Tiny labeled universe for smoke / unit tests. Not SciREX."""
    del seed
    aspects = ("task", "method", "material", "metric")
    rows: list[InstanceRecord] = []
    for doc_i in range(int(n_docs)):
        document_id = f"smoke-doc-{doc_i:03d}"
        v2_split = ("train", "calibration", "development")[doc_i % 3]
        for aspect in aspects[: 2 + (doc_i % 3)]:
            question = f"What is the {aspect} of this paper?"
            gold = f"{aspect}-value-{doc_i}"
            texts = [
                f"We propose a {gold} system for the target study.",
                f"Previous work uses another {aspect} called distractor-{doc_i}.",
                f"The baseline {aspect} is not-{gold}.",
                "Unrelated background sentence about citations.",
            ]
            labels = ("positive", "ambiguous", "hard_negative", "easy_negative")
            primaries = (
                "robust_positive",
                "gold_supporting_robust_negative",
                "hard_negative",
                "easy_negative",
            )
            chunks = []
            cursor = 0
            for chunk_id, (text, label, primary) in enumerate(zip(texts, labels, primaries)):
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=text,
                        label=label,
                        primary_class=primary,
                        position=chunk_id,
                        section_id="experiments" if chunk_id == 0 else "related",
                        section_title="Experiments" if chunk_id == 0 else "Related Work",
                        char_start=cursor,
                        char_end=cursor + len(text),
                        token_start=chunk_id * 8,
                        token_end=chunk_id * 8 + len(text.split()),
                        gold_supporting=label == "positive",
                        previous_chunk_id=None if chunk_id == 0 else chunk_id - 1,
                        next_chunk_id=chunk_id + 1 if chunk_id + 1 < len(texts) else None,
                    )
                )
                cursor += len(text) + 1
            rows.append(
                InstanceRecord(
                    document_id=document_id,
                    question_id=f"{document_id}::{aspect}",
                    aspect=aspect,
                    question=question,
                    gold_answers=(gold,),
                    split=v2_split,
                    chunks=tuple(chunks),
                    gold_spans=(
                        GoldSpan(
                            chunk_id=0,
                            start_char=chunks[0].text.find(gold),
                            end_char=chunks[0].text.find(gold) + len(gold),
                            text=gold,
                        ),
                    ),
                    source_split="train",
                )
            )
    return tuple(rows)


def write_records(path: Path, records: Sequence[InstanceRecord]) -> None:
    write_jsonl(path, [item.to_dict() for item in records])
