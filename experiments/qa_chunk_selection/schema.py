"""In-memory records for one (document_id, question_id) instance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_LABELS = ("positive", "hard_negative", "easy_negative", "ambiguous")


@dataclass(frozen=True)
class GoldSpan:
    chunk_id: int
    start_char: int
    end_char: int
    text: str
    document_start_char: int = 0
    document_end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    text: str
    label: str
    primary_class: str
    position: int
    section_id: str
    section_title: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    gold_supporting: bool
    previous_chunk_id: int | None
    next_chunk_id: int | None
    eligible_for_qa: bool = True

    def __post_init__(self) -> None:
        if self.label not in EXPERIMENT_LABELS:
            raise ValueError(f"unknown experiment label: {self.label}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstanceRecord:
    document_id: str
    question_id: str
    aspect: str
    question: str
    gold_answers: tuple[str, ...]
    split: str
    chunks: tuple[ChunkRecord, ...]
    gold_spans: tuple[GoldSpan, ...] = ()
    instance_status: str = "ok"
    source_split: str = "train"

    @property
    def positives(self) -> tuple[ChunkRecord, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.label == "positive")

    @property
    def hard_negatives(self) -> tuple[ChunkRecord, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.label == "hard_negative")

    @property
    def easy_negatives(self) -> tuple[ChunkRecord, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.label == "easy_negative")

    def chunk_by_id(self, chunk_id: int) -> ChunkRecord:
        for chunk in self.chunks:
            if chunk.chunk_id == int(chunk_id):
                return chunk
        raise KeyError(chunk_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_answers"] = list(self.gold_answers)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        payload["gold_spans"] = [span.to_dict() for span in self.gold_spans]
        return payload


def instances_by_split(
    records: Sequence[InstanceRecord], split: str
) -> tuple[InstanceRecord, ...]:
    return tuple(item for item in records if item.split == split)


def document_ids_of(records: Iterable[InstanceRecord]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in records:
        if item.document_id not in seen:
            seen.append(item.document_id)
    return tuple(seen)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        items = list(value)
    except TypeError:
        return [str(value)]
    return [str(item) for item in items if item is not None and str(item).strip()]


def require_mapping(row: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise KeyError(f"missing fields: {missing}")
