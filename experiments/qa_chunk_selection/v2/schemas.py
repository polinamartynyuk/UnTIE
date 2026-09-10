"""Gold-free runtime records. Scorers must not see labels or gold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord

from .contract import FORBIDDEN_RUNTIME_FIELDS


@dataclass(frozen=True)
class RuntimeChunk:
    chunk_id: int
    text: str
    position: int
    section_id: str
    section_title: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    previous_chunk_id: int | None
    next_chunk_id: int | None
    eligible_for_qa: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        assert_no_gold_fields(payload)
        return payload


@dataclass(frozen=True)
class RuntimeInstance:
    document_id: str
    question_id: str
    aspect: str
    question: str
    split: str
    chunks: tuple[RuntimeChunk, ...]

    def chunk_by_id(self, chunk_id: int) -> RuntimeChunk:
        for chunk in self.chunks:
            if chunk.chunk_id == int(chunk_id):
                return chunk
        raise KeyError(chunk_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        assert_no_gold_fields(payload)
        return payload


def chunk_to_runtime(chunk: ChunkRecord) -> RuntimeChunk:
    return RuntimeChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        position=chunk.position,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        previous_chunk_id=chunk.previous_chunk_id,
        next_chunk_id=chunk.next_chunk_id,
        eligible_for_qa=chunk.eligible_for_qa,
    )


def to_runtime(instance: InstanceRecord) -> RuntimeInstance:
    return RuntimeInstance(
        document_id=instance.document_id,
        question_id=instance.question_id,
        aspect=instance.aspect,
        question=instance.question,
        split=instance.split,
        chunks=tuple(chunk_to_runtime(chunk) for chunk in instance.chunks),
    )


def assert_no_gold_fields(payload: Any, *, path: str = "$") -> None:
    if isinstance(payload, dict):
        leaked = sorted(FORBIDDEN_RUNTIME_FIELDS & set(payload))
        if leaked:
            raise ValueError(f"gold/label field leaked into runtime artifact at {path}: {leaked}")
        for key, value in payload.items():
            assert_no_gold_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_no_gold_fields(value, path=f"{path}[{index}]")


def source_offsets(chunk: RuntimeChunk | ChunkRecord, local_start: int, local_end: int) -> tuple[int, int]:
    start = int(chunk.char_start) + max(int(local_start), 0)
    end = int(chunk.char_start) + max(int(local_end), int(local_start))
    return start, end


def assert_extractive_trace(
    *,
    chunk_text: str,
    answer: str,
    local_start: int,
    local_end: int,
    source_start: int,
    source_end: int,
    chunk_char_start: int,
) -> None:
    if not str(answer or ""):
        return
    extracted = chunk_text[int(local_start) : int(local_end)]
    if extracted != answer:
        raise ValueError("chunk-local offsets do not reconstruct answer_text")
    reconstructed_start = int(chunk_char_start) + int(local_start)
    reconstructed_end = int(chunk_char_start) + int(local_end)
    if reconstructed_start != int(source_start) or reconstructed_end != int(source_end):
        raise ValueError("source offsets are not chunk_char_start + local offsets")


def rank_ids(chunks: Sequence[RuntimeChunk], scores: Sequence[float]) -> list[int]:
    if len(chunks) != len(scores):
        raise ValueError("scores must align with chunks")
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )
    return [int(chunks[index].chunk_id) for index in order]
