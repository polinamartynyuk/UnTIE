"""Local context for annotators. Reuses V3 neighbor-sentence helpers."""

from __future__ import annotations

from typing import Any

from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.structure import split_sentences
from experiments.qa_chunk_selection.v3.structure import neighbor_sentences


def chunk_sentences(text: str) -> list[dict[str, Any]]:
    rows = []
    for index, start, sent in split_sentences(text or ""):
        rows.append({"sentence_id": int(index), "start_char": int(start), "text": sent})
    return rows


def local_context(item: InstanceRecord, chunk: ChunkRecord) -> dict[str, Any]:
    prev_chunk = None
    next_chunk = None
    for other in item.chunks:
        if other.chunk_id == chunk.previous_chunk_id:
            prev_chunk = other
        if other.chunk_id == chunk.next_chunk_id:
            next_chunk = other
    neighbors = neighbor_sentences(
        chunk.text,
        prev_chunk.text if prev_chunk is not None else None,
        next_chunk.text if next_chunk is not None else None,
    )
    return {
        "section_title": chunk.section_title or "",
        "previous_sentence": neighbors["previous_sentence"],
        "next_sentence": neighbors["next_sentence"],
        "previous_chunk_id": chunk.previous_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "previous_chunk_text": (prev_chunk.text if prev_chunk is not None else ""),
        "next_chunk_text": (next_chunk.text if next_chunk is not None else ""),
        "chunk_sentences": chunk_sentences(chunk.text),
        "position": chunk.position,
        "chunk_length": len(chunk.text),
    }
