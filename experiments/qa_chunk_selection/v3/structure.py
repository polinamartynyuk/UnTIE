"""Discourse / local structural context. Not the old D2 section-title overlap score."""

from __future__ import annotations

import re
from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.structure import split_sentences
from experiments.qa_chunk_selection.v2.schemas import RuntimeChunk, RuntimeInstance

FIRST_PERSON = re.compile(
    r"\b(we use|we evaluate|our model|our experiments|we train|we propose|in this paper|this work)\b",
    re.I,
)
COMPARATIVE = re.compile(
    r"\b(previous work|related work|baseline|compared with|compared to|et al\.?|reported|uses)\b",
    re.I,
)
CITATION = re.compile(r"(\[\d+\]|\(\d{4}\)|\bet al\b)", re.I)


def neighbor_sentences(text: str, prev_text: str | None, next_text: str | None) -> dict[str, str]:
    prev_sents = split_sentences(prev_text or "")
    next_sents = split_sentences(next_text or "")
    own = split_sentences(text)
    prev = prev_sents[-1][2] if prev_sents else ""
    nxt = next_sents[0][2] if next_sents else (own[1][2] if len(own) > 1 else "")
    return {"previous_sentence": prev, "next_sentence": nxt}


def discourse_features(text: str, section_title: str, position: int, n_chunks: int) -> dict[str, Any]:
    return {
        "section_title": section_title or "",
        "section_position": float(position) / max(int(n_chunks) - 1, 1),
        "has_first_person": float(bool(FIRST_PERSON.search(text))),
        "has_comparative": float(bool(COMPARATIVE.search(text))),
        "has_citation": float(bool(CITATION.search(text))),
    }


def wrapped_chunk_text(instance: RuntimeInstance | InstanceRecord, chunk: RuntimeChunk | Any) -> str:
    chunks = list(instance.chunks)
    n = len(chunks)
    prev = None
    nxt = None
    for other in chunks:
        if other.chunk_id == getattr(chunk, "previous_chunk_id", None):
            prev = other.text
        if other.chunk_id == getattr(chunk, "next_chunk_id", None):
            nxt = other.text
    neighbors = neighbor_sentences(chunk.text, prev, nxt)
    title = getattr(chunk, "section_title", "") or ""
    return (
        f"[SECTION] {title}\n"
        f"[PREV] {neighbors['previous_sentence']}\n"
        f"[CHUNK] {chunk.text}\n"
        f"[NEXT] {neighbors['next_sentence']}"
    )


def structural_score_fn(base_predict, *, wrapped: bool):
    def score_fn(runtime: RuntimeInstance) -> list[float]:
        pairs = []
        for chunk in runtime.chunks:
            text = wrapped_chunk_text(runtime, chunk) if wrapped else chunk.text
            pairs.append((runtime.question, text))
        return list(base_predict(pairs))

    return score_fn


def error_analysis_categories(text: str) -> list[str]:
    tags = []
    if FIRST_PERSON.search(text):
        tags.append("first_person")
    if COMPARATIVE.search(text):
        tags.append("comparative")
    if CITATION.search(text):
        tags.append("citation")
    return tags or ["none"]
