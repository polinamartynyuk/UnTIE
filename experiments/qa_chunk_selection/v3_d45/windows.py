"""Gold-localized diagnostic windows. Gold is not a runtime selector."""

from __future__ import annotations

from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord
from experiments.qa_chunk_selection.structure import split_sentences

WINDOW_TYPES = ("full_chunk", "gold_sentence", "minimal_gold_window", "sentence_pm1")


def locate_gold(chunk: ChunkRecord, item: InstanceRecord) -> tuple[int, int] | None:
    for span in item.gold_spans or ():
        if int(span.chunk_id) != int(chunk.chunk_id):
            continue
        start, end = int(span.start_char), int(span.end_char)
        if 0 <= start < end <= len(chunk.text):
            return start, end
    text = chunk.text or ""
    folded = text.casefold()
    for gold in item.gold_answers or ():
        needle = str(gold or "").strip()
        if not needle:
            continue
        pos = folded.find(needle.casefold())
        if pos >= 0:
            return pos, pos + len(needle)
    return None


def covering_sentence(text: str, start: int, end: int) -> tuple[int, int, str]:
    sentences = split_sentences(text)
    for s0, s1, sent in sentences:
        if min(s1, end) > max(s0, start):
            return s0, s1, sent
    start_i = max(0, int(start))
    end_i = min(len(text), max(start_i + 1, int(end)))
    return start_i, end_i, text[start_i:end_i]


def sentence_pm1(text: str, start: int, end: int) -> tuple[int, int, str]:
    sentences = split_sentences(text)
    if not sentences:
        return 0, len(text), text
    index = 0
    for i, (s0, s1, _sent) in enumerate(sentences):
        if min(s1, end) > max(s0, start):
            index = i
            break
    lo = max(0, index - 1)
    hi = min(len(sentences) - 1, index + 1)
    a0, _, _ = sentences[lo]
    _, b1, _ = sentences[hi]
    return a0, b1, text[a0:b1]


def minimal_window(text: str, start: int, end: int, *, pad: int = 8) -> tuple[int, int, str]:
    a0 = max(0, int(start) - int(pad))
    b1 = min(len(text), int(end) + int(pad))
    if b1 <= a0:
        return 0, len(text), text
    return a0, b1, text[a0:b1]


def windows_for_positive(item: InstanceRecord, chunk: ChunkRecord) -> dict[str, dict[str, Any]]:
    text = chunk.text or ""
    located = locate_gold(chunk, item)
    payload: dict[str, dict[str, Any]] = {
        "full_chunk": {"text": text, "local_start": 0, "local_end": len(text), "localized": True},
    }
    if located is None:
        for name in ("gold_sentence", "minimal_gold_window", "sentence_pm1"):
            payload[name] = {"text": "", "local_start": None, "local_end": None, "localized": False}
        return payload
    g0, g1 = located
    s0, s1, sent = covering_sentence(text, g0, g1)
    m0, m1, mini = minimal_window(text, g0, g1)
    p0, p1, plus = sentence_pm1(text, s0, s1)
    payload["gold_sentence"] = {"text": sent, "local_start": s0, "local_end": s1, "localized": True}
    payload["minimal_gold_window"] = {"text": mini, "local_start": m0, "local_end": m1, "localized": True}
    payload["sentence_pm1"] = {"text": plus, "local_start": p0, "local_end": p1, "localized": True}
    return payload


def as_window_chunk(chunk: ChunkRecord, window: dict[str, Any]) -> ChunkRecord:
    text = str(window.get("text") or "")
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        text=text,
        label=chunk.label,
        primary_class=chunk.primary_class,
        position=chunk.position,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
        char_start=int(chunk.char_start) + int(window.get("local_start") or 0),
        char_end=int(chunk.char_start) + int(window.get("local_start") or 0) + len(text),
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        gold_supporting=chunk.gold_supporting,
        previous_chunk_id=chunk.previous_chunk_id,
        next_chunk_id=chunk.next_chunk_id,
    )
