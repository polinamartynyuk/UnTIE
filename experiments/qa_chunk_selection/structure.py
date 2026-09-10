"""D2–D5 structure / evidence helpers. Gold is not used to pick neighbors."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .metrics import normalized_answer
from .ranking import cosine_scores, min_max
from .schema import ChunkRecord, InstanceRecord


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    cleaned = str(text or "")
    if not cleaned.strip():
        return []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    parts = _SENTENCE.split(cleaned)
    if len(parts) == 1:
        return [(0, len(cleaned), cleaned)]
    for part in parts:
        start = cleaned.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        if part.strip():
            spans.append((start, end, part))
        cursor = end
    return spans


def structural_mix(
    retrieval: Sequence[float],
    answerability: Sequence[float],
    section: Sequence[float],
    *,
    w_retrieval: float = 0.5,
    w_answerability: float = 0.3,
    w_section: float = 0.2,
) -> list[float]:
    return [
        w_retrieval * a + w_answerability * b + w_section * c
        for a, b, c in zip(min_max(retrieval), min_max(answerability), min_max(section))
    ]


def expand_neighbors(
    instance: InstanceRecord,
    seed_ids: Sequence[int],
    *,
    encode,
    max_chunks: int,
    max_neighbors: int,
    hops: int = 1,
    semantic_threshold: float = 0.35,
) -> list[int]:
    by_id = {chunk.chunk_id: chunk for chunk in instance.chunks}
    selected: list[int] = []
    seen: set[int] = set()
    frontier = [int(item) for item in seed_ids if int(item) in by_id]
    if not frontier:
        frontier = [instance.chunks[0].chunk_id] if instance.chunks else []
    texts = [chunk.text for chunk in instance.chunks]
    vectors = np.asarray(encode(texts), dtype=float) if texts else np.zeros((0, 1))
    id_to_index = {chunk.chunk_id: index for index, chunk in enumerate(instance.chunks)}
    for _ in range(max(int(hops), 1)):
        nxt: list[int] = []
        for chunk_id in frontier:
            chunk = by_id.get(chunk_id)
            if chunk is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            selected.append(chunk_id)
            candidates: list[int] = []
            if chunk.previous_chunk_id is not None:
                candidates.append(chunk.previous_chunk_id)
            if chunk.next_chunk_id is not None:
                candidates.append(chunk.next_chunk_id)
            same = [
                other.chunk_id
                for other in instance.chunks
                if other.section_id == chunk.section_id and other.chunk_id != chunk.chunk_id
            ]
            candidates.extend(same[: max_neighbors])
            if vectors.size and chunk_id in id_to_index:
                src = vectors[id_to_index[chunk_id]]
                sims = cosine_scores(src, vectors)
                ranked = sorted(
                    range(len(instance.chunks)),
                    key=lambda index: -float(sims[index]),
                )
                for index in ranked[1 : max_neighbors + 1]:
                    if sims[index] >= semantic_threshold:
                        candidates.append(instance.chunks[index].chunk_id)
            for cand in candidates:
                if cand not in seen and cand in by_id:
                    nxt.append(cand)
        frontier = nxt
        if len(selected) >= max_chunks:
            break
    for chunk_id in frontier:
        if chunk_id not in seen:
            selected.append(chunk_id)
            seen.add(chunk_id)
        if len(selected) >= max_chunks:
            break
    return selected[: max_chunks]


def sentence_evidence(
    question: str,
    chunks: Sequence[ChunkRecord],
    score_fn,
    *,
    top_sentences: int = 4,
    window: int = 0,
) -> tuple[str, list[dict[str, object]]]:
    units: list[dict[str, object]] = []
    for chunk in chunks:
        sentences = split_sentences(chunk.text)
        for sent_idx, (start, end, text) in enumerate(sentences):
            units.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "sentence_index": sent_idx,
                    "start_char": start,
                    "end_char": end,
                    "text": text,
                    "position": (chunk.position, sent_idx),
                }
            )
    if not units:
        return "", []
    fake_chunks = [
        ChunkRecord(
            chunk_id=index,
            text=str(unit["text"]),
            label="easy_negative",
            primary_class="easy_negative",
            position=index,
            section_id="",
            section_title="",
            char_start=0,
            char_end=len(str(unit["text"])),
            token_start=0,
            token_end=0,
            gold_supporting=False,
            previous_chunk_id=None,
            next_chunk_id=None,
        )
        for index, unit in enumerate(units)
    ]
    scores = score_fn(question, fake_chunks)
    order = sorted(range(len(units)), key=lambda index: -float(scores[index]))
    keep = set(order[: max(int(top_sentences), 1)])
    if window:
        extra: set[int] = set()
        by_chunk: dict[int, list[int]] = {}
        for index, unit in enumerate(units):
            by_chunk.setdefault(int(unit["chunk_id"]), []).append(index)
        for index in list(keep):
            chunk_id = int(units[index]["chunk_id"])
            local = by_chunk[chunk_id]
            loc = local.index(index)
            for offset in range(1, window + 1):
                if loc - offset >= 0:
                    extra.add(local[loc - offset])
                if loc + offset < len(local):
                    extra.add(local[loc + offset])
        keep |= extra
    kept = [units[index] for index in range(len(units)) if index in keep]
    kept.sort(key=lambda item: item["position"])
    mapping = []
    pieces: list[str] = []
    cursor = 0
    for unit in kept:
        text = str(unit["text"]).strip()
        mapping.append(
            {
                "assembled_start": cursor,
                "assembled_end": cursor + len(text),
                "source_chunk_id": unit["chunk_id"],
                "source_start_char": unit["start_char"],
                "source_end_char": unit["end_char"],
            }
        )
        pieces.append(text)
        cursor += len(text) + 1
    return " ".join(pieces), mapping


def aggregate_answers(
    spans: Sequence[dict[str, object]],
    chunks: Sequence[ChunkRecord],
) -> dict[str, object]:
    """Exact-normalized aggregation with overlapping-chunk de-duplication."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    groups: dict[str, list[dict[str, object]]] = {}
    for span in spans:
        key = normalized_answer(str(span.get("text") or ""))
        if not key:
            continue
        groups.setdefault(key, []).append(span)
    best_key = ""
    best_score = float("-inf")
    unique_count = 0
    for key, items in groups.items():
        regions: list[tuple[int, int, int]] = []
        unique_scores: list[float] = []
        for item in items:
            chunk = by_id.get(int(item["chunk_id"]))
            if chunk is None:
                continue
            doc_start = chunk.char_start + int(item.get("start_char") or 0)
            doc_end = chunk.char_start + int(item.get("end_char") or 0)
            overlap = False
            for other_start, other_end, _ in regions:
                inter = min(doc_end, other_end) - max(doc_start, other_start)
                union = max(doc_end, other_end) - min(doc_start, other_start)
                if union > 0 and inter / float(union) >= 0.5:
                    overlap = True
                    break
            if overlap:
                continue
            regions.append((doc_start, doc_end, int(item["chunk_id"])))
            unique_scores.append(float(item.get("score") or 0.0))
        if not unique_scores:
            continue
        shifted = np.asarray(unique_scores, dtype=float)
        offset = float(np.max(shifted))
        score = offset + float(np.log(np.sum(np.exp(shifted - offset))))
        if score > best_score:
            best_score = score
            best_key = key
            unique_count = len(unique_scores)
    text = ""
    source_chunk = None
    if best_key:
        for span in spans:
            if normalized_answer(str(span.get("text") or "")) == best_key:
                text = str(span.get("text") or "")
                source_chunk = span.get("chunk_id")
                break
    return {
        "text": text,
        "normalized": best_key,
        "score": best_score if best_key else 0.0,
        "n_unique_regions": unique_count,
        "source_chunk_id": source_chunk,
    }
