"""D4-v2 sentence-level evidence. Implemented, not in DEFAULT_STAGES_V2."""

from __future__ import annotations

from typing import Any, Sequence

from experiments.qa_chunk_selection.schema import GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.structure import split_sentences

from .hn_flags import is_training_eligible_hn
from .schemas import RuntimeChunk, source_offsets


SENTENCE_SPLITTER_VERSION = "regex_lookbehind_v1"


def sentence_units(item: InstanceRecord) -> list[dict[str, Any]]:
    units = []
    for chunk in item.chunks:
        for sent_idx, (start, end, text) in enumerate(split_sentences(chunk.text)):
            src_start, src_end = source_offsets(chunk, start, end)
            units.append(
                {
                    "sentence_id": f"{chunk.chunk_id}:{sent_idx}",
                    "chunk_id": chunk.chunk_id,
                    "sentence_index": sent_idx,
                    "text": text,
                    "chunk_start_char": start,
                    "chunk_end_char": end,
                    "source_start_char": src_start,
                    "source_end_char": src_end,
                    "label": "unlabeled",
                    "cross_sentence_gold": False,
                    "chunk_label": chunk.label,
                }
            )
    _label_sentences(units, item.gold_spans, item)
    return units


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return min(a1, b1) > max(a0, b0)


def _contains(outer0: int, outer1: int, inner0: int, inner1: int) -> bool:
    return outer0 <= inner0 and inner1 <= outer1


def _label_sentences(units: Sequence[dict[str, Any]], spans: Sequence[GoldSpan], item: InstanceRecord) -> None:
    by_chunk = {}
    for unit in units:
        by_chunk.setdefault(int(unit["chunk_id"]), []).append(unit)
    for span in spans:
        local_units = by_chunk.get(int(span.chunk_id), [])
        covering = [
            unit
            for unit in local_units
            if _overlaps(int(unit["chunk_start_char"]), int(unit["chunk_end_char"]), span.start_char, span.end_char)
        ]
        if len(covering) > 1:
            for unit in covering:
                unit["cross_sentence_gold"] = True
            continue
        for unit in covering:
            if _contains(int(unit["chunk_start_char"]), int(unit["chunk_end_char"]), span.start_char, span.end_char):
                unit["label"] = "positive"
    flags = {(item.question_id, chunk.chunk_id): chunk for chunk in item.chunks}
    for unit in units:
        chunk = flags.get((item.question_id, int(unit["chunk_id"])))
        if chunk is None:
            continue
        if unit["label"] == "positive" or unit["cross_sentence_gold"]:
            continue
        if chunk.label == "hard_negative":
            # training eligibility is applied by the trainer via hn flags
            unit["label"] = "hn_sentence"


def d4_training_examples(
    item: InstanceRecord,
    flags,
) -> list[dict[str, Any]]:
    examples = []
    for unit in sentence_units(item):
        if unit["cross_sentence_gold"]:
            continue
        if unit["label"] == "positive":
            examples.append({**unit, "y": 1})
        elif unit["label"] == "hn_sentence" and is_training_eligible_hn(flags, item.question_id, int(unit["chunk_id"])):
            examples.append({**unit, "y": 0})
    return examples


def assemble_d4b(units: Sequence[dict[str, Any]], selected_ids: Sequence[str], *, window: int) -> tuple[str, list[dict[str, Any]]]:
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for unit in units:
        by_chunk.setdefault(int(unit["chunk_id"]), []).append(unit)
    keep: set[str] = set(selected_ids)
    if window:
        for sid in list(selected_ids):
            chunk_id, _, sent = sid.partition(":")
            local = by_chunk.get(int(chunk_id), [])
            indexes = [unit["sentence_index"] for unit in local]
            if int(sent) not in indexes:
                continue
            loc = indexes.index(int(sent))
            for offset in range(1, int(window) + 1):
                if loc - offset >= 0:
                    keep.add(str(local[loc - offset]["sentence_id"]))
                if loc + offset < len(local):
                    keep.add(str(local[loc + offset]["sentence_id"]))
    kept = [unit for unit in units if unit["sentence_id"] in keep]
    kept.sort(key=lambda unit: (int(unit["source_start_char"]), int(unit["sentence_index"])))
    pieces = []
    mapping = []
    cursor = 0
    for unit in kept:
        text = str(unit["text"]).strip()
        mapping.append(
            {
                "assembled_start": cursor,
                "assembled_end": cursor + len(text),
                "source_chunk_id": unit["chunk_id"],
                "source_start_char": unit["source_start_char"],
                "source_end_char": unit["source_end_char"],
            }
        )
        pieces.append(text)
        cursor += len(text) + 1
    return " ".join(pieces), mapping


def project_assembled_span(assembled: str, mapping: Sequence[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    text = assembled[int(start) : int(end)]
    for row in mapping:
        if int(row["assembled_start"]) <= start < int(row["assembled_end"]) or int(row["assembled_start"]) < end <= int(row["assembled_end"]):
            local = start - int(row["assembled_start"])
            width = end - start
            src_start = int(row["source_start_char"]) + max(local, 0)
            return {
                "text": text,
                "source_chunk_id": row["source_chunk_id"],
                "source_start_char": src_start,
                "source_end_char": src_start + width,
            }
    return {"text": text, "source_chunk_id": None, "source_start_char": None, "source_end_char": None}
