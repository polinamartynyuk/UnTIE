"""D5-v2 aggregation. Does not logsumexp raw QA logits across passages."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import normalized_answer


def unique_supports(spans: Sequence[Mapping[str, Any]], *, iou: float = 0.5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    regions: list[tuple[int, int]] = []
    for span in spans:
        start = int(span.get("source_start_char") or 0)
        end = int(span.get("source_end_char") or 0)
        overlap = False
        for other_start, other_end in regions:
            inter = min(end, other_end) - max(start, other_start)
            union = max(end, other_end) - min(start, other_start)
            if union > 0 and inter / float(union) >= iou:
                overlap = True
                break
        if overlap:
            continue
        regions.append((start, end))
        kept.append(dict(span))
    return kept


def aggregate_d5(
    spans: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    score_field: str = "candidate_score",
    lambda_support: float = 0.0,
) -> dict[str, Any]:
    unique = unique_supports(spans)
    if mode == "D5a":
        best = max(unique, key=lambda row: float(row.get(score_field) or 0.0), default=None)
        if best is None:
            return {"text": "", "mode": mode, "n_unique_regions": 0, "supporting_spans": []}
        return {
            "text": best.get("answer_text") or best.get("text") or "",
            "mode": mode,
            "score": float(best.get(score_field) or 0.0),
            "n_unique_regions": 1,
            "supporting_spans": [best],
            "source_chunk_id": best.get("source_chunk_id") or best.get("chunk_id"),
        }
    groups: dict[str, list[dict[str, Any]]] = {}
    for span in unique:
        key = normalized_answer(str(span.get("answer_text") or span.get("text") or ""))
        if not key:
            continue
        groups.setdefault(key, []).append(span)
    best_key = ""
    best_score = float("-inf")
    best_items: list[dict[str, Any]] = []
    for key, items in groups.items():
        scores = [float(item.get(score_field) or 0.0) for item in items]
        if mode == "D5b":
            score = max(scores) if scores else float("-inf")
        else:
            score = (max(scores) if scores else float("-inf")) + float(lambda_support) * math.log(1.0 + len(items))
        if score > best_score:
            best_score = score
            best_key = key
            best_items = items
    text = ""
    source = None
    if best_items:
        text = str(best_items[0].get("answer_text") or best_items[0].get("text") or "")
        source = best_items[0].get("source_chunk_id") or best_items[0].get("chunk_id")
    return {
        "text": text,
        "normalized": best_key,
        "mode": mode,
        "score": best_score if best_key else 0.0,
        "n_unique_regions": len(best_items),
        "supporting_spans": best_items,
        "source_chunk_id": source,
        "lambda_support": lambda_support,
        "score_field": score_field,
    }
