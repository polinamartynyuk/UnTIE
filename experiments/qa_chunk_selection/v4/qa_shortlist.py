"""Frozen QA on B0 top-K. Gold is not used at call time."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.qa import answer_chunk
from experiments.qa_chunk_selection.v2.schemas import to_runtime

from .answers import normalize_predicted_answer


def answer_one(
    item: InstanceRecord,
    chunk_id: int,
    answer_fn: Callable[[str, str], dict[str, Any]],
    *,
    cache,
    model_id: str,
    inference_hash: str,
    question: str | None = None,
) -> dict[str, Any]:
    runtime = to_runtime(item)
    chunk = next(chunk for chunk in runtime.chunks if int(chunk.chunk_id) == int(chunk_id))
    payload = answer_chunk(
        answer_fn,
        question if question is not None else item.question,
        chunk,
        cache=cache,
        model_id=model_id,
        inference_hash=inference_hash,
    )
    payload["normalized_answer"] = normalize_predicted_answer(str(payload.get("qa_answer") or ""))
    payload["chunk_id"] = int(chunk_id)
    payload["question_id"] = item.question_id
    raw = float(payload.get("qa_span_score") or 0.0)
    margin = payload.get("qa_span_margin")
    payload["qa_raw_score"] = raw
    payload["qa_margin"] = float(margin) if margin is not None else 0.0
    return payload


def relative_scores(rows: Mapping[int, Mapping[str, Any]], field: str) -> dict[int, float]:
    values = {int(cid): float(row.get(field) or 0.0) for cid, row in rows.items()}
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = max(hi - lo, 1e-6)
    return {cid: (value - lo) / span for cid, value in values.items()}
