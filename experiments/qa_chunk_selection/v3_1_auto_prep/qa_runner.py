"""Frozen extractive QA calls. Reuses V2 answer_chunk + FeatureCache. Weights unchanged."""

from __future__ import annotations

from typing import Any, Callable

from experiments.qa_chunk_selection.schema import ChunkRecord
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.qa import answer_chunk, best_two_spans
from experiments.qa_chunk_selection.v2.schemas import chunk_to_runtime


def run_qa(
    answer_fn: Callable[[str, str], dict[str, Any]],
    question: str,
    chunk: ChunkRecord,
    *,
    cache: FeatureCache,
    model_id: str,
    inference_hash: str,
    qa_answerer=None,
) -> dict[str, Any]:
    runtime = chunk_to_runtime(chunk)
    payload = dict(
        answer_chunk(
            answer_fn,
            question,
            runtime,
            cache=cache,
            model_id=model_id,
            inference_hash=inference_hash,
        )
    )
    text = chunk.text or ""
    answer = str(payload.get("qa_answer") or "")
    start = int(payload.get("qa_chunk_start_char") or 0)
    end = int(payload.get("qa_chunk_end_char") or 0)
    valid = True
    flags: list[str] = []
    if answer and answer not in text:
        flags.append("answer_not_in_chunk")
        valid = False
    if start < 0 or end > len(text) or (answer and start >= end):
        flags.append("offset_out_of_range")
        valid = False
    margin = payload.get("qa_span_margin")
    if margin is None and qa_answerer is not None:
        try:
            from experiments.qa_chunk_selection.multipassage import encode_passage

            passage = encode_passage(qa_answerer, question, chunk)
            best, second = best_two_spans(passage)
            if best is not None and second is not None:
                margin = float(best - second)
        except Exception:
            margin = None
    if margin is None:
        margin = 0.0
    return {
        "question": question,
        "qa_predicted_span": answer,
        "qa_span_start": start,
        "qa_span_end": end,
        "qa_raw_span_score": float(payload.get("qa_span_score") or 0.0),
        "qa_margin": float(margin),
        "qa_start_logit": payload.get("qa_start_logit"),
        "qa_end_logit": payload.get("qa_end_logit"),
        "reader_tokens": int(payload.get("reader_tokens") or max(1, len(text.split()))),
        "cache_hit": bool(payload.get("cache_hit")),
        "valid_span": valid,
        "span_flags": flags,
    }
