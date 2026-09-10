"""Cached frozen-reader calls. qa_span_score is a confidence proxy, not extractability."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from experiments.qa_chunk_selection.multipassage import (
    PassageLogits,
    encode_passage,
    global_best_span,
    independent_best_span,
)

from .cache import FeatureCache, cache_key, text_hash
from .schemas import RuntimeChunk, RuntimeInstance, source_offsets


def qa_cache_parts(question: str, chunk: RuntimeChunk, *, model_id: str, inference_hash: str) -> dict[str, str]:
    return {
        "document_chunk_id": str(chunk.chunk_id),
        "question_text_hash": text_hash(question),
        "chunk_text_hash": text_hash(chunk.text),
        "model_identifier": model_id,
        "inference_config_hash": inference_hash,
    }


def answer_chunk(
    answer_fn: Callable[[str, str], dict[str, Any]],
    question: str,
    chunk: RuntimeChunk,
    *,
    cache: FeatureCache | None = None,
    model_id: str = "qa",
    inference_hash: str = "default",
) -> dict[str, Any]:
    key = cache_key(**qa_cache_parts(question, chunk, model_id=model_id, inference_hash=inference_hash))
    if cache is not None:
        hit = cache.get("qa", key)
        if hit is not None:
            hit["cache_hit"] = True
            return hit
    result = dict(answer_fn(question, chunk.text))
    local_start = int(result.get("start") or 0)
    local_end = int(result.get("end") or 0)
    answer = str(result.get("answer") or "")
    src_start, src_end = source_offsets(chunk, local_start, local_end)
    payload = {
        "qa_answer": answer,
        "qa_chunk_id": chunk.chunk_id,
        "qa_chunk_start_char": local_start,
        "qa_chunk_end_char": local_end,
        "source_start_char": src_start,
        "source_end_char": src_end,
        "qa_start_logit": float(result.get("start_logit") or 0.0),
        "qa_end_logit": float(result.get("end_logit") or 0.0),
        "qa_span_score": float(result.get("start_logit") or 0.0) + float(result.get("end_logit") or 0.0),
        "qa_confidence_proxy": float(result.get("start_logit") or 0.0) + float(result.get("end_logit") or 0.0),
        "qa_span_margin": None,
        "reader_tokens": max(1, len(chunk.text.split())),
        "qa_forward_passes": 1,
        "cache_hit": False,
    }
    if cache is not None:
        cache.set("qa", key, payload)
    return payload


def best_two_spans(passage: PassageLogits, *, max_answer_tokens: int = 64) -> tuple[float | None, float | None]:
    context = [index for index, flag in enumerate(passage.token_is_context) if flag]
    if not context:
        return None, None
    context_set = set(context)
    scores: list[float] = []
    for start in context:
        last_end = min(start + max_answer_tokens, context[-1] + 1)
        for end in range(start, last_end):
            if end not in context_set:
                continue
            scores.append(float(passage.start_logits[start] + passage.end_logits[end]))
    if not scores:
        return None, None
    scores.sort(reverse=True)
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    return best, second


def answer_top_m(
    instance: RuntimeInstance,
    ranked_ids: Sequence[int],
    *,
    answer_fn: Callable[[str, str], dict[str, Any]],
    qa_answerer=None,
    m: int = 1,
    cache: FeatureCache | None = None,
    model_id: str = "qa",
    inference_hash: str = "default",
    mode: str = "top1",
) -> dict[str, Any]:
    chosen = [instance.chunk_by_id(int(chunk_id)) for chunk_id in ranked_ids[: max(int(m), 1)]]
    if not chosen:
        return {"qa_answer": "", "used_m": 0, "reader_tokens": 0, "qa_forward_passes": 0}
    if mode == "global" and qa_answerer is not None:
        passages = []
        tokens = 0
        for chunk in chosen:
            # Reuse independent cache by still calling encode; cache is span-level.
            passages.append(encode_passage(qa_answerer, instance.question, _legacy_chunk(chunk)))
            tokens += len(passages[-1].start_logits)
        span = global_best_span(passages)
        if span is None:
            return {
                "qa_answer": "",
                "used_m": len(chosen),
                "reader_tokens": tokens,
                "qa_forward_passes": len(chosen),
                "qa_mode": "global",
            }
        chunk = instance.chunk_by_id(span.chunk_id)
        src_start, src_end = source_offsets(chunk, span.start_char, span.end_char)
        best, second = best_two_spans(passages[0])
        margin = None
        for passage in passages:
            if passage.chunk_id == span.chunk_id:
                best, second = best_two_spans(passage)
                if best is not None and second is not None:
                    margin = float(best - second)
        return {
            "qa_answer": span.text,
            "qa_chunk_id": span.chunk_id,
            "qa_chunk_start_char": span.start_char,
            "qa_chunk_end_char": span.end_char,
            "source_start_char": src_start,
            "source_end_char": src_end,
            "qa_start_logit": span.start_logit,
            "qa_end_logit": span.end_logit,
            "qa_span_score": span.logit_sum,
            "qa_confidence_proxy": span.logit_sum,
            "qa_span_margin": margin,
            "used_m": len(chosen),
            "reader_tokens": tokens,
            "qa_forward_passes": len(chosen),
            "qa_mode": "global",
        }
    best_row = None
    tokens = 0
    forwards = 0
    for chunk in chosen:
        row = answer_chunk(
            answer_fn,
            instance.question,
            chunk,
            cache=cache,
            model_id=model_id,
            inference_hash=inference_hash,
        )
        tokens += int(row.get("reader_tokens") or 0)
        forwards += int(row.get("qa_forward_passes") or 1)
        if best_row is None or float(row.get("qa_span_score") or 0) > float(best_row.get("qa_span_score") or 0):
            best_row = row
    assert best_row is not None
    best_row = dict(best_row)
    best_row["used_m"] = len(chosen)
    best_row["reader_tokens"] = tokens
    best_row["qa_forward_passes"] = forwards
    best_row["qa_mode"] = "independent"
    return best_row


def _legacy_chunk(chunk: RuntimeChunk):
    from experiments.qa_chunk_selection.schema import ChunkRecord

    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        label="easy_negative",
        primary_class="easy_negative",
        position=chunk.position,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        token_start=chunk.token_start,
        token_end=chunk.token_end,
        gold_supporting=False,
        previous_chunk_id=chunk.previous_chunk_id,
        next_chunk_id=chunk.next_chunk_id,
        eligible_for_qa=chunk.eligible_for_qa,
    )


def lexical_overlap(query: str, text: str) -> float:
    import re

    token = re.compile(r"\w+", re.UNICODE)
    query_tokens = set(token.findall(query.casefold()))
    tokens = token.findall(text.casefold())
    if not query_tokens or not tokens:
        return 0.0
    overlap = sum(item in query_tokens for item in tokens)
    return float(overlap) / float(len(tokens))
