"""Automatic Phase 1 sanity checks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .assign import InstanceResult
from .config import ALLOWED_ASPECTS, ALLOWED_STATUS, CONTAINMENT_MAX_EXTRA_TOKENS, POSITIVE_MATCH_METHOD
from .match import classify_match


def _identity_key(value: str) -> str:
    from untie.aspect_qa.normalize import identity_key as _key

    return _key(value)


def check_instances(
    results: Sequence[InstanceResult],
    *,
    source_chunks: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    match_method: str | None = None,
    max_extra_tokens: int | None = None,
) -> list[str]:
    errors: list[str] = []
    method = match_method or POSITIVE_MATCH_METHOD
    extra_cap = CONTAINMENT_MAX_EXTRA_TOKENS if max_extra_tokens is None else int(max_extra_tokens)
    seen_classes: dict[tuple[str, str, int], str] = {}
    for result in results:
        if result.aspect not in ALLOWED_ASPECTS:
            errors.append(f"illegal aspect {result.aspect}")
        if result.status not in ALLOWED_STATUS:
            errors.append(f"illegal status {result.qa_instance_id} {result.status}")
        if result.aspect == "score":
            errors.append("score aspect in Phase 1")
        if not result.extractable_gold_answers:
            errors.append(f"empty extractable gold {result.qa_instance_id}")
        if result.enrichment_mode != "E0":
            errors.append(f"enrichment_mode {result.enrichment_mode}")
        used_classes: dict[int, str] = {}
        for chunk in result.chunks:
            key = (result.document_id, result.aspect, chunk.chunk_id)
            if key in seen_classes and seen_classes[key] != chunk.primary_class:
                errors.append(f"conflicting class {key}")
            seen_classes[key] = chunk.primary_class
            if chunk.chunk_id in used_classes and used_classes[chunk.chunk_id] != chunk.primary_class:
                errors.append(f"exclusive class clash {result.qa_instance_id} {chunk.chunk_id}")
            used_classes[chunk.chunk_id] = chunk.primary_class
            if chunk.qa_positive:
                matched = classify_match(
                    chunk.predicted_answer,
                    result.extractable_gold_answers,
                    method=method,
                    max_extra_tokens=extra_cap,
                )
                if not matched.matched:
                    errors.append(f"qa_positive without extractable match {result.qa_instance_id}:{chunk.chunk_id}")
                elif matched.extra_tokens is not None and matched.extra_tokens > extra_cap:
                    errors.append(
                        f"qa_positive extra_tokens>{extra_cap} {result.qa_instance_id}:{chunk.chunk_id}"
                    )
                if not chunk.matched_gold_answer:
                    errors.append(f"qa_positive missing matched_gold_answer {result.qa_instance_id}:{chunk.chunk_id}")
                elif _identity_key(chunk.matched_gold_answer) not in {
                    _identity_key(item) for item in result.extractable_gold_answers
                }:
                    errors.append(f"matched gold outside extractable set {result.qa_instance_id}")
            if chunk.primary_class == "hard_negative":
                if chunk.qa_positive or chunk.gold_supporting:
                    errors.append(f"hard_negative polluted {result.qa_instance_id}:{chunk.chunk_id}")
                if chunk.same_cluster_as_positive and result.fn_filter_status == "applied":
                    errors.append(f"FN left in hard_negative {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "evidence_positive_qa_failed":
                if not chunk.gold_supporting or chunk.qa_positive:
                    errors.append(f"bad evidence_failed {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "false_negative_candidate":
                if chunk.qa_positive or chunk.gold_supporting:
                    errors.append(f"FN polluted {result.qa_instance_id}:{chunk.chunk_id}")
            if source_chunks is not None:
                source = source_chunks.get((result.document_id, chunk.chunk_id))
                if source is None:
                    errors.append(f"unknown chunk_id {result.document_id}:{chunk.chunk_id}")
                else:
                    if int(source["char_start"]) != int(chunk.char_start) or int(source["char_end"]) != int(chunk.char_end):
                        errors.append(f"offset drift {result.document_id}:{chunk.chunk_id}")
                    if str(source["chunk_text"]) != str(chunk.text):
                        errors.append(f"chunk text drift {result.document_id}:{chunk.chunk_id}")
        if result.instance_status == "no_qa_positive" and result.hard_negative_ids:
            errors.append(f"HN set built without qa_positive {result.qa_instance_id}")
        hn = set(result.hard_negative_ids)
        if hn & set(result.fn_ids):
            errors.append(f"FN in HN {result.qa_instance_id}")
        if hn & set(result.positive_ids):
            errors.append(f"positive in HN {result.qa_instance_id}")
        if hn & set(result.evidence_failed_ids):
            errors.append(f"evidence_failed in HN {result.qa_instance_id}")
    return errors
