"""Sanity checks. Gold sets must match preprocessing; HN must be robust_negative without gold."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.phase1_scirex.match import classify_match
from experiments.phase1_scirex.tables import as_list
from untie.aspect_qa.config import ASPECT_QUESTIONS
from untie.aspect_qa.normalize import identity_key

from .assign import InstanceResult
from .config import ALLOWED_ASPECTS, ALLOWED_STATUS
from .questions import questions_for


def check_instances(
    results: Sequence[InstanceResult],
    *,
    source_chunks: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    preprocessing_gold: Mapping[tuple[str, str], Sequence[str]] | None = None,
    match_method: str,
    max_extra_tokens: int,
    n_questions: int,
) -> list[str]:
    errors: list[str] = []
    for result in results:
        if result.aspect not in ALLOWED_ASPECTS:
            errors.append(f"illegal aspect {result.aspect}")
        if result.status not in ALLOWED_STATUS:
            errors.append(f"illegal status {result.qa_instance_id}")
        if result.n_questions != n_questions:
            errors.append(f"n_questions {result.n_questions} != {n_questions}")
        expected_questions = questions_for(result.aspect, n_questions)
        if tuple(result.questions) != expected_questions:
            errors.append(f"question variants drifted {result.qa_instance_id}")
        if result.canonical_question != ASPECT_QUESTIONS[result.aspect]:
            errors.append(f"canonical baseline drifted {result.qa_instance_id}")
        if not result.extractable_gold_answers:
            errors.append(f"empty extractable gold {result.qa_instance_id}")
        if preprocessing_gold is not None:
            source_gold = [
                identity_key(item)
                for item in preprocessing_gold.get((result.document_id, result.aspect), [])
            ]
            got = [identity_key(item) for item in result.extractable_gold_answers]
            if source_gold and got != source_gold:
                errors.append(f"extractable gold changed {result.qa_instance_id}")
        used: dict[int, str] = {}
        for chunk in result.chunks:
            if chunk.chunk_id in used and used[chunk.chunk_id] != chunk.primary_class:
                errors.append(f"class clash {result.qa_instance_id} {chunk.chunk_id}")
            used[chunk.chunk_id] = chunk.primary_class
            if chunk.n_questions != n_questions:
                errors.append(f"chunk n_questions {result.qa_instance_id}:{chunk.chunk_id}")
            expected_cons = chunk.n_correct / float(n_questions)
            if abs(chunk.qa_consistency - expected_cons) > 1e-9:
                errors.append(f"QAConsistency formula {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.any_correct != (chunk.n_correct > 0):
                errors.append(f"any_correct diagnostic {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "robust_positive" and chunk.consistency_class != "robust_positive":
                errors.append(f"robust_positive without threshold {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "question_sensitive" and chunk.n_correct == n_questions:
                errors.append(f"full-consistency marked sensitive {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "hard_negative":
                if chunk.gold_supporting or chunk.qa_consistency > 0 or chunk.any_correct:
                    errors.append(f"HN polluted {result.qa_instance_id}:{chunk.chunk_id}")
                if chunk.consistency_class != "robust_negative":
                    errors.append(f"HN not robust_negative {result.qa_instance_id}:{chunk.chunk_id}")
            if chunk.primary_class == "gold_supporting_robust_negative":
                if not chunk.gold_supporting or chunk.qa_consistency > 0:
                    errors.append(f"bad gold robust_negative {result.qa_instance_id}:{chunk.chunk_id}")
            for pred in chunk.predictions:
                matched = classify_match(
                    pred.predicted_answer,
                    result.extractable_gold_answers,
                    method=match_method,
                    max_extra_tokens=max_extra_tokens,
                )
                if bool(matched.matched) != bool(pred.qa_correct):
                    errors.append(
                        f"qa_correct mismatch {result.qa_instance_id}:{chunk.chunk_id}:q{pred.question_id}"
                    )
            if source_chunks is not None:
                source = source_chunks.get((result.document_id, chunk.chunk_id))
                if source is None:
                    errors.append(f"unknown chunk {result.document_id}:{chunk.chunk_id}")
                else:
                    if int(source["char_start"]) != int(chunk.char_start) or int(source["char_end"]) != int(chunk.char_end):
                        errors.append(f"offset drift {result.document_id}:{chunk.chunk_id}")
                    if str(source["chunk_text"]) != str(chunk.text):
                        errors.append(f"chunk text drift {result.document_id}:{chunk.chunk_id}")
        if result.instance_status == "no_robust_positive" and result.hard_negative_ids:
            errors.append(f"HN without robust_positive {result.qa_instance_id}")
        hn = set(result.hard_negative_ids)
        if hn & set(result.robust_positive_ids):
            errors.append(f"positive in HN {result.qa_instance_id}")
        if hn & set(result.question_sensitive_ids):
            errors.append(f"question_sensitive in HN {result.qa_instance_id}")
        if hn & set(result.gold_failed_ids):
            errors.append(f"gold_supporting in HN {result.qa_instance_id}")
        if hn & set(result.fn_ids):
            errors.append(f"FN in HN {result.qa_instance_id}")
    return errors
