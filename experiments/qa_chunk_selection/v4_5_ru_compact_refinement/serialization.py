"""Strict prediction serialization that avoids the historical score-map defect."""

from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction, enrich_prediction
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v4.evaluate import coerce_extractive_qa
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import apply_unicode_rouge

from .io_guard import null_undefined_pairwise, validate_finite


def strict_prediction(item, *, ranked_ids, qa, flags, method, extra=None):
    ids = [int(value) for value in ranked_ids]
    universe = [int(chunk.chunk_id) for chunk in item.chunks]
    if len(ids) != len(set(ids)) or set(ids) != set(universe):
        raise ValueError("ranked_ids must be a unique permutation of candidate chunk ids")
    scores = {cid: float(len(ids) - index) for index, cid in enumerate(ids)}
    runtime = to_runtime(item)
    qa_safe, leftover = coerce_extractive_qa(item, qa, scores=scores, extra_chunk_ids=ids[:1])
    aligned_scores = [scores[int(chunk.chunk_id)] for chunk in runtime.chunks]
    raw = build_raw_prediction(runtime, ranking_method=method, scores=aligned_scores,
                               ranked_ids=ids, qa=qa_safe, extra=extra)
    if leftover and not raw.get("qa_answer"):
        raw["qa_answer"] = leftover
    row = enrich_prediction(raw, item, flags=flags, skip_bertscore=True, ks=(1, 5, 10))
    row = apply_unicode_rouge(row, item.gold_answers)
    row = null_undefined_pairwise(row)
    validate_finite(row)
    if row.get("selected_chunk_id") != row.get("ranked_ids", [None])[0]:
        raise AssertionError("serialized selected chunk and ranking disagree")
    score_top = max(row["scores"], key=row["scores"].get)
    if int(score_top) != int(row["selected_chunk_id"]):
        raise AssertionError("serialized score map and ranking disagree")
    return row
