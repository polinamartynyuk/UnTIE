"""Raw inference vs gold-based scoring. Scorers never receive gold objects."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import (
    coverage_at_k,
    hit_at_k,
    hn_at_k,
    hn_bucket,
    pairwise_pos_gt_hn,
    qa_row,
    reciprocal_rank,
    summarize_ranking,
)
from experiments.qa_chunk_selection.schema import InstanceRecord

from .freeze import assert_eval_split_allowed
from .hn_flags import flag_index
from .io_util import write_json, write_jsonl
from .schemas import RuntimeInstance, assert_extractive_trace, assert_no_gold_fields, rank_ids, source_offsets, to_runtime
from .timing import timed_block

ScoreFn = Callable[[RuntimeInstance], list[float]]
AnswerFn = Callable[[RuntimeInstance, Sequence[int]], dict[str, Any]]


def build_raw_prediction(
    instance: RuntimeInstance,
    *,
    ranking_method: str,
    scores: Sequence[float],
    ranked_ids: Sequence[int],
    qa: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = int(ranked_ids[0]) if ranked_ids else None
    qa_payload = dict(qa or {})
    chunk_id = qa_payload.get("qa_chunk_id", selected)
    local_start = _int_field(qa_payload, "qa_chunk_start_char", "qa_start", default=0)
    local_end = _int_field(qa_payload, "qa_chunk_end_char", "qa_end", default=0)
    source_start = qa_payload.get("source_start_char")
    source_end = qa_payload.get("source_end_char")
    answer = str(qa_payload.get("qa_answer") or "")
    if chunk_id is not None and answer:
        chunk = instance.chunk_by_id(int(chunk_id))
        if source_start is None or source_end is None:
            source_start, source_end = source_offsets(chunk, local_start, local_end)
        assert_extractive_trace(
            chunk_text=chunk.text,
            answer=answer,
            local_start=local_start,
            local_end=local_end,
            source_start=int(source_start),
            source_end=int(source_end),
            chunk_char_start=chunk.char_start,
        )
    mapped = {chunk.chunk_id: float(score) for chunk, score in zip(instance.chunks, scores)}
    row = {
        "document_id": instance.document_id,
        "question_id": instance.question_id,
        "aspect": instance.aspect,
        "question": instance.question,
        "ranking_method": ranking_method,
        "candidate_count": len(instance.chunks),
        "selected_chunk_id": selected,
        "selected_chunk_ids": list(ranked_ids),
        "qa_chunk_id": chunk_id,
        "qa_answer": answer,
        "qa_chunk_start_char": local_start,
        "qa_chunk_end_char": local_end,
        "source_start_char": source_start,
        "source_end_char": source_end,
        "qa_confidence_proxy": qa_payload.get("qa_confidence_proxy"),
        "qa_span_score": qa_payload.get("qa_span_score"),
        "qa_span_margin": qa_payload.get("qa_span_margin"),
        "qa_start_logit": qa_payload.get("qa_start_logit"),
        "qa_end_logit": qa_payload.get("qa_end_logit"),
        "used_m": qa_payload.get("used_m", 1),
        "reader_tokens": qa_payload.get("reader_tokens", 0),
        "qa_forward_passes": qa_payload.get("qa_forward_passes", 0),
        "final_ranking_score": mapped.get(selected, 0.0) if selected is not None else 0.0,
        "ranked_ids": list(ranked_ids),
        "scores": mapped,
    }
    if extra:
        row.update(dict(extra))
    assert_no_gold_fields(row)
    return row


def enrich_prediction(
    raw: Mapping[str, Any],
    gold: InstanceRecord,
    *,
    flags: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    skip_bertscore: bool = True,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    ranked = [int(item) for item in raw.get("ranked_ids") or []]
    scores = {int(key): float(value) for key, value in (raw.get("scores") or {}).items()}
    pos = [chunk.chunk_id for chunk in gold.positives]
    hn_all = [chunk.chunk_id for chunk in gold.hard_negatives]
    hn_clean = [
        chunk.chunk_id
        for chunk in gold.hard_negatives
        if not (flags or {}).get((gold.question_id, chunk.chunk_id), {}).get("hn_gold_string_warning")
    ]
    selected = raw.get("selected_chunk_id")
    selected_label = None
    if selected is not None:
        try:
            selected_label = gold.chunk_by_id(int(selected)).label
        except KeyError:
            selected_label = None
    qa_chunk = raw.get("qa_chunk_id")
    qa_label = None
    qa_primary = None
    if qa_chunk is not None:
        try:
            chunk = gold.chunk_by_id(int(qa_chunk))
            qa_label = chunk.label
            qa_primary = chunk.primary_class
        except KeyError:
            pass
    pair_all, n_all = pairwise_pos_gt_hn(scores, pos, hn_all)
    pair_clean, n_clean = pairwise_pos_gt_hn(scores, pos, hn_clean)
    row = dict(raw)
    row.update(
        {
            "split": gold.split,
            "gold_answers": list(gold.gold_answers),
            "selected_chunk_label": selected_label,
            "qa_chunk_label": qa_label,
            "qa_chunk_primary_class": qa_primary,
            "n_positive": len(pos),
            "n_hard_negative": len(hn_all),
            "n_hard_negative_clean": len(hn_clean),
            "hn_bucket": hn_bucket(len(hn_all)),
            "mrr": reciprocal_rank(ranked, pos),
            "pairwise_pos_gt_hn": pair_all,
            "pairwise_all_hn": pair_all,
            "pairwise_clean_hn": pair_clean,
            "n_pairs": n_all,
            "n_pairs_clean": n_clean,
            "selected_is_hn": selected_label == "hard_negative",
        }
    )
    for k in ks:
        row[f"hit@{k}"] = hit_at_k(ranked, pos, k)
        row[f"coverage@{k}"] = coverage_at_k(ranked, pos, k)
        row[f"hn@{k}"] = hn_at_k(ranked, hn_all, k)
        row[f"hn@{k}_all"] = hn_at_k(ranked, hn_all, k)
        row[f"hn@{k}_clean"] = hn_at_k(ranked, hn_clean, k)
    row.update(qa_row(raw.get("qa_answer"), gold.gold_answers, skip_bertscore=skip_bertscore))
    if selected_label == "positive":
        row["f1_given_selected_positive"] = row.get("token_f1")
    else:
        row["f1_given_selected_positive"] = None
    return row


def run_inference(
    records: Sequence[InstanceRecord],
    score_fn: ScoreFn,
    *,
    ranking_method: str,
    split: str,
    output_dir,
    config,
    progress=None,
    answer_fn: AnswerFn | None = None,
    skip_qa: bool = False,
    flags: Sequence[dict[str, Any]] | None = None,
    extra_raw: Callable[[RuntimeInstance, Sequence[float], Sequence[int]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    assert_eval_split_allowed(
        split,
        allow_internal_test=bool(config.allow_internal_test),
        output_dir=Path(config.output_dir),
        require_freeze=True,
    )
    gold_allowed = split != "internal_test" or bool(config.allow_internal_test)
    flag_map = flag_index(flags or [])
    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    with timed_block() as timing:
        for index, gold in enumerate(records, start=1):
            runtime = to_runtime(gold)
            scores = score_fn(runtime)
            ranked = rank_ids(runtime.chunks, scores)
            qa = None
            if answer_fn is not None and not skip_qa:
                qa = answer_fn(runtime, ranked)
            extra = extra_raw(runtime, scores, ranked) if extra_raw else None
            raw = build_raw_prediction(
                runtime,
                ranking_method=ranking_method,
                scores=scores,
                ranked_ids=ranked,
                qa=qa,
                extra=extra,
            )
            raw_rows.append(raw)
            if gold_allowed:
                scored_rows.append(
                    enrich_prediction(
                        raw,
                        gold,
                        flags=flag_map,
                        skip_bertscore=config.skip_bertscore,
                        ks=config.top_k_values,
                    )
                )
            if progress:
                progress.count(index, len(records), item=f"{ranking_method}/{split}")
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "predictions_raw.jsonl", raw_rows)
    payload: dict[str, Any] = {
        "n": len(raw_rows),
        "split": split,
        "ranking_method": ranking_method,
        "elapsed_s": timing["elapsed_s"],
        "peak_gpu_mb": timing["peak_gpu_mb"],
        "metric_scales": {"token_f1": "0..100", "char_f1": "0..1", "rouge_l_f1": "0..1", "exact_match": "0..1"},
    }
    if scored_rows:
        write_jsonl(run_dir / "predictions_scored.jsonl", scored_rows)
        summary = summarize_ranking(scored_rows, ks=config.top_k_values)
        summary["pairwise_all_hn"] = _mean([row.get("pairwise_all_hn") for row in scored_rows if row.get("n_pairs")])
        summary["pairwise_clean_hn"] = _mean(
            [row.get("pairwise_clean_hn") for row in scored_rows if row.get("n_pairs_clean")]
        )
        pos_f1 = [row.get("f1_given_selected_positive") for row in scored_rows if row.get("selected_chunk_label") == "positive"]
        summary["f1_given_selected_positive"] = _mean(pos_f1)
        summary["top1_ambiguous_rate"] = _mean(
            [float(row.get("selected_chunk_label") == "ambiguous") for row in scored_rows]
        )
        summary["qa_source_ambiguous_rate"] = _mean(
            [float(row.get("qa_chunk_label") == "ambiguous") for row in scored_rows]
        )
        summary.update(payload)
        write_json(run_dir / "metrics.json", summary)
        payload = summary
    else:
        write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "runtime.json", {"elapsed_s": timing["elapsed_s"], "peak_gpu_mb": timing["peak_gpu_mb"]})
    return {"raw": raw_rows, "scored": scored_rows, "summary": payload, "run_dir": str(run_dir)}


def _mean(values: Sequence[Any]) -> float:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _int_field(payload: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in payload and payload[key] is not None and payload[key] != "":
            return int(payload[key])
    return int(default)
