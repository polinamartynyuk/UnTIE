"""Write multi-question tables without touching preprocessing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from experiments.phase1_scirex.report import assert_isolated_output

from .assign import InstanceResult
from .config import MultiQuestionConfig


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def question_prediction_rows(results: Sequence[InstanceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        for chunk in item.chunks:
            for pred in chunk.predictions:
                rows.append(
                    {
                        "document_id": item.document_id,
                        "aspect": item.aspect,
                        "qa_instance_id": item.qa_instance_id,
                        "split": item.split,
                        "chunk_id": chunk.chunk_id,
                        "question_id": pred.question_id,
                        "is_canonical_question": pred.question_id == 0,
                        "question": pred.question,
                        "qa_predicted_answer": pred.predicted_answer,
                        "matched_gold_answer": pred.matched_gold_answer,
                        "qa_correct": pred.qa_correct,
                        "qa_confidence": pred.qa_confidence,
                        "answer_quality": pred.answer_quality,
                        "match_type": pred.match_type,
                        "extra_tokens": pred.extra_tokens,
                        "answer_start": pred.start,
                        "answer_end": pred.end,
                        "bm25_score": pred.bm25_score,
                        "bm25_normalized": pred.bm25_normalized,
                        "bm25_rank": pred.bm25_rank,
                        "extractable_gold_answers": list(item.extractable_gold_answers),
                    }
                )
    return rows


def chunk_rows(results: Sequence[InstanceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        for chunk in item.chunks:
            rows.append(
                {
                    "document_id": item.document_id,
                    "aspect": item.aspect,
                    "qa_instance_id": item.qa_instance_id,
                    "split": item.split,
                    "canonical_question": item.canonical_question,
                    "n_questions": item.n_questions,
                    "status": item.status,
                    "instance_status": item.instance_status,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "gold_supporting": chunk.gold_supporting,
                    "n_correct": chunk.n_correct,
                    "qa_consistency": chunk.qa_consistency,
                    "consistency_class": chunk.consistency_class,
                    "any_correct": chunk.any_correct,
                    "robust_positive_threshold": item.robust_positive_threshold,
                    "primary_class": chunk.primary_class,
                    "mean_normalized_bm25": chunk.mean_normalized_bm25,
                    "mean_rank": chunk.mean_rank,
                    "rrf_score": chunk.rrf_score,
                    "hardness": chunk.hardness,
                    "bm25_aggregate": item.bm25_aggregate,
                    "selected_K": item.selected_k,
                    "cluster_id": chunk.cluster_id,
                    "same_cluster_as_positive": chunk.same_cluster_as_positive,
                    "fn_filter_status": item.fn_filter_status,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "section_type": chunk.section_type,
                    "section_title": chunk.section_title,
                    "extractable_gold_answers": list(item.extractable_gold_answers),
                    "unextractable_gold_answers": list(item.unextractable_gold_answers),
                }
            )
    return rows


def instance_rows(results: Sequence[InstanceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "document_id": item.document_id,
                "aspect": item.aspect,
                "qa_instance_id": item.qa_instance_id,
                "split": item.split,
                "canonical_question": item.canonical_question,
                "questions": list(item.questions),
                "n_questions": item.n_questions,
                "status": item.status,
                "instance_status": item.instance_status,
                "robust_positive_threshold": item.robust_positive_threshold,
                "extractable_gold_answers": list(item.extractable_gold_answers),
                "unextractable_gold_answers": list(item.unextractable_gold_answers),
                "robust_positive_chunk_ids": item.robust_positive_ids,
                "question_sensitive_chunk_ids": item.question_sensitive_ids,
                "gold_supporting_robust_negative_ids": item.gold_failed_ids,
                "false_negative_candidate_ids": item.fn_ids,
                "hard_negative_chunk_ids": item.hard_negative_ids,
                "easy_negative_chunk_ids": item.easy_negative_ids,
                "fn_filter_status": item.fn_filter_status,
                "selected_K": item.selected_k,
                "bm25_aggregate": item.bm25_aggregate,
            }
        )
    return rows


def _pairwise_agreement(results: Sequence[InstanceResult]) -> dict[str, float]:
    agree = 0
    total = 0
    for item in results:
        n_q = item.n_questions
        if n_q < 2:
            continue
        for chunk in item.chunks:
            flags = [pred.qa_correct for pred in chunk.predictions]
            for left in range(n_q):
                for right in range(left + 1, n_q):
                    total += 1
                    agree += int(flags[left] == flags[right])
    return {
        "pairwise_qa_correct_agreement": (agree / total) if total else 0.0,
        "pairwise_pairs_total": total,
    }


def collect_statistics(results: Sequence[InstanceResult]) -> dict[str, Any]:
    n = len(results)
    chunks = [chunk for item in results for chunk in item.chunks]
    by_class = Counter(chunk.primary_class for chunk in chunks)
    by_cons = Counter(chunk.consistency_class for chunk in chunks)
    n_q0_correct = sum(
        1
        for chunk in chunks
        for pred in chunk.predictions
        if pred.question_id == 0 and pred.qa_correct
    )
    top5_overlap = _bm25_top5_overlap(results)
    return {
        "instances_total": n,
        "documents_total": len({item.document_id for item in results}),
        "aspect_chunk_pairs_total": len(chunks),
        "no_robust_positive_instances": sum(
            item.instance_status == "no_robust_positive" for item in results
        ),
        "instances_with_robust_positive": sum(1 for item in results if item.robust_positive_ids),
        "instances_with_pos_and_hn": sum(
            1 for item in results if item.robust_positive_ids and item.hard_negative_ids
        ),
        "mean_qa_consistency": (
            sum(chunk.qa_consistency for chunk in chunks) / len(chunks) if chunks else 0.0
        ),
        "canonical_correct_chunks": n_q0_correct,
        "any_correct_chunks": sum(1 for chunk in chunks if chunk.any_correct),
        "robust_positive_chunks": by_class.get("robust_positive", 0),
        "question_sensitive_chunks": by_class.get("question_sensitive", 0),
        "robust_negative_chunks": by_cons.get("robust_negative", 0),
        "gold_supporting_robust_negative_chunks": by_class.get(
            "gold_supporting_robust_negative", 0
        ),
        "hard_negative_chunks": by_class.get("hard_negative", 0),
        "false_negative_candidates": by_class.get("false_negative_candidate", 0),
        "easy_negative_chunks": by_class.get("easy_negative", 0),
        "primary_class_counts": dict(by_class),
        "consistency_class_counts": dict(by_cons),
        "fn_filter_status": dict(Counter(item.fn_filter_status for item in results)),
        "by_aspect": {
            aspect: _aspect_stats([item for item in results if item.aspect == aspect])
            for aspect in ("method", "task", "metric", "material")
        },
        "bm25_top5_overlap": top5_overlap,
        **_pairwise_agreement(results),
    }


def _aspect_stats(results: Sequence[InstanceResult]) -> dict[str, Any]:
    if not results:
        return {}
    chunks = [chunk for item in results for chunk in item.chunks]
    return {
        "instances": len(results),
        "no_robust_positive_instances": sum(
            item.instance_status == "no_robust_positive" for item in results
        ),
        "robust_positive_chunks": sum(len(item.robust_positive_ids) for item in results),
        "question_sensitive_chunks": sum(len(item.question_sensitive_ids) for item in results),
        "hard_negative_chunks": sum(len(item.hard_negative_ids) for item in results),
        "mean_qa_consistency": (
            sum(chunk.qa_consistency for chunk in chunks) / len(chunks) if chunks else 0.0
        ),
        "pos_hn_instances": sum(
            1 for item in results if item.robust_positive_ids and item.hard_negative_ids
        ),
    }


def _bm25_top5_overlap(results: Sequence[InstanceResult], top_n: int = 5) -> dict[str, float]:
    pairs = {"mean_normalized_vs_rrf": [], "mean_normalized_vs_mean_rank": [], "rrf_vs_mean_rank": []}
    for item in results:
        pool = [
            chunk
            for chunk in item.chunks
            if chunk.consistency_class == "robust_negative" and not chunk.gold_supporting
        ]
        if len(pool) < 2:
            continue

        def top_ids(key: str) -> set[int]:
            if key == "mean_normalized":
                ranked = sorted(pool, key=lambda chunk: (-chunk.mean_normalized_bm25, chunk.chunk_id))
            elif key == "rrf":
                ranked = sorted(pool, key=lambda chunk: (-chunk.rrf_score, chunk.chunk_id))
            else:
                ranked = sorted(pool, key=lambda chunk: (chunk.mean_rank, chunk.chunk_id))
            return {chunk.chunk_id for chunk in ranked[: min(top_n, len(ranked))]}

        mean_ids = top_ids("mean_normalized")
        rrf_ids = top_ids("rrf")
        rank_ids = top_ids("mean_rank")
        pairs["mean_normalized_vs_rrf"].append(_jaccard(mean_ids, rrf_ids))
        pairs["mean_normalized_vs_mean_rank"].append(_jaccard(mean_ids, rank_ids))
        pairs["rrf_vs_mean_rank"].append(_jaccard(rrf_ids, rank_ids))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in pairs.items()}


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / float(len(left | right))


def write_outputs(
    output_dir: Path,
    *,
    results: Sequence[InstanceResult],
    config: MultiQuestionConfig,
    extra_config: dict[str, Any],
    stats: dict[str, Any],
    check_errors: Sequence[str],
    threshold_selection: dict[str, Any] | None = None,
) -> dict[str, str]:
    chunks = pd.DataFrame(chunk_rows(results))
    questions = pd.DataFrame(question_prediction_rows(results))
    instances = pd.DataFrame(instance_rows(results))
    written = {
        "qa_chunk_dataset": str(output_dir / "qa_chunk_dataset.parquet"),
        "qa_question_predictions": str(output_dir / "qa_question_predictions.parquet"),
        "qa_instances": str(output_dir / "qa_instances.parquet"),
        "experiment_config": str(output_dir / "experiment_config.json"),
        "statistics": str(output_dir / "statistics.json"),
        "sanity_errors": str(output_dir / "sanity_errors.json"),
        "phase1_summary": str(output_dir / "phase1_summary.md"),
    }
    _write_parquet(chunks, output_dir / "qa_chunk_dataset.parquet")
    _write_parquet(questions, output_dir / "qa_question_predictions.parquet")
    _write_parquet(instances, output_dir / "qa_instances.parquet")
    for name, label in [
        ("robust_positive_chunks", "robust_positive"),
        ("question_sensitive_chunks", "question_sensitive"),
        ("gold_supporting_robust_negative", "gold_supporting_robust_negative"),
        ("false_negative_candidates", "false_negative_candidate"),
        ("hard_negative_chunks", "hard_negative"),
        ("easy_negative_chunks", "easy_negative"),
    ]:
        subset = chunks[chunks["primary_class"] == label] if not chunks.empty else chunks
        _write_parquet(subset, output_dir / f"{name}.parquet")
        written[name] = str(output_dir / f"{name}.parquet")
    payload = {**config.to_dict(), **extra_config}
    (output_dir / "experiment_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sanity_errors.json").write_text(
        json.dumps(list(check_errors), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if threshold_selection is not None:
        slim = {key: value for key, value in threshold_selection.items() if key != "results"}
        (output_dir / "threshold_selection.json").write_text(
            json.dumps(slim, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        written["threshold_selection"] = str(output_dir / "threshold_selection.json")
    (output_dir / "phase1_summary.md").write_text(
        render_summary(config, extra_config, stats, check_errors, threshold_selection),
        encoding="utf-8",
    )
    return written


def render_summary(
    config: MultiQuestionConfig,
    extra_config: dict[str, Any],
    stats: dict[str, Any],
    check_errors: Sequence[str],
    threshold_selection: dict[str, Any] | None,
) -> str:
    lines = [
        f"# Multi-question Phase 1 ({config.mode})",
        "",
        f"- split: `{config.split}`",
        f"- n_questions: {config.n_questions}",
        f"- match: {config.positive_match_method} extra≤{config.containment_max_extra_tokens}",
        f"- primary positive: robust_positive (QAConsistency ≥ threshold)",
        f"- any_correct is diagnostic only",
        f"- robust_positive_threshold: {extra_config.get('robust_positive_threshold')}",
        f"- BM25 aggregate used for HN: {config.bm25_aggregate}",
        f"- documents: {stats.get('documents_total')} instances: {stats.get('instances_total')}",
        f"- no_robust_positive: {stats.get('no_robust_positive_instances')}",
        f"- robust_positive chunks: {stats.get('robust_positive_chunks')}",
        f"- question_sensitive chunks: {stats.get('question_sensitive_chunks')}",
        f"- robust_negative chunks: {stats.get('robust_negative_chunks')}",
        f"- hard_negatives: {stats.get('hard_negative_chunks')}",
        f"- pairwise qa_correct agreement: {stats.get('pairwise_qa_correct_agreement')}",
        f"- BM25 top-5 overlap: {json.dumps(stats.get('bm25_top5_overlap'), ensure_ascii=False)}",
        f"- sanity_errors: {len(check_errors)}",
        "",
        "## Statistics",
        "",
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        "",
    ]
    if threshold_selection:
        lines.extend(
            [
                "## DEV threshold selection",
                "",
                json.dumps(
                    {key: value for key, value in threshold_selection.items() if key != "results"},
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                "",
            ]
        )
    return "\n".join(lines)
