"""Write Phase 1 tables and summary without touching preprocessing files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .assign import InstanceResult
from .config import Phase1Config


PROTECTED_INPUT = "datasets/scirex_aspect_qa_preprocessing"


def assert_isolated_output(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    protected = (project_root / PROTECTED_INPUT).resolve()
    if resolved == protected or protected in resolved.parents:
        raise ValueError(f"Phase 1 refuses to write into preprocessing dir: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def instance_rows(results: Sequence[InstanceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "document_id": item.document_id,
                "aspect": item.aspect,
                "qa_instance_id": item.qa_instance_id,
                "split": item.split,
                "question": item.question,
                "status": item.status,
                "instance_status": item.instance_status,
                "enrichment_mode": item.enrichment_mode,
                "extractable_gold_answers": list(item.extractable_gold_answers),
                "unextractable_gold_answers": list(item.unextractable_gold_answers),
                "positive_chunk_ids": item.positive_ids,
                "evidence_positive_qa_failed_ids": item.evidence_failed_ids,
                "false_negative_candidate_ids": item.fn_ids,
                "confirmed_false_negative_ids": item.confirmed_fn_ids,
                "hard_negative_chunk_ids": item.hard_negative_ids,
                "easy_negative_chunk_ids": item.easy_negative_ids,
                "fn_filter_status": item.fn_filter_status,
                "selected_K": item.selected_k,
            }
        )
    return rows


def chunk_dataset_rows(results: Sequence[InstanceResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        for chunk in item.chunks:
            rows.append(
                {
                    "document_id": item.document_id,
                    "aspect": item.aspect,
                    "qa_instance_id": item.qa_instance_id,
                    "split": item.split,
                    "question": item.question,
                    "status": item.status,
                    "instance_status": item.instance_status,
                    "enrichment_mode": item.enrichment_mode,
                    "extractable_gold_answers": list(item.extractable_gold_answers),
                    "unextractable_gold_answers": list(item.unextractable_gold_answers),
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "gold_supporting": chunk.gold_supporting,
                    "qa_positive": chunk.qa_positive,
                    "matched_gold_answer": chunk.matched_gold_answer,
                    "match_type": chunk.match_type,
                    "extra_tokens": chunk.extra_tokens,
                    "qa_predicted_answer": chunk.predicted_answer,
                    "qa_confidence": chunk.qa_confidence,
                    "answer_quality": chunk.answer_quality,
                    "answer_start": chunk.start,
                    "answer_end": chunk.end,
                    "start_logit": chunk.start_logit,
                    "end_logit": chunk.end_logit,
                    "bm25_score": chunk.bm25_score,
                    "dense_question_similarity": chunk.dense_question_similarity,
                    "selected_K": item.selected_k,
                    "cluster_id": chunk.cluster_id,
                    "same_cluster_as_positive": chunk.same_cluster_as_positive,
                    "fn_filter_status": item.fn_filter_status,
                    "primary_class": chunk.primary_class,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "section_type": chunk.section_type,
                    "section_title": chunk.section_title,
                }
            )
    return rows


def class_table(rows: Sequence[dict[str, Any]], primary_class: str) -> pd.DataFrame:
    return pd.DataFrame([row for row in rows if row["primary_class"] == primary_class])


def collect_statistics(
    results: Sequence[InstanceResult],
    *,
    excluded_no_extractable: int,
    documents_in_split: int,
) -> dict[str, Any]:
    aspects = ("method", "task", "metric", "material")
    n_unique_chunks = len({(item.document_id, chunk.chunk_id) for item in results for chunk in item.chunks})
    overall = _aspect_stats(results)
    by_aspect = {name: _aspect_stats([item for item in results if item.aspect == name]) for name in aspects}
    return {
        "documents_total": documents_in_split,
        "qa_instances_total": len(results),
        "chunks_total": n_unique_chunks,
        "aspect_chunk_pairs_total": sum(len(item.chunks) for item in results),
        "excluded_no_extractable_gold": excluded_no_extractable,
        "no_qa_positive_instances": sum(item.instance_status == "no_qa_positive" for item in results),
        "overall": overall,
        "by_aspect": by_aspect,
        "fn_filter_status": dict(Counter(item.fn_filter_status for item in results)),
        "problems": [problem for item in results for problem in item.problems],
    }


def _aspect_stats(results: Sequence[InstanceResult]) -> dict[str, Any]:
    if not results:
        return {}
    n_pos = sum(len(item.positive_ids) for item in results)
    n_hn = sum(len(item.hard_negative_ids) for item in results)
    n_fn = sum(len(item.fn_ids) for item in results)
    n_fail = sum(len(item.evidence_failed_ids) for item in results)
    n_easy = sum(len(item.easy_negative_ids) for item in results)
    n_gold_sup = sum(sum(1 for chunk in item.chunks if chunk.gold_supporting) for item in results)
    n_ext = sum(len(item.extractable_gold_answers) for item in results)
    n_unext = sum(len(item.unextractable_gold_answers) for item in results)
    return {
        "instances_total": len(results),
        "ready_instances": sum(item.status == "ready" for item in results),
        "partially_extractable_instances": sum(item.status == "partially_extractable" for item in results),
        "no_qa_positive_instances": sum(item.instance_status == "no_qa_positive" for item in results),
        "extractable_gold_answers_total": n_ext,
        "unextractable_gold_answers_total": n_unext,
        "extractable_gold_answers_per_instance_mean": n_ext / len(results),
        "positive_chunks_total": n_pos,
        "positive_per_instance_mean": n_pos / len(results),
        "gold_supporting_chunks_total": n_gold_sup,
        "evidence_positive_qa_failed_total": n_fail,
        "hard_negatives_total": n_hn,
        "hard_negatives_per_instance_mean": n_hn / len(results),
        "false_negative_candidates_total": n_fn,
        "false_negative_rate": n_fn / max(n_fn + n_hn, 1),
        "easy_negatives_total": n_easy,
        "confirmed_false_negatives_total": sum(len(item.confirmed_fn_ids) for item in results),
    }


def write_outputs(
    output_dir: Path,
    *,
    results: Sequence[InstanceResult],
    config: Phase1Config,
    extra_config: dict[str, Any],
    stats: dict[str, Any],
    check_errors: Sequence[str],
) -> dict[str, str]:
    rows = chunk_dataset_rows(results)
    dataset = pd.DataFrame(rows)
    written = {
        "qa_chunk_dataset": str(output_dir / "qa_chunk_dataset.parquet"),
        "qa_chunk_predictions": str(output_dir / "qa_chunk_predictions.parquet"),
        "qa_instances": str(output_dir / "qa_instances.parquet"),
        "positive_chunks": str(output_dir / "positive_chunks.parquet"),
        "evidence_positive_qa_failed": str(output_dir / "evidence_positive_qa_failed.parquet"),
        "false_negative_candidates": str(output_dir / "false_negative_candidates.parquet"),
        "hard_negative_chunks": str(output_dir / "hard_negative_chunks.parquet"),
        "easy_negative_chunks": str(output_dir / "easy_negative_chunks.parquet"),
        "clustering_diagnostics": str(output_dir / "clustering_diagnostics.parquet"),
        "gap_statistics": str(output_dir / "gap_statistics.parquet"),
        "experiment_config": str(output_dir / "experiment_config.json"),
        "statistics": str(output_dir / "statistics.json"),
        "sanity_errors": str(output_dir / "sanity_errors.json"),
        "phase1_summary": str(output_dir / "phase1_summary.md"),
    }
    _write_parquet(dataset, output_dir / "qa_chunk_dataset.parquet")
    pred_cols = [
        "document_id",
        "aspect",
        "qa_instance_id",
        "question",
        "chunk_id",
        "qa_predicted_answer",
        "answer_start",
        "answer_end",
        "start_logit",
        "end_logit",
        "qa_confidence",
        "matched_gold_answer",
        "answer_quality",
        "qa_positive",
    ]
    _write_parquet(dataset[pred_cols], output_dir / "qa_chunk_predictions.parquet")
    _write_parquet(pd.DataFrame(instance_rows(results)), output_dir / "qa_instances.parquet")
    for name, label in [
        ("positive_chunks", "qa_positive"),
        ("evidence_positive_qa_failed", "evidence_positive_qa_failed"),
        ("false_negative_candidates", "false_negative_candidate"),
        ("hard_negative_chunks", "hard_negative"),
        ("easy_negative_chunks", "easy_negative"),
    ]:
        _write_parquet(class_table(rows, label), output_dir / f"{name}.parquet")
    cluster_rows = []
    gap_rows = []
    for item in results:
        cluster_rows.append(
            {
                "document_id": item.document_id,
                "aspect": item.aspect,
                "qa_instance_id": item.qa_instance_id,
                "fn_filter_status": item.fn_filter_status,
                "selected_K": item.selected_k,
                "cluster_sizes": list(item.cluster_sizes),
            }
        )
        for k, gap, sk in zip(range(1, len(item.gap_values) + 1), item.gap_values, item.gap_sk):
            gap_rows.append(
                {
                    "document_id": item.document_id,
                    "aspect": item.aspect,
                    "qa_instance_id": item.qa_instance_id,
                    "k": k,
                    "gap": gap,
                    "sk": sk,
                }
            )
    _write_parquet(pd.DataFrame(cluster_rows), output_dir / "clustering_diagnostics.parquet")
    _write_parquet(pd.DataFrame(gap_rows), output_dir / "gap_statistics.parquet")
    (output_dir / "experiment_config.json").write_text(
        json.dumps({**config.to_dict(), **extra_config}, indent=2, ensure_ascii=False) + "\n",
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
    (output_dir / "phase1_summary.md").write_text(
        render_summary(results, config, extra_config, stats, check_errors),
        encoding="utf-8",
    )
    return written


def render_summary(
    results: Sequence[InstanceResult],
    config: Phase1Config,
    extra_config: dict[str, Any],
    stats: dict[str, Any],
    check_errors: Sequence[str],
) -> str:
    overall = stats.get("overall") or {}
    lines = [
        "# Phase 1 SciREX: positive / FN / hard-negative",
        "",
        "## Dataset",
        "",
        f"- corpus: SciREX, enrichment_mode={config.enrichment_mode}, score aspect disabled",
        f"- split used: `{config.split}` (inherited; no new split)",
        f"- debug={config.debug}",
        f"- documents_total={stats.get('documents_total')}",
        f"- qa_instances_total={stats.get('qa_instances_total')}",
        f"- aspect_chunk_pairs_total={stats.get('aspect_chunk_pairs_total')}",
        f"- excluded no_extractable_gold={stats.get('excluded_no_extractable_gold')}",
        f"- no_qa_positive={stats.get('no_qa_positive_instances')}",
        "",
        "## QA model",
        "",
        f"- qa_model: {extra_config.get('qa_model')}",
        f"- embedding_model: {extra_config.get('embedding_model')}",
        f"- match: {config.positive_match_method}",
        f"- containment_max_extra_tokens: {config.containment_max_extra_tokens}",
        "",
        "## Hard-negative mining",
        "",
        f"- BM25(question, chunk), k1={config.bm25_k1}, b={config.bm25_b}",
        f"- top_n={config.hard_negative_top_n}",
        "",
        "## False-negative filter (F1 main)",
        "",
        "- C0 passage-only embeddings",
        "- K-means + Gap Statistic + hard assignment",
        "- Gumbel-Softmax is optional ablation only; temperature decay is off",
        f"- gap_bootstrap={config.gap_bootstrap}, k_max_cap={config.k_max_cap}",
        f"- fn_filter_status: {stats.get('fn_filter_status')}",
        "",
        "## Results overall",
        "",
        json.dumps(overall, indent=2, ensure_ascii=False),
        "",
        "## Results by aspect",
        "",
        json.dumps(stats.get("by_aspect"), indent=2, ensure_ascii=False),
        "",
        "## Examples",
        "",
    ]
    lines.extend(_examples(results))
    lines.extend(
        [
            "",
            "## Sanity",
            "",
            f"- check_errors={len(check_errors)}",
            "",
            "## Limitations",
            "",
            "- gold_supporting comes from confirmed mentions of the same aspect",
            "- one instance may have several extractable_gold_answers",
            "- no_extractable_gold is excluded from the main experiment",
            "- cluster label is not gold relevance",
            "- this adapts mHFN to within-document aspect-level extractive QA",
            "- main FN filter is hard K-means assignment",
            "- Gumbel-Softmax is optional; no temperature decay because the encoder is frozen",
            "- external enrichment is unused in E0",
            "",
        ]
    )
    if stats.get("problems"):
        lines.append("## Data problems")
        lines.append("")
        for problem in stats["problems"][:50]:
            lines.append(f"- {problem}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _examples(results: Sequence[InstanceResult], n: int = 3) -> list[str]:
    lines: list[str] = []
    shown = {"qa_positive": 0, "hard_negative": 0, "false_negative_candidate": 0}
    for item in results:
        for chunk in item.chunks:
            label = chunk.primary_class
            if label not in shown or shown[label] >= n:
                continue
            shown[label] += 1
            lines.append(f"### {label} #{shown[label]}")
            lines.append("")
            lines.append(f"- document_id: `{item.document_id}`")
            lines.append(f"- aspect: `{item.aspect}`")
            lines.append(f"- question: {item.question}")
            lines.append(f"- extractable_gold_answers: {item.extractable_gold_answers}")
            lines.append(f"- chunk_id: {chunk.chunk_id}")
            lines.append(f"- predicted: {chunk.predicted_answer!r}")
            lines.append(f"- matched_gold_answer: {chunk.matched_gold_answer!r}")
            lines.append(f"- match_type: {chunk.match_type}")
            lines.append(f"- extra_tokens: {chunk.extra_tokens}")
            lines.append(f"- gold_supporting: {chunk.gold_supporting}")
            lines.append(f"- bm25: {chunk.bm25_score:.3f}")
            lines.append(f"- text: {chunk.text[:400]!r}")
            lines.append("")
        if all(value >= n for value in shown.values()):
            break
    return lines
