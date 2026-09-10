"""R1 reader oracle. Diagnostic-only; never used for model selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.sanity import gold_string_in_text
from experiments.qa_chunk_selection.schema import InstanceRecord
from untie.aspect_qa.normalize import identity_key

from .io_util import write_json, write_jsonl
from .qa import answer_chunk
from .schemas import chunk_to_runtime


def _gold_stats(text: str, golds: Sequence[str]) -> dict[str, Any]:
    hay = identity_key(text)
    present = gold_string_in_text(golds, text)
    count = 0
    rel = 0.0
    if hay:
        for gold in golds:
            key = identity_key(gold)
            if key:
                count += hay.count(key)
        if golds:
            key = identity_key(golds[0])
            pos = hay.find(key) if key else -1
            rel = float(pos / max(len(hay), 1)) if pos >= 0 else 0.0
    return {"gold_string_present": present, "gold_occurrence_count": count, "gold_relative_position": rel}


def run_reader_oracle(
    records: Sequence[InstanceRecord],
    *,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
    output_dir,
    progress=None,
    skip_bertscore: bool = True,
) -> dict[str, Any]:
    per_positive = []
    per_question = []
    by_aspect = defaultdict(list)
    for index, item in enumerate(records, start=1):
        rows = []
        for chunk in item.positives:
            pred = answer_chunk(
                answer_fn,
                item.question,
                chunk_to_runtime(chunk),
                cache=cache,
                model_id=model_id,
                inference_hash=inference_hash,
            )
            metrics = qa_row(pred.get("qa_answer"), item.gold_answers, skip_bertscore=skip_bertscore)
            gold_stats = _gold_stats(chunk.text, item.gold_answers)
            row = {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "chunk_id": chunk.chunk_id,
                "chunk_length_tokens": chunk.token_end - chunk.token_start,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "predicted_answer": pred.get("qa_answer"),
                "qa_start_logit": pred.get("qa_start_logit"),
                "qa_end_logit": pred.get("qa_end_logit"),
                "qa_span_score": pred.get("qa_span_score"),
                "qa_confidence_proxy": pred.get("qa_confidence_proxy"),
                "qa_chunk_start_char": pred.get("qa_chunk_start_char"),
                "qa_chunk_end_char": pred.get("qa_chunk_end_char"),
                "source_start_char": pred.get("source_start_char"),
                "source_end_char": pred.get("source_end_char"),
                "oracle": True,
                "diagnostic_only": True,
                **metrics,
                **gold_stats,
            }
            rows.append(row)
            per_positive.append(row)
        f1s = [float(row.get("token_f1") or 0.0) for row in rows]
        ems = [float(row.get("exact_match") or 0.0) for row in rows]
        qrow = {
            "document_id": item.document_id,
            "question_id": item.question_id,
            "aspect": item.aspect,
            "split": item.split,
            "n_positive": len(rows),
            "reader_solvable_EM": float(any(value >= 1.0 for value in ems)),
            "reader_solvable_F1>0": float(any(value > 0 for value in f1s)),
            "reader_solvable_F1>=50": float(any(value >= 50 for value in f1s)),
            "reader_solvable_F1>=80": float(any(value >= 80 for value in f1s)),
            "best_positive_token_f1": max(f1s) if f1s else 0.0,
            "best_positive_chunk_id": rows[int(f1s.index(max(f1s)))]["chunk_id"] if f1s else None,
            "oracle": True,
            "diagnostic_only": True,
        }
        per_question.append(qrow)
        by_aspect[item.aspect].append(qrow)
        if progress:
            progress.count(index, len(records), item=f"reader-oracle/{item.split}")
    summary = {
        "n_questions": len(per_question),
        "n_positives": len(per_positive),
        "oracle": True,
        "diagnostic_only": True,
        "mean_best_positive_token_f1": _mean([row["best_positive_token_f1"] for row in per_question]),
        "reader_solvable_EM": _mean([row["reader_solvable_EM"] for row in per_question]),
        "reader_solvable_F1>0": _mean([row["reader_solvable_F1>0"] for row in per_question]),
        "reader_solvable_F1>=50": _mean([row["reader_solvable_F1>=50"] for row in per_question]),
        "reader_solvable_F1>=80": _mean([row["reader_solvable_F1>=80"] for row in per_question]),
        "by_aspect": {
            aspect: {
                "n": len(rows),
                "reader_solvable_EM": _mean([row["reader_solvable_EM"] for row in rows]),
                "best_positive_token_f1": _mean([row["best_positive_token_f1"] for row in rows]),
            }
            for aspect, rows in by_aspect.items()
        },
    }
    write_jsonl(output_dir / "per_positive.jsonl", per_positive)
    write_jsonl(output_dir / "per_question.jsonl", per_question)
    write_json(output_dir / "summary.json", summary)
    return summary


def refuse_oracle_for_selection(*_args, **_kwargs) -> None:
    raise RuntimeError("reader oracle is diagnostic_only and must not be used for model selection")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")
