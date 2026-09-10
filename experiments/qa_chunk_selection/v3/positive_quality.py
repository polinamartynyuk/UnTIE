"""Positive-quality analysis: which gold-positives the frozen reader can extract from."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.qa import answer_chunk
from experiments.qa_chunk_selection.v2.reader_oracle import _gold_stats
from experiments.qa_chunk_selection.v2.schemas import chunk_to_runtime

from .io_util import write_json, write_jsonl
from .structure import error_analysis_categories, neighbor_sentences


def band(f1: float) -> str:
    if f1 >= 80:
        return "reader_easy"
    if f1 >= 50:
        return "reader_medium"
    if f1 > 0:
        return "reader_hard"
    return "reader_failed"


def run_positive_quality(
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
    rows = []
    for index, item in enumerate(records, start=1):
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
            f1 = float(metrics.get("token_f1") or 0.0)
            gold_stats = _gold_stats(chunk.text, item.gold_answers)
            prev = next((other.text for other in item.chunks if other.chunk_id == chunk.previous_chunk_id), None)
            nxt = next((other.text for other in item.chunks if other.chunk_id == chunk.next_chunk_id), None)
            neighbors = neighbor_sentences(chunk.text, prev, nxt)
            rows.append(
                {
                    "document_id": item.document_id,
                    "question_id": item.question_id,
                    "aspect": item.aspect,
                    "split": item.split,
                    "chunk_id": chunk.chunk_id,
                    "chunk_length": len(chunk.text),
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "token_f1": f1,
                    "exact_match": metrics.get("exact_match"),
                    "band": band(f1),
                    "predicted_answer": pred.get("qa_answer"),
                    "qa_span_score": pred.get("qa_span_score"),
                    "qa_confidence_proxy": pred.get("qa_confidence_proxy"),
                    "discourse_tags": error_analysis_categories(chunk.text),
                    "previous_sentence": neighbors["previous_sentence"],
                    "next_sentence": neighbors["next_sentence"],
                    **gold_stats,
                }
            )
        if progress:
            progress.count(index, len(records), item=f"positive-quality/{item.split}")
    by_band = defaultdict(int)
    by_aspect = defaultdict(list)
    for row in rows:
        by_band[row["band"]] += 1
        by_aspect[row["aspect"]].append(row["token_f1"])
    competing = sum(1 for row in rows if row["band"] in {"reader_hard", "reader_failed"} and row.get("gold_occurrence_count", 0) >= 1)
    long_fail = sum(1 for row in rows if row["band"] in {"reader_hard", "reader_failed"} and row["chunk_length"] >= 240)
    d4_gate = {
        "pass": bool(rows) and (competing / max(len(rows), 1) >= 0.25 or long_fail / max(len(rows), 1) >= 0.25),
        "n_positives": len(rows),
        "n_hard_or_failed_with_gold_present": competing,
        "n_hard_or_failed_long": long_fail,
        "reason": "local noise / competing entities / long context among reader-failed positives",
    }
    unique_answers = defaultdict(set)
    for item in records:
        for chunk in item.positives:
            for gold in item.gold_answers:
                unique_answers[(item.document_id, item.aspect)].add(str(gold).casefold())
    multi_span = 0
    for item in records:
        if len(item.positives) >= 2 and len(item.gold_answers) >= 1:
            multi_span += 1
    d5_gate = {
        "pass": bool(records) and (multi_span / max(len(records), 1) >= 0.20),
        "n_questions_with_ge2_positives": multi_span,
        "n_questions": len(records),
    }
    summary = {
        "n_positives": len(rows),
        "bands": dict(by_band),
        "mean_f1_by_aspect": {key: float(sum(vals) / len(vals)) for key, vals in by_aspect.items() if vals},
        "d4_gate": d4_gate,
        "d5_gate": d5_gate,
        "diagnostic_only": True,
    }
    write_jsonl(output_dir / "positives.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "d4_gate.json", d4_gate)
    write_json(output_dir / "d5_gate.json", d5_gate)
    return summary
