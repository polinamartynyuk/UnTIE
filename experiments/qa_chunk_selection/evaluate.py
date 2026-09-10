"""Shared evaluation: rank candidates, optional QA, write run artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .io_util import write_json, write_jsonl
from .metrics import qa_row, ranking_row, summarize_ranking
from .multipassage import encode_passage, global_best_span
from .progress import ProgressLogger
from .ranking import ScoreFn, rank_ids, score_map
from .schema import ChunkRecord, InstanceRecord


def evaluate_ranking(
    records: Sequence[InstanceRecord],
    score_fn: ScoreFn | None,
    *,
    experiment: str,
    split: str,
    ks: Sequence[int],
    progress: ProgressLogger | None = None,
    answer_fn: Callable[[str, str], Mapping[str, Any]] | None = None,
    skip_qa: bool = False,
    skip_bertscore: bool = True,
    select_k: int = 1,
    qa_mode: str = "top1",
    qa_answerer=None,
    instance_score_fn: Callable[[InstanceRecord], list[float]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    started = time.time()
    peak_gpu = 0.0
    n_chunks = 0
    n_tokens = 0
    for index, instance in enumerate(records, start=1):
        t0 = time.time()
        if instance_score_fn is not None:
            scores = instance_score_fn(instance)
        elif score_fn is not None:
            scores = score_fn(instance.question, instance.chunks)
        else:
            raise ValueError("score_fn or instance_score_fn is required")
        ranked = rank_ids(instance.chunks, scores)
        mapped = score_map(instance.chunks, scores)
        n_chunks += len(instance.chunks)
        row = ranking_row(instance, ranked, mapped, ks=ks)
        row["experiment"] = experiment
        row["latency_s"] = time.time() - t0
        chosen_ids = ranked[: max(int(select_k), 1)]
        row["selected_chunk_ids"] = chosen_ids
        if answer_fn is not None and not skip_qa:
            prediction = _run_qa(
                instance,
                chosen_ids,
                answer_fn,
                qa_mode=qa_mode,
                qa_answerer=qa_answerer,
            )
            row.update(prediction)
            row.update(qa_row(prediction.get("qa_answer"), instance.gold_answers, skip_bertscore=skip_bertscore))
            n_tokens += int(prediction.get("reader_tokens") or 0)
        rows.append(row)
        for chunk, score in zip(instance.chunks, scores):
            score_rows.append(
                {
                    "document_id": instance.document_id,
                    "question_id": instance.question_id,
                    "chunk_id": chunk.chunk_id,
                    "label": chunk.label,
                    "score": float(score),
                    "rank": ranked.index(chunk.chunk_id) + 1 if chunk.chunk_id in ranked else None,
                    "experiment": experiment,
                    "split": split,
                }
            )
        peak_gpu = max(peak_gpu, _gpu_mb())
        if progress:
            progress.count(index, len(records), item=f"{experiment}/{split}")
    summary = summarize_ranking(rows, ks=ks)
    summary.update(
        {
            "experiment": experiment,
            "split": split,
            "elapsed_s": time.time() - started,
            "chunks_processed": n_chunks,
            "reader_tokens": n_tokens,
            "latency_mean": float(sum(row["latency_s"] for row in rows) / len(rows)) if rows else 0.0,
            "peak_gpu_mb": peak_gpu,
        }
    )
    return {"rows": rows, "chunk_scores": score_rows, "summary": summary}


def _run_qa(
    instance: InstanceRecord,
    chunk_ids: Sequence[int],
    answer_fn: Callable[[str, str], Mapping[str, Any]],
    *,
    qa_mode: str,
    qa_answerer,
) -> dict[str, Any]:
    by_id = {chunk.chunk_id: chunk for chunk in instance.chunks}
    selected = [by_id[int(item)] for item in chunk_ids if int(item) in by_id]
    if not selected:
        return {
            "qa_answer": "",
            "qa_start": 0,
            "qa_end": 0,
            "qa_score": 0.0,
            "qa_start_logit": 0.0,
            "qa_end_logit": 0.0,
            "reader_tokens": 0,
            "qa_mode": qa_mode,
        }
    if qa_mode == "global" and qa_answerer is not None:
        passages = [encode_passage(qa_answerer, instance.question, chunk) for chunk in selected]
        span = global_best_span(passages)
        tokens = sum(len(item.start_logits) for item in passages)
        if span is None:
            return {
                "qa_answer": "",
                "qa_start": 0,
                "qa_end": 0,
                "qa_score": 0.0,
                "qa_start_logit": 0.0,
                "qa_end_logit": 0.0,
                "reader_tokens": tokens,
                "qa_mode": "global",
                "qa_chunk_id": None,
            }
        return {
            "qa_answer": span.text,
            "qa_start": span.start_char,
            "qa_end": span.end_char,
            "qa_score": span.logit_sum,
            "qa_start_logit": span.start_logit,
            "qa_end_logit": span.end_logit,
            "reader_tokens": tokens,
            "qa_mode": "global",
            "qa_chunk_id": span.chunk_id,
        }
    best: Mapping[str, Any] | None = None
    best_chunk: ChunkRecord | None = None
    tokens = 0
    for chunk in selected:
        result = answer_fn(instance.question, chunk.text)
        tokens += max(1, len(chunk.text.split()))
        score = float(result.get("start_logit", 0.0)) + float(result.get("end_logit", 0.0))
        if best is None or score > float(best.get("start_logit", 0.0)) + float(best.get("end_logit", 0.0)):
            best = result
            best_chunk = chunk
    assert best is not None
    return {
        "qa_answer": str(best.get("answer") or ""),
        "qa_start": int(best.get("start") or 0),
        "qa_end": int(best.get("end") or 0),
        "qa_score": float(best.get("score") or 0.0),
        "qa_start_logit": float(best.get("start_logit") or 0.0),
        "qa_end_logit": float(best.get("end_logit") or 0.0),
        "reader_tokens": tokens,
        "qa_mode": "independent",
        "qa_chunk_id": None if best_chunk is None else best_chunk.chunk_id,
    }


def write_run(
    output_dir: Path,
    *,
    experiment: str,
    split: str,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "predictions.jsonl", payload["rows"])
    write_jsonl(run_dir / "chunk_scores.jsonl", payload["chunk_scores"])
    write_json(run_dir / "metrics.json", payload["summary"])
    write_json(run_dir / "config_resolved.json", dict(config))
    write_json(
        run_dir / "runtime.json",
        {
            "experiment": experiment,
            "split": split,
            "elapsed_s": payload["summary"].get("elapsed_s"),
            "chunks_processed": payload["summary"].get("chunks_processed"),
            "reader_tokens": payload["summary"].get("reader_tokens"),
            "peak_gpu_mb": payload["summary"].get("peak_gpu_mb"),
            "latency_mean": payload["summary"].get("latency_mean"),
        },
    )


def _gpu_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:
        return 0.0
    return 0.0
