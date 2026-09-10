"""V3 inference: reuse V2 gold-free scoring, then attach overlay metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v2.schemas import RuntimeInstance, rank_ids, to_runtime
from experiments.qa_chunk_selection.v2.timing import timed_block

from .freeze import assert_eval_split_allowed
from .io_util import write_json, write_jsonl
from .metrics_v3 import enrich_v3, summarize_v3

ScoreFn = Callable[[RuntimeInstance], list[float]]
AnswerFn = Callable[[RuntimeInstance, Sequence[int]], dict[str, Any]]


def run_inference_v3(
    records: Sequence[InstanceRecord],
    score_fn: ScoreFn,
    *,
    ranking_method: str,
    split: str,
    output_dir,
    config,
    overlay: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    progress=None,
    answer_fn: AnswerFn | None = None,
    skip_qa: bool = False,
    flags: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assert_eval_split_allowed(
        split,
        allow_internal_test=bool(config.allow_internal_test),
        output_dir=Path(config.output_dir),
        require_freeze=True,
    )
    gold_allowed = split != "internal_test" or bool(config.allow_internal_test)
    flag_map = flag_index(flags or [])
    overlay = overlay or {}
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
            raw = build_raw_prediction(
                runtime,
                ranking_method=ranking_method,
                scores=scores,
                ranked_ids=ranked,
                qa=qa,
            )
            raw_rows.append(raw)
            if gold_allowed:
                scored_rows.append(
                    enrich_v3(
                        raw,
                        gold,
                        overlay,
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
        summary = summarize_v3(scored_rows, ks=config.top_k_values)
        summary.update(payload)
        write_json(run_dir / "metrics.json", summary)
        payload = summary
    else:
        write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "runtime.json", {"elapsed_s": timing["elapsed_s"], "peak_gpu_mb": timing["peak_gpu_mb"]})
    return {"raw": raw_rows, "scored": scored_rows, "summary": payload, "run_dir": str(run_dir)}
