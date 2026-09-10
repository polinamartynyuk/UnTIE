"""D4 oracle: same frozen QA on gold-localized windows. No selector training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

from experiments.qa_chunk_selection.metrics import mean_or_nan, qa_row
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.positive_quality import band
from experiments.qa_chunk_selection.v3_1_auto_prep.qa_runner import run_qa

from .windows import WINDOW_TYPES, as_window_chunk, windows_for_positive


def run_d4_oracle(
    records: Sequence[InstanceRecord],
    *,
    answer_fn: Callable,
    cache,
    model_id: str,
    qa_answerer=None,
    progress: ProgressLogger | None = None,
    skip_bertscore: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    log = progress or ProgressLogger()
    rows: list[dict[str, Any]] = []
    total = max(sum(len(item.positives) for item in records), 1)
    index = 0
    for item in records:
        for chunk in item.positives:
            index += 1
            windows = windows_for_positive(item, chunk)
            scored: dict[str, Any] = {}
            for name in WINDOW_TYPES:
                window = windows[name]
                if not window.get("localized") or not str(window.get("text") or "").strip():
                    scored[name] = {"token_f1": None, "exact_match": None, "answer": "", "localized": False}
                    continue
                payload = run_qa(
                    answer_fn,
                    item.question,
                    as_window_chunk(chunk, window),
                    cache=cache,
                    model_id=model_id,
                    inference_hash=f"v3_d45_d4_{name}",
                    qa_answerer=qa_answerer,
                )
                metrics = qa_row(payload.get("qa_predicted_span"), item.gold_answers, skip_bertscore=skip_bertscore)
                scored[name] = {
                    "token_f1": float(metrics.get("token_f1") or 0.0),
                    "exact_match": float(metrics.get("exact_match") or 0.0),
                    "answer": payload.get("qa_predicted_span"),
                    "localized": True,
                    "local_start": window.get("local_start"),
                    "local_end": window.get("local_end"),
                }
            full_f1 = float(scored["full_chunk"].get("token_f1") or 0.0)
            row = {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "chunk_id": chunk.chunk_id,
                "full_band": band(full_f1),
                "windows": scored,
                "delta_gold_sentence": _delta(scored, "gold_sentence"),
                "delta_minimal": _delta(scored, "minimal_gold_window"),
                "delta_sentence_pm1": _delta(scored, "sentence_pm1"),
            }
            rows.append(row)
            log.count(index, total, item="d4_oracle")
    return rows, summarize_d4(rows)


def summarize_d4(rows: Sequence[dict[str, Any]], *, delta_f1_min: float = 1.0) -> dict[str, Any]:
    by_window: dict[str, list[float]] = {name: [] for name in WINDOW_TYPES}
    by_aspect: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    hard = [row for row in rows if row.get("full_band") in {"reader_hard", "reader_failed"}]
    for row in rows:
        for name in WINDOW_TYPES:
            value = ((row.get("windows") or {}).get(name) or {}).get("token_f1")
            if value is None:
                continue
            by_window[name].append(float(value))
            by_aspect[str(row.get("aspect"))][name].append(float(value))
    means = {name: mean_or_nan(values) for name, values in by_window.items()}
    full = float(means.get("full_chunk") or 0.0)
    best_name = max((name for name in WINDOW_TYPES if name != "full_chunk"), key=lambda name: float(means.get(name) or -1.0), default="gold_sentence")
    best = float(means.get(best_name) or 0.0)
    hard_full = mean_or_nan([float(((row.get("windows") or {}).get("full_chunk") or {}).get("token_f1") or 0.0) for row in hard])
    hard_best = mean_or_nan(
        [
            max(
                (
                    float(((row.get("windows") or {}).get(name) or {}).get("token_f1") or 0.0)
                    for name in WINDOW_TYPES
                    if ((row.get("windows") or {}).get(name) or {}).get("token_f1") is not None
                ),
                default=0.0,
            )
            for row in hard
        ]
    )
    gain = best - full
    return {
        "n_positives": len(rows),
        "n_localized_gold_sentence": sum(1 for row in rows if ((row.get("windows") or {}).get("gold_sentence") or {}).get("localized")),
        "mean_token_f1": means,
        "mean_token_f1_by_aspect": {
            aspect: {name: mean_or_nan(values) for name, values in windows.items()} for aspect, windows in by_aspect.items()
        },
        "best_window": best_name,
        "oracle_gain_vs_full": gain,
        "hard_or_failed": {"n": len(hard), "full_chunk": hard_full, "best_window": hard_best},
        "runtime_d4_allowed": bool(rows) and gain >= float(delta_f1_min),
        "delta_f1_min": float(delta_f1_min),
        "note": "oracle uses gold to choose the window; this is not a trained runtime selector",
    }


def _delta(scored: dict[str, Any], name: str) -> float | None:
    full = (scored.get("full_chunk") or {}).get("token_f1")
    other = (scored.get(name) or {}).get("token_f1")
    if full is None or other is None:
        return None
    return float(other) - float(full)
