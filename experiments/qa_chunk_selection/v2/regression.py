"""E0-R1 metric replay and optional E0-R2 checkpoint inference replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.qa_chunk_selection.io_util import load_json, read_jsonl
from experiments.qa_chunk_selection.metrics import summarize_ranking

from .io_util import write_json

E0_COMPARE = {
    "a": ("hit@1", "token_f1"),
    "b": ("hit@1", "hn@1", "pairwise_pos_gt_hn", "token_f1"),
    "c": ("token_f1",),
    "d1": ("hit@1",),
    "d2": ("hit@1", "token_f1"),
}


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return abs(float(left) - float(right)) <= float(atol) + float(rtol) * abs(float(right))


def regression_r1(e0_dir: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared: dict[str, Any] = {}
    for method, keys in E0_COMPARE.items():
        pred_path = e0_dir / "runs" / method / "42" / "dev" / "predictions.jsonl"
        metrics_path = e0_dir / "runs" / method / "42" / "dev" / "metrics.json"
        if not pred_path.exists() or not metrics_path.exists():
            mismatches.append({"method": method, "error": "missing E0 artifacts"})
            continue
        rows = read_jsonl(pred_path)
        recomputed = summarize_ranking(rows, ks=(1, 5, 10))
        saved = load_json(metrics_path)
        method_cmp = {}
        for key in keys:
            old = saved.get(key)
            new = recomputed.get(key)
            ok = old is not None and new is not None and _close(float(new), float(old), atol=atol, rtol=rtol)
            method_cmp[key] = {"saved": old, "recomputed": new, "ok": ok}
            if not ok:
                mismatches.append({"method": method, "metric": key, "saved": old, "recomputed": new})
        compared[method] = method_cmp
    return {"level": "E0-R1", "ok": not mismatches, "compared": compared, "mismatches": mismatches}


def regression_r2_compare(replay_metrics: dict[str, Any], e0_dir: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared: dict[str, Any] = {}
    for method, keys in E0_COMPARE.items():
        saved_path = e0_dir / "runs" / method / "42" / "dev" / "metrics.json"
        replay = replay_metrics.get(method)
        if not saved_path.exists():
            mismatches.append({"method": method, "error": "missing E0 metrics"})
            continue
        if not isinstance(replay, dict) or not replay:
            mismatches.append({"method": method, "error": "missing replay summary"})
            continue
        saved = load_json(saved_path)
        method_cmp = {}
        for key in keys:
            ok = key in replay and _close(float(replay[key]), float(saved[key]), atol=atol, rtol=rtol)
            method_cmp[key] = {"saved": saved.get(key), "replay": replay.get(key), "ok": ok}
            if not ok:
                mismatches.append({"method": method, "metric": key, "saved": saved.get(key), "replay": replay.get(key)})
        compared[method] = method_cmp
    return {"level": "E0-R2", "ok": not mismatches, "compared": compared, "mismatches": mismatches}


def write_regression_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
