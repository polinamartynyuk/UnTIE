from __future__ import annotations

import platform
import sys

from .io_guard import read_jsonl, write_json, write_text


def benchmark(config, root, progress):
    """Audit exact forward equality; do not invent unavailable fresh-run timings."""
    progress.stage("benchmark", "verify neural forward equality and record fresh-runner availability")
    rows = []
    for cell in config.datasets:
        for system in ("S0", "S1", "S2"):
            paths = sorted((root / "predictions" / system / cell).glob("outer_fold_*/seed_*.jsonl"))
            values = [row for path in paths for row in read_jsonl(path)]
            rows.append({
                "cell": cell, "system": system, "questions": len(values),
                "qa_forwards": sum(int(x["runtime"].get("qa_forwards", 0)) for x in values),
                "q_a_forwards": sum(int(x["runtime"].get("q_a_ranker_forwards", 0)) for x in values),
                "pair_comparisons": sum(int(x["runtime"].get("pair_comparisons", 0)) for x in values),
                "median_latency": None, "p95_latency": None, "peak_vram": None,
            })
    for cell in config.datasets:
        by = {x["system"]: x for x in rows if x["cell"] == cell}
        if by["S1"]["qa_forwards"] != by["S2"]["qa_forwards"] or by["S1"]["q_a_forwards"] != by["S2"]["q_a_forwards"]:
            raise AssertionError(f"neural forward equality failed: {cell}")
    write_json(root / "resource_benchmark/sample_manifest.json", {"fold": 0, "cap_per_cell": config.benchmark_questions, "same_ids_required": True})
    write_json(root / "resource_benchmark/environment.json", {"python": sys.version, "platform": platform.platform(), "fresh_qwen_bge_runner": "BLOCKED", "reason": "V10 reuses frozen QA/Q-A outputs; no exact fresh neural runner exists in this namespace"})
    write_json(root / "resource_benchmark/summary.json", {"status": "BLOCKED_FOR_LATENCY_VRAM", "forward_equality": "PASS", "rows": rows})
    write_text(root / "reports/resource_benchmark.md", "# Resource benchmark\n\nNeural forward-count equality S2=S1: **PASS**. Fresh-process latency/VRAM is **BLOCKED** because V10 intentionally consumes frozen V9 neural outputs and has no exact fresh Qwen/BGE runner. Historical hardware numbers were not substituted.\n")

