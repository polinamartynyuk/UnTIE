"""Fresh-process resource benchmark for V9 S0/S1/S2."""
from __future__ import annotations

from collections import defaultdict
import json
import platform
import subprocess
import sys
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .io_guard import read_jsonl, write_json, write_text


def _sample_manifest(output: Path, cells: tuple[str, ...], cap: int) -> dict:
    result = {"contract": "LOCAL outer-fold-0 seed-42; round-robin by aspect and K", "cells": {}}
    for cell in cells:
        rows = read_jsonl(output / "predictions/S0" / cell / "outer_fold_0/seed_42.jsonl")
        strata = defaultdict(list)
        for row in rows:
            runtime = row["runtime"]
            strata[(runtime["canonical_aspect"], int(runtime["effective_k"]))].append(runtime["question_id"])
        for values in strata.values():
            values.sort()
        chosen = []
        while len(chosen) < min(cap, len(rows)) and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(chosen) < min(cap, len(rows)):
                    chosen.append(strata[key].pop(0))
        result["cells"][cell] = {"question_ids": chosen, "questions": len(chosen)}
    write_json(output / "resource_benchmark/sample_manifest.json", result)
    return result


def run_benchmarks(config, output: Path, progress) -> None:
    progress.stage("benchmark", "fresh-process S0/S1/S2 with neural-forward invariance checks")
    sample = _sample_manifest(output, config.datasets, config.benchmark_questions)
    rows = []
    for cell in config.datasets:
        for system in ("S0", "S1", "S2"):
            progress.log("resource worker", cell=cell, system=system)
            command = [
                sys.executable, "-m", "experiments.qa_chunk_selection_v9.benchmark_worker",
                "--cell", cell, "--system", system, "--run-root", str(output),
                "--v5-run", str(config.v5_run), "--device", config.device,
                "--sample-manifest", str(output / "resource_benchmark/sample_manifest.json"),
                "--warmup", "10", "--repeats", "3",
            ]
            completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"resource worker failed {cell}/{system}: {(completed.stderr or completed.stdout)[-8000:]}")
            line = next((x for x in reversed(completed.stdout.splitlines()) if x.strip().startswith("{")), None)
            if line is None:
                raise RuntimeError(f"no benchmark JSON for {cell}/{system}")
            row = json.loads(line)
            rows.append(row)
            write_json(output / f"resource_benchmark/raw/{cell}_{system}.json", row)
    invariance = {}
    for cell in config.datasets:
        s1 = next(x for x in rows if x["cell"] == cell and x["system"] == "S1")
        s2 = next(x for x in rows if x["cell"] == cell and x["system"] == "S2")
        invariance[cell] = {
            "qa_forwards_equal": s1["per_question_forward_counts"] == s2["per_question_forward_counts"],
            "q_a_ranker_forwards_equal": all(
                s1["per_question_forward_counts"][question]["q_a_ranker_forwards"]
                == s2["per_question_forward_counts"][question]["q_a_ranker_forwards"]
                for question in s1["per_question_forward_counts"]
            ),
            "loaded_parameter_count_equal": s1["loaded_parameter_count"] == s2["loaded_parameter_count"],
            "median_latency_ratio": s2["median_latency_seconds"] / max(s1["median_latency_seconds"], 1e-12),
            "p95_latency_ratio": s2["p95_latency_seconds"] / max(s1["p95_latency_seconds"], 1e-12),
            "vram_ratio": (s2["peak_gpu_vram_bytes"] / max(s1["peak_gpu_vram_bytes"], 1)) if s1["peak_gpu_vram_bytes"] is not None else None,
        }
        invariance[cell]["median_latency_target_pass"] = invariance[cell]["median_latency_ratio"] <= 1.10
        invariance[cell]["p95_latency_target_pass"] = invariance[cell]["p95_latency_ratio"] <= 1.15
        invariance[cell]["vram_target_pass"] = (
            invariance[cell]["vram_ratio"] is None or invariance[cell]["vram_ratio"] <= 1.05
        )
        if not all(invariance[cell][x] for x in ("qa_forwards_equal", "q_a_ranker_forwards_equal", "loaded_parameter_count_equal")):
            raise RuntimeError(f"S1/S2 neural resource invariance failed: {cell}")
    write_json(output / "resource_benchmark/summary.json", {
        "runs": rows, "sample_manifest": sample, "systems": ["S0", "S1", "S2"],
        "invariance": invariance,
        "targets": {"median_latency_ratio_max": 1.10, "p95_latency_ratio_max": 1.15, "vram_ratio_max": 1.05},
        "external_references": {
            "Qwen": {"status": "BLOCKED_NO_EXACT_FRESH_RUNNER"},
            "BGE_RU": {"status": "CONTEXT_ONLY"},
        },
    })
    try:
        import torch
        torch_version, cuda = torch.__version__, torch.version.cuda
    except ImportError:
        torch_version, cuda = None, None
    write_json(output / "resource_benchmark/environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch_version,
        "cuda": cuda, "executable": sys.executable,
    })
    lines = [
        "# V9 resource benchmark", "",
        "Fresh process per cell and system; warm-up requests are excluded.", "",
        "| Cell | System | Median s | P95 s | QA forwards | Q-A ranker forwards | Meta params | Peak VRAM bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cell']} | {row['system']} | {row['median_latency_seconds']:.6f} | "
            f"{row['p95_latency_seconds']:.6f} | {row['mean_qa_forwards']:.3f} | "
            f"{row['mean_q_a_ranker_forwards']:.3f} | {row['trainable_meta_parameter_count']} | "
            f"{row['peak_gpu_vram_bytes']} |"
        )
    write_text(output / "reports/resource_benchmark.md", "\n".join(lines) + "\n")

