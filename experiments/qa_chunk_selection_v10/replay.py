from __future__ import annotations

from pathlib import Path

from .io_guard import read_jsonl, write_json


def _signature(rows):
    output = []
    for row in rows:
        runtime = row["runtime"]; evaluation = row["evaluation"]
        top_k = runtime.get("b0_top_k", [])
        candidates = [(int(x["chunk_id"]), int(x["rank"])) for x in top_k]
        answers = [(int(runtime["base_chunk_id"]), str(runtime.get("baseline_answer") or ""))]
        answers += [(int(x["chunk_id"]), str(x.get("answer") or "")) for x in runtime.get("evaluated_candidates", [])]
        output.append((
            str(runtime["question_id"]), tuple(candidates), tuple(sorted(answers)),
            int(runtime["selected_chunk_id"]), str(runtime["final_answer"]),
            int(evaluation["changed_top1"]), float(evaluation["token_f1"]),
        ))
    return sorted(output)


def replay(config, destination: Path, progress):
    reports = {"S0": {}, "S1_H2": {}}
    for index, cell in enumerate(config.datasets, 1):
        progress.count(index, len(config.datasets), item="S0/H2 exact historical replay", extra={"cell": cell})
        for fold in config.outer_folds:
            for seed in config.seeds:
                paths = {
                    "S0": (
                        config.v8_run / "branch_hrr/predictions/local" / cell / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "H0.jsonl",
                        config.v9_run / "predictions/S0" / cell / f"outer_fold_{fold}" / f"seed_{seed}.jsonl",
                    ),
                    "S1_H2": (
                        config.v8_run / "branch_hrr/predictions/local" / cell / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "H2.jsonl",
                        config.v9_run / "predictions/S1" / cell / f"outer_fold_{fold}" / f"seed_{seed}.jsonl",
                    ),
                }
                for system, (left, right) in paths.items():
                    if not left.is_file() or not right.is_file(): raise FileNotFoundError(f"replay input missing: {left} / {right}")
                    match = _signature(read_jsonl(left)) == _signature(read_jsonl(right))
                    key = f"{cell}/outer_fold_{fold}/seed_{seed}"
                    reports[system][key] = {"status": "PASS" if match else "FAIL", "v8": str(left), "v9": str(right)}
                    if not match: raise RuntimeError(f"H2_REPLAY_FAILED: {key}/{system}")
    write_json(destination / "s0_replay.json", reports["S0"])
    write_json(destination / "s1_h2_replay.json", reports["S1_H2"])
    return reports
