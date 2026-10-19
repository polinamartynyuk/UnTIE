"""Read-only replay of the consistency-only shared-lambda gate."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib

from experiments.final_official_split_benchmark.analyze import paired_bootstrap
from experiments.final_official_split_benchmark.io_guard import (
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
    write_text,
)
from experiments.final_official_split_benchmark.proposed import apply_runtime, runtime_alternatives
from experiments.qa_chunk_selection_h3_shared_lambda_ablation.models import SYSTEM_SHARED
from experiments.qa_chunk_selection_h3_shrinkage_ablation.models import (
    BlendPolicy,
    apply_blend_policy,
)
from experiments.qa_chunk_selection_h3_shrinkage_ablation.pipeline import (
    DEFAULT_CELLS,
    Pipeline as ReplayPipeline,
)


SYSTEM_CONSISTENCY = "H3-CONSISTENCY-GATE-3"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class Config:
    source_run: Path
    shared_run: Path
    run_dir: Path
    cells: tuple[str, ...] = DEFAULT_CELLS
    systems: tuple[str, ...] = (SYSTEM_CONSISTENCY,)
    seeds: tuple[int, ...] = SEEDS
    inner_folds: int = 4
    bootstrap: int = 10_000
    resume: bool = False

    def validate(self):
        if not (self.source_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed final benchmark missing: {self.source_run}")
        if not (self.shared_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed shared-lambda run missing: {self.shared_run}")
        if set(self.cells) - set(DEFAULT_CELLS):
            raise ValueError(f"unknown cells: {sorted(set(self.cells)-set(DEFAULT_CELLS))}")
        if self.systems != (SYSTEM_CONSISTENCY,) or self.seeds != SEEDS:
            raise ValueError("system and seeds are frozen by the consistency-gate protocol")
        if self.bootstrap < 100:
            raise ValueError("bootstrap must be >= 100")

    def payload(self):
        return {
            "source_run": str(self.source_run.resolve()),
            "shared_run": str(self.shared_run.resolve()),
            "run_dir": str(self.run_dir.resolve()),
            "cells": list(self.cells),
            "systems": list(self.systems),
            "seeds": list(self.seeds),
            "bootstrap": self.bootstrap,
            "resume": self.resume,
        }


def consistency_decision(development_lambda: float, refit_lambda: float) -> dict:
    development_lambda = float(development_lambda)
    refit_lambda = float(refit_lambda)
    stable = development_lambda == refit_lambda
    return {
        "development_lambda": development_lambda,
        "refit_lambda": refit_lambda,
        "exactly_stable": stable,
        "accepted_lambda": refit_lambda if stable else 1.0,
        "fallback_to_h3": not stable,
        "reason": "EXACT_TRAIN_REFIT_AGREEMENT" if stable else "UNSTABLE_SELECTION_FALLBACK_H3",
        "confidence_interval_gate": False,
    }


class Pipeline(ReplayPipeline):
    def _shared_selection_path(self, phase, cell):
        return self.config.shared_run / phase / cell / SYSTEM_SHARED / "shared_selection.json"

    def _policy_path(self, cell, seed, accepted_lambda):
        if accepted_lambda == 0.0:
            return self.config.source_run / "models" / cell / "proposed" / "P2" / f"seed_{seed}/policy.joblib", "P2/H2_ENDPOINT"
        if accepted_lambda == 1.0:
            return self.config.source_run / "models" / cell / "proposed" / "P4" / f"seed_{seed}/policy.joblib", "P4/H3_DEFAULT"
        return self.config.shared_run / "refit" / cell / SYSTEM_SHARED / f"seed_{seed}/policy.joblib", "SHARED_BLEND_POLICY"

    def _gate_plan(self, cell):
        development = read_json(self._shared_selection_path("development", cell))
        refit = read_json(self._shared_selection_path("refit", cell))
        decision = consistency_decision(development["selected_lambda"], refit["selected_lambda"])
        policies = {}
        for seed in self.config.seeds:
            path, origin = self._policy_path(cell, seed, decision["accepted_lambda"])
            if not path.is_file():
                raise FileNotFoundError(f"required frozen policy missing: {path}")
            policies[str(seed)] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "origin": origin,
            }
        return {"cell": cell, "decision": decision, "policies": policies}

    def _shared_inputs(self):
        result = {}
        for cell in self.config.cells:
            for phase in ("development", "refit"):
                path = self._shared_selection_path(phase, cell)
                if not path.is_file():
                    raise FileNotFoundError(path)
                result[str(path.relative_to(self.config.shared_run))] = sha256(path)
        for cell in self.config.cells:
            plan = self._gate_plan(cell)
            for seed, value in plan["policies"].items():
                path = Path(value["path"])
                root = self.config.source_run if self.config.source_run in path.parents else self.config.shared_run
                result[f"policy::{root.name}::{path.relative_to(root)}"] = value["sha256"]
        for relative in ("test_analysis/primary_metrics.json", "test_analysis/question_metrics.json"):
            path = self.config.shared_run / relative
            result[relative] = sha256(path)
        return result

    def _manifest(self):
        source = self._hash_inputs()
        shared = self._shared_inputs()
        stable = {k: v for k, v in self.config.payload().items() if k not in {"resume", "run_dir"}}
        plans = {cell: self._gate_plan(cell) for cell in self.config.cells}
        payload = {
            "experiment": "H3 consistency-only shared-lambda gate",
            "config": stable,
            "source_input_hashes": source,
            "shared_input_hashes": shared,
            "gate_plans": plans,
            "contract": "accept refit lambda iff it exactly equals TRAIN lambda; otherwise canonical H3",
            "confidence_interval_gate": False,
            "source_is_read_only": True,
            "test_status": "post-hoc exploratory replay on already exposed final TEST",
        }
        path = self.output / "manifest.json"
        if path.is_file() and self.config.resume:
            if read_json(path) != payload:
                raise RuntimeError("resume fingerprint mismatch")
        else:
            write_json(path, payload, immutable=True)
        self.input_hashes, self.shared_hashes = source, shared

    def development(self):
        self.progress.stage("gate-audit", "freeze exact TRAIN/refit consistency decisions")
        plans = {}
        for index, cell in enumerate(self.config.cells, 1):
            plan = self._gate_plan(cell)
            plans[cell] = plan
            write_json(self.output / "gate_plans" / f"{cell}.json", plan, immutable=True)
            self.progress.log(
                "consistency decision frozen",
                cell=cell,
                development_lambda=plan["decision"]["development_lambda"],
                refit_lambda=plan["decision"]["refit_lambda"],
                accepted_lambda=plan["decision"]["accepted_lambda"],
                reason=plan["decision"]["reason"],
            )
            self.progress.count(index, len(self.config.cells), item="gate plans")
        write_json(self.output / "development/gate_summary.json", plans, immutable=True)
        write_json(self.output / "development/COMPLETE.json", {"status": "COMPLETE"}, immutable=True)

    @staticmethod
    def _runtime_decisions(runtime, policy, system):
        del system
        if not isinstance(policy, BlendPolicy):
            decisions = apply_runtime(runtime, policy)
            grouped = defaultdict(list)
            for row in runtime:
                grouped[str(row["question_id"])].append(row)
            for qid, decision in decisions.items():
                ordered = sorted(
                    grouped[qid], key=lambda row: (int(row["b0_rank"]), int(row["chunk_id"]))
                )
                by_id = {int(row["chunk_id"]): row for row in ordered}
                decision["selected_row"] = by_id[int(decision["selected_chunk_id"])]
                decision["candidate_chunk_ids"] = [int(row["chunk_id"]) for row in ordered]
            return decisions
        grouped = defaultdict(list)
        for row in runtime:
            grouped[str(row["question_id"])].append(row)
        alternatives = runtime_alternatives(runtime)
        choices, predicted = apply_blend_policy(alternatives, policy) if alternatives else ([], ([], []))
        utility, harm = predicted
        diagnostics = {
            (str(row["question_id"]), int(row["chunk_id"])): {
                "utility": float(u), "harm": float(h)
            }
            for row, u, h in zip(alternatives, utility, harm)
        }
        selected = {str(row["question_id"]): int(row["selected_chunk_id"]) for row in choices}
        output = {}
        for qid, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: (int(row["b0_rank"]), int(row["chunk_id"])))
            cid = selected.get(qid, int(ordered[0]["chunk_id"]))
            by_id = {int(row["chunk_id"]): row for row in ordered}
            output[qid] = {
                "selected_chunk_id": cid,
                "selected_row": by_id[cid],
                "candidate_chunk_ids": [int(row["chunk_id"]) for row in ordered],
                "candidate_scores": {
                    str(chunk): value for (question, chunk), value in diagnostics.items() if question == qid
                },
                "fallback_reason": "NO_ALTERNATIVE_CANDIDATE" if qid not in selected else "NONE",
            }
        return output

    def refit_and_test(self):
        self.progress.stage("test-replay", "apply frozen consistency decisions to frozen TEST features")
        summaries, questions = {}, {}
        for index, cell in enumerate(self.config.cells, 1):
            plan = read_json(self.output / "gate_plans" / f"{cell}.json")
            runtime = read_jsonl(self._source(f"test_predictions/_runtime_features/{cell}.jsonl"))
            all_rows = []
            for seed in self.config.seeds:
                policy = joblib.load(plan["policies"][str(seed)]["path"])
                evaluated = self._evaluate_seed(cell, SYSTEM_CONSISTENCY, seed, runtime, policy)
                all_rows.extend(evaluated)
                write_jsonl(
                    self.output / "test_analysis" / SYSTEM_CONSISTENCY / f"{cell}_seed_{seed}.jsonl",
                    evaluated,
                )
            averaged = self._average_seeds(all_rows)
            questions[cell] = {SYSTEM_CONSISTENCY: averaged}
            summaries[cell] = {SYSTEM_CONSISTENCY: self._summary(averaged, len(all_rows))}
            write_jsonl(self.output / "test_analysis" / SYSTEM_CONSISTENCY / f"{cell}.jsonl", all_rows)
            self.progress.count(index, len(self.config.cells), item="TEST datasets")
        write_json(self.output / "test_analysis/primary_metrics.json", summaries)
        write_json(self.output / "test_analysis/question_metrics.json", questions)
        write_json(self.output / "test_analysis/COMPLETE.json", {"status": "COMPLETE"})

    def compare(self):
        result = super().compare()
        previous_metrics = read_json(self.config.shared_run / "test_analysis/primary_metrics.json")
        previous_questions = read_json(self.config.shared_run / "test_analysis/question_metrics.json")
        current_questions = read_json(self.output / "test_analysis/question_metrics.json")
        for cell in self.config.cells:
            result.setdefault("previous_shared_lambda", {})[cell] = previous_metrics[cell][SYSTEM_SHARED]
            result["paired_bootstrap"][cell][f"{SYSTEM_CONSISTENCY}_vs_previous_shared"] = paired_bootstrap(
                current_questions[cell][SYSTEM_CONSISTENCY],
                previous_questions[cell][SYSTEM_SHARED],
                self.config.bootstrap,
            )
        write_json(self.output / "comparison.json", result)
        return result

    def report(self, comparison):
        lines = [
            "# H3 consistency-only gate results",
            "",
            "> Post-hoc exploratory replay. Exact TRAIN/refit lambda agreement is required; "
            "otherwise the system uses canonical H3/P4. No confidence-interval gate is used.",
            "",
            "| Dataset | P2 | P4 | Previous shared-3 | Consistency gate | Accepted λ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cell in self.config.cells:
            base = comparison["source_baselines"][cell]
            previous = comparison["previous_shared_lambda"][cell]
            current = comparison["new_systems"][cell][SYSTEM_CONSISTENCY]
            decision = read_json(self.output / "gate_plans" / f"{cell}.json")["decision"]
            lines.append(
                f"| {cell} | {base['P2']['token_f1']:.4f} | {base['P4']['token_f1']:.4f} | "
                f"{previous['token_f1']:.4f} | {current['token_f1']:.4f} | "
                f"{decision['accepted_lambda']:.2f} |"
            )
        path = self.output / "reports/consistency_gate_results.md"
        write_text(path, "\n".join(lines) + "\n")
        return path

    def _assert_source_unchanged(self):
        super()._assert_source_unchanged()
        after = self._shared_inputs()
        changed = {
            key: {"before": self.shared_hashes.get(key), "after": value}
            for key, value in after.items()
            if self.shared_hashes.get(key) != value
        }
        write_json(
            self.output / "shared_run_immutability_check.json",
            {"status": "PASS" if not changed else "FAIL", "changed": changed, "hashes": after},
        )
        if changed:
            raise RuntimeError(f"shared-lambda source changed: {sorted(changed)}")
