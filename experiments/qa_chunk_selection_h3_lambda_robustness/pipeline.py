"""Isolated robustness experiment for shared-lambda selection."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from experiments.final_official_split_benchmark.analyze import paired_bootstrap
from experiments.final_official_split_benchmark.io_guard import (
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
    write_text,
)
from experiments.final_official_split_benchmark.proposed import runtime_alternatives
from experiments.qa_chunk_selection_h3_shrinkage_ablation.pipeline import (
    DEFAULT_CELLS,
    Pipeline as ReplayPipeline,
)

from .models import (
    SEEDS_10,
    SEEDS_3,
    SYSTEMS,
    SYSTEM_SHARED_10,
    SYSTEM_STABILITY_10,
    SYSTEM_STABILITY_3,
    apply_blend_policy,
    build_policies,
    lambda_evidence,
    prepare_states,
    select_shared,
    stability_decision,
)


SYSTEM_SEEDS = {
    SYSTEM_SHARED_10: SEEDS_10,
    SYSTEM_STABILITY_3: SEEDS_3,
    SYSTEM_STABILITY_10: SEEDS_10,
}


@dataclass(frozen=True)
class Config:
    source_run: Path
    previous_run: Path
    run_dir: Path
    cells: tuple[str, ...] = DEFAULT_CELLS
    systems: tuple[str, ...] = SYSTEMS
    seeds: tuple[int, ...] = SEEDS_10
    inner_folds: int = 4
    bootstrap: int = 10_000
    gate_bootstrap: int = 5_000
    resume: bool = False

    def validate(self):
        if not (self.source_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed source benchmark missing: {self.source_run}")
        if not (self.previous_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed previous shared-lambda run missing: {self.previous_run}")
        if set(self.cells) - set(DEFAULT_CELLS):
            raise ValueError(f"unknown cells: {sorted(set(self.cells)-set(DEFAULT_CELLS))}")
        if self.systems != SYSTEMS or self.seeds != SEEDS_10:
            raise ValueError("systems and seed sets are frozen by the robustness protocol")
        if self.inner_folds < 2 or self.bootstrap < 100 or self.gate_bootstrap < 100:
            raise ValueError("invalid folds/bootstrap configuration")

    def payload(self):
        return {
            "source_run": str(self.source_run.resolve()),
            "previous_run": str(self.previous_run.resolve()),
            "run_dir": str(self.run_dir.resolve()),
            "cells": list(self.cells),
            "systems": list(self.systems),
            "seeds": list(self.seeds),
            "seed_sets": {"three": list(SEEDS_3), "ten": list(SEEDS_10)},
            "inner_folds": self.inner_folds,
            "bootstrap": self.bootstrap,
            "gate_bootstrap": self.gate_bootstrap,
            "resume": self.resume,
        }


class Pipeline(ReplayPipeline):
    def _previous_inputs(self):
        paths = [
            self.config.previous_run / "test_analysis/primary_metrics.json",
            self.config.previous_run / "test_analysis/question_metrics.json",
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        return {str(path.relative_to(self.config.previous_run)): sha256(path) for path in paths}

    def _manifest(self):
        current = self._hash_inputs()
        previous = self._previous_inputs()
        stable = {k: v for k, v in self.config.payload().items() if k not in {"resume", "run_dir"}}
        payload = {
            "experiment": "H3 lambda robustness: 10 seeds and stability gate",
            "config": stable,
            "source_input_hashes": current,
            "previous_shared_lambda_hashes": previous,
            "source_is_read_only": True,
            "stability_gate": {
                "default": "lambda=1/H3",
                "accept_non_h3_only_if": [
                    "exact same lambda selected on TRAIN and TRAIN+VAL",
                    "document-bootstrap lower 95% OOF CI versus lambda=1 is above zero",
                ],
            },
            "test_status": "post-hoc exploratory replay on already exposed final TEST",
        }
        path = self.output / "manifest.json"
        if path.is_file() and self.config.resume:
            if read_json(path) != payload:
                raise RuntimeError("resume fingerprint mismatch")
        else:
            write_json(path, payload, immutable=True)
        self.input_hashes, self.previous_hashes = current, previous

    def _expected_policy_paths(self, phase, cell):
        return {
            system: {
                seed: self.output / phase / cell / system / f"seed_{seed}/policy.joblib"
                for seed in SYSTEM_SEEDS[system]
            }
            for system in SYSTEMS
        }

    def _fit_phase(self, phase, cell, rows):
        root = self.output / phase / cell
        summary_path = root / "lambda_robustness_selection.json"
        paths = self._expected_policy_paths(phase, cell)
        if self.config.resume and summary_path.is_file() and all(
            path.is_file() for values in paths.values() for path in values.values()
        ):
            self.progress.log("lambda robustness cache hit", phase=phase, cell=cell)
            policies = {
                system: {seed: joblib.load(path) for seed, path in values.items()}
                for system, values in paths.items()
            }
            return policies, read_json(summary_path)

        self.progress.log("lambda robustness fit started", phase=phase, cell=cell, rows=len(rows))

        def callback(message, **fields):
            self.progress.log(message, phase=phase, cell=cell, **fields)

        states = prepare_states(rows, SEEDS_10, self.config.inner_folds, callback)
        shared3 = select_shared(states, SEEDS_3)
        shared10 = select_shared(states, SEEDS_10)
        evidence3 = lambda_evidence(
            states, SEEDS_3, shared3["selected_lambda"], self.config.gate_bootstrap
        )
        evidence10 = lambda_evidence(
            states, SEEDS_10, shared10["selected_lambda"], self.config.gate_bootstrap
        )

        if phase == "development":
            gate3 = {
                "accepted_lambda": shared3["selected_lambda"],
                "gate_pass": None,
                "reason": "PROVISIONAL_DEVELOPMENT_SELECTION",
                "evidence": evidence3,
            }
            gate10 = {
                "accepted_lambda": shared10["selected_lambda"],
                "gate_pass": None,
                "reason": "PROVISIONAL_DEVELOPMENT_SELECTION",
                "evidence": evidence10,
            }
        else:
            development = read_json(
                self.output / "development" / cell / "lambda_robustness_selection.json"
            )
            gate3 = stability_decision(development["shared_3"], shared3, evidence3)
            gate10 = stability_decision(development["shared_10"], shared10, evidence10)

        policies = {
            SYSTEM_SHARED_10: build_policies(
                states,
                SEEDS_10,
                shared10["selected_lambda"],
                SYSTEM_SHARED_10,
                shared10,
            ),
            SYSTEM_STABILITY_3: build_policies(
                states,
                SEEDS_3,
                gate3["accepted_lambda"],
                SYSTEM_STABILITY_3,
                shared3,
                gate3,
            ),
            SYSTEM_STABILITY_10: build_policies(
                states,
                SEEDS_10,
                gate10["accepted_lambda"],
                SYSTEM_STABILITY_10,
                shared10,
                gate10,
            ),
        }
        summary = {
            "phase": phase,
            "cell": cell,
            "shared_3": shared3,
            "shared_10": shared10,
            "evidence_3": evidence3,
            "evidence_10": evidence10,
            "gate_3": gate3,
            "gate_10": gate10,
            "systems": {
                SYSTEM_SHARED_10: {
                    "seeds": list(SEEDS_10),
                    "accepted_lambda": shared10["selected_lambda"],
                },
                SYSTEM_STABILITY_3: {
                    "seeds": list(SEEDS_3),
                    "accepted_lambda": gate3["accepted_lambda"],
                },
                SYSTEM_STABILITY_10: {
                    "seeds": list(SEEDS_10),
                    "accepted_lambda": gate10["accepted_lambda"],
                },
            },
        }
        write_json(summary_path, summary)
        for system, values in policies.items():
            for seed, policy in values.items():
                path = paths[system][seed]
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(policy, path)
                write_json(path.parent / "selection.json", policy.selection)
        self.progress.log(
            "lambda robustness fit complete",
            phase=phase,
            cell=cell,
            shared_3=shared3["selected_lambda"],
            shared_10=shared10["selected_lambda"],
            stability_3=gate3["accepted_lambda"],
            stability_10=gate10["accepted_lambda"],
        )
        return policies, summary

    def development(self):
        self.progress.stage("development", "TRAIN OOF seed-count and provisional gate analysis")
        output = {}
        for index, cell in enumerate(self.config.cells, 1):
            train = read_jsonl(self._source(f"development/{cell}/oof_h2_rows.jsonl"))
            val = read_jsonl(self._source(f"development/{cell}/val_h2_rows.jsonl"))
            policies, selection = self._fit_phase("development", cell, train)
            output[cell] = {"selection": selection, "systems": {}}
            for system in SYSTEMS:
                metrics = []
                for seed in SYSTEM_SEEDS[system]:
                    choices, _ = apply_blend_policy(self._seeded(val, seed), policies[system][seed])
                    metrics.append({"seed": seed, **self._choice_metrics(choices)})
                keys = [key for key in metrics[0] if key != "seed"]
                output[cell]["systems"][system] = {
                    "per_seed": metrics,
                    "mean": {key: float(np.mean([row[key] for row in metrics])) for key in keys},
                }
            self.progress.count(index, len(self.config.cells), item="development datasets")
        write_json(self.output / "development/validation_summary.json", output)
        write_json(self.output / "development/COMPLETE.json", {"status": "COMPLETE"})

    @staticmethod
    def _runtime_decisions(runtime, policy, system):
        del system
        grouped = defaultdict(list)
        for row in runtime:
            grouped[str(row["question_id"])].append(row)
        alternatives = runtime_alternatives(runtime)
        choices, predicted = apply_blend_policy(alternatives, policy) if alternatives else ([], ([], []))
        utility, harm = predicted
        diagnostics = {
            (str(row["question_id"]), int(row["chunk_id"])): {
                "utility": float(u),
                "harm": float(h),
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
                    str(chunk): value
                    for (question, chunk), value in diagnostics.items()
                    if question == qid
                },
                "fallback_reason": "NO_ALTERNATIVE_CANDIDATE" if qid not in selected else "NONE",
            }
        return output

    def refit_and_test(self):
        self.progress.stage("refit-test", "TRAIN+VAL OOF stability gate and frozen TEST replay")
        summaries, questions = {}, {}
        for index, cell in enumerate(self.config.cells, 1):
            train = read_jsonl(self._source(f"refit/{cell}/oof_h2_rows.jsonl"))
            runtime = read_jsonl(self._source(f"test_predictions/_runtime_features/{cell}.jsonl"))
            policies, selection = self._fit_phase("refit", cell, train)
            summaries[cell], questions[cell] = {}, {}
            for system in SYSTEMS:
                all_rows = []
                for seed in SYSTEM_SEEDS[system]:
                    evaluated = self._evaluate_seed(cell, system, seed, runtime, policies[system][seed])
                    all_rows.extend(evaluated)
                    write_jsonl(
                        self.output / "test_analysis" / system / f"{cell}_seed_{seed}.jsonl",
                        evaluated,
                    )
                averaged = self._average_seeds(all_rows)
                questions[cell][system] = averaged
                summaries[cell][system] = self._summary(averaged, len(all_rows))
                write_jsonl(self.output / "test_analysis" / system / f"{cell}.jsonl", all_rows)
            write_json(self.output / "test_analysis/selections" / f"{cell}.json", selection)
            self.progress.count(index, len(self.config.cells), item="final datasets")
        write_json(self.output / "test_analysis/primary_metrics.json", summaries)
        write_json(self.output / "test_analysis/question_metrics.json", questions)
        write_json(self.output / "test_analysis/COMPLETE.json", {"status": "COMPLETE"})

    def compare(self):
        result = super().compare()
        previous_metrics = read_json(self.config.previous_run / "test_analysis/primary_metrics.json")
        previous_questions = read_json(self.config.previous_run / "test_analysis/question_metrics.json")
        current_questions = read_json(self.output / "test_analysis/question_metrics.json")
        for cell in self.config.cells:
            result.setdefault("previous_shared_lambda", {})[cell] = previous_metrics[cell]["H3-SHARED-LAMBDA"]
            for system in SYSTEMS:
                result["paired_bootstrap"][cell][f"{system}_vs_previous_shared"] = paired_bootstrap(
                    current_questions[cell][system],
                    previous_questions[cell]["H3-SHARED-LAMBDA"],
                    self.config.bootstrap,
                )
        write_json(self.output / "comparison.json", result)
        return result

    def report(self, comparison):
        lines = [
            "# H3 lambda robustness results",
            "",
            "> Post-hoc exploratory replay. The stability gate defaults to H3 and accepts shrinkage "
            "only after exact TRAIN/refit agreement and positive lower 95% OOF bootstrap bound.",
            "",
            "| Dataset | P2 | P4 | Previous shared-3 | Shared-10 | Stability-3 | Stability-10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for cell in self.config.cells:
            base = comparison["source_baselines"][cell]
            prev = comparison["previous_shared_lambda"][cell]
            new = comparison["new_systems"][cell]
            lines.append(
                f"| {cell} | {base['P2']['token_f1']:.4f} | {base['P4']['token_f1']:.4f} | "
                f"{prev['token_f1']:.4f} | {new[SYSTEM_SHARED_10]['token_f1']:.4f} | "
                f"{new[SYSTEM_STABILITY_3]['token_f1']:.4f} | "
                f"{new[SYSTEM_STABILITY_10]['token_f1']:.4f} |"
            )
        lines.extend(["", "## Final gate decisions", ""])
        for cell in self.config.cells:
            selection = read_json(self.output / "refit" / cell / "lambda_robustness_selection.json")
            lines.append(
                f"- **{cell}**: raw-3={selection['shared_3']['selected_lambda']}, "
                f"gate-3={selection['gate_3']['accepted_lambda']} ({selection['gate_3']['reason']}); "
                f"raw-10={selection['shared_10']['selected_lambda']}, "
                f"gate-10={selection['gate_10']['accepted_lambda']} ({selection['gate_10']['reason']})."
            )
        path = self.output / "reports/lambda_robustness_results.md"
        write_text(path, "\n".join(lines) + "\n")
        return path

    def _assert_source_unchanged(self):
        super()._assert_source_unchanged()
        after = self._previous_inputs()
        changed = {
            key: {"before": self.previous_hashes.get(key), "after": value}
            for key, value in after.items()
            if self.previous_hashes.get(key) != value
        }
        write_json(
            self.output / "previous_shared_immutability_check.json",
            {"status": "PASS" if not changed else "FAIL", "changed": changed, "hashes": after},
        )
        if changed:
            raise RuntimeError(f"previous shared-lambda result changed: {sorted(changed)}")
