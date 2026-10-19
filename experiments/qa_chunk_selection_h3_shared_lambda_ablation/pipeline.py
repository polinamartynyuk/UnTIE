"""Replay pipeline for one dataset-level lambda shared by all three seeds."""
from __future__ import annotations

import copy
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

from .models import SYSTEM_SHARED, apply_blend_policy, fit_shared_lambda_policies


@dataclass(frozen=True)
class Config:
    source_run: Path
    previous_run: Path
    run_dir: Path
    cells: tuple[str, ...] = DEFAULT_CELLS
    systems: tuple[str, ...] = (SYSTEM_SHARED,)
    seeds: tuple[int, ...] = (42, 43, 44)
    inner_folds: int = 4
    bootstrap: int = 10_000
    resume: bool = False

    def validate(self) -> None:
        if not (self.source_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed source benchmark missing: {self.source_run}")
        if not (self.previous_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed seed-specific blend run missing: {self.previous_run}")
        if set(self.cells) - set(DEFAULT_CELLS):
            raise ValueError(f"unknown cells: {sorted(set(self.cells) - set(DEFAULT_CELLS))}")
        if self.systems != (SYSTEM_SHARED,):
            raise ValueError(f"systems must be exactly {(SYSTEM_SHARED,)}")
        if len(self.seeds) < 2 or self.inner_folds < 2 or self.bootstrap < 100:
            raise ValueError("at least two seeds, inner_folds >= 2 and bootstrap >= 100 are required")

    def payload(self) -> dict:
        return {
            "source_run": str(self.source_run.resolve()),
            "previous_run": str(self.previous_run.resolve()),
            "run_dir": str(self.run_dir.resolve()),
            "cells": list(self.cells),
            "systems": list(self.systems),
            "seeds": list(self.seeds),
            "inner_folds": self.inner_folds,
            "bootstrap": self.bootstrap,
            "resume": self.resume,
        }


class Pipeline(ReplayPipeline):
    def _previous_inputs(self) -> dict[str, str]:
        paths = [
            self.config.previous_run / "test_analysis/primary_metrics.json",
            self.config.previous_run / "test_analysis/question_metrics.json",
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"previous blend artifact missing: {path}")
        return {str(path.relative_to(self.config.previous_run)): sha256(path) for path in paths}

    def _manifest(self) -> None:
        current = self._hash_inputs()
        previous = self._previous_inputs()
        stable = {
            key: value
            for key, value in self.config.payload().items()
            if key not in {"resume", "run_dir"}
        }
        payload = {
            "experiment": "H3 dataset-level shared-lambda ablation",
            "config": stable,
            "source_input_hashes": current,
            "previous_blend_hashes": previous,
            "source_is_read_only": True,
            "selection_contract": (
                "one lambda per dataset/phase selected by mean document-grouped OOF Token F1 "
                "across all seeds; seed-specific models and thresholds"
            ),
            "test_status": "post-hoc exploratory replay on already exposed final TEST",
        }
        path = self.output / "manifest.json"
        if path.is_file() and self.config.resume:
            if read_json(path) != payload:
                raise RuntimeError("resume fingerprint mismatch")
        else:
            write_json(path, payload, immutable=True)
        self.input_hashes = current
        self.previous_hashes = previous

    def _fit_family(self, phase: str, cell: str, rows: list[dict]):
        root = self.output / phase / cell / SYSTEM_SHARED
        shared_path = root / "shared_selection.json"
        policy_paths = {seed: root / f"seed_{seed}/policy.joblib" for seed in self.config.seeds}
        if (
            self.config.resume
            and shared_path.is_file()
            and all(path.is_file() for path in policy_paths.values())
        ):
            self.progress.log("shared-lambda family cache hit", phase=phase, cell=cell)
            return {seed: joblib.load(path) for seed, path in policy_paths.items()}, read_json(shared_path)

        self.progress.log(
            "shared-lambda family fit started",
            phase=phase,
            cell=cell,
            rows=len(rows),
            seeds=list(self.config.seeds),
        )

        def callback(message: str, **fields) -> None:
            self.progress.log(message, phase=phase, cell=cell, system=SYSTEM_SHARED, **fields)

        policies, selection = fit_shared_lambda_policies(
            rows,
            self.config.seeds,
            self.config.inner_folds,
            callback,
        )
        write_json(shared_path, selection)
        for seed, policy in policies.items():
            destination = policy_paths[seed]
            destination.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(policy, destination)
            write_json(
                destination.parent / "selection.json",
                {
                    "phase": phase,
                    "cell": cell,
                    "system": SYSTEM_SHARED,
                    "seed": seed,
                    "shared_lambda": selection["selected_lambda"],
                    "selection": policy.selection,
                },
            )
        self.progress.log(
            "shared-lambda family fit complete",
            phase=phase,
            cell=cell,
            selected_lambda=selection["selected_lambda"],
        )
        return policies, selection

    def development(self) -> None:
        self.progress.stage(
            "development",
            "choose one lambda from mean TRAIN OOF across seeds; evaluate untouched VAL",
        )
        summary = {}
        for index, cell in enumerate(self.config.cells, 1):
            train_rows = read_jsonl(self._source(f"development/{cell}/oof_h2_rows.jsonl"))
            val_rows = read_jsonl(self._source(f"development/{cell}/val_h2_rows.jsonl"))
            policies, selection = self._fit_family("development", cell, train_rows)
            seed_metrics = []
            for seed in self.config.seeds:
                choices, _ = apply_blend_policy(self._seeded(val_rows, seed), policies[seed])
                metric = {"seed": seed, **self._choice_metrics(choices)}
                seed_metrics.append(metric)
                write_json(
                    self.output
                    / "development"
                    / cell
                    / SYSTEM_SHARED
                    / f"seed_{seed}/validation_metrics.json",
                    metric,
                )
            keys = [key for key in seed_metrics[0] if key != "seed"]
            summary[cell] = {
                "selected_lambda": selection["selected_lambda"],
                "per_seed": seed_metrics,
                "mean": {
                    key: float(np.mean([value[key] for value in seed_metrics])) for key in keys
                },
            }
            self.progress.count(index, len(self.config.cells), item="development datasets")
        write_json(self.output / "development/validation_summary.json", summary)
        write_json(self.output / "development/COMPLETE.json", {"status": "COMPLETE"})

    @staticmethod
    def _runtime_decisions(runtime: list[dict], policy, system: str) -> dict[str, dict]:
        del system
        grouped = defaultdict(list)
        for row in runtime:
            grouped[str(row["question_id"])].append(row)
        alternatives = runtime_alternatives(runtime)
        choices, predicted = apply_blend_policy(alternatives, policy) if alternatives else ([], ([], []))
        utility, harm = predicted
        diagnostics = {
            (str(row["question_id"]), int(row["chunk_id"])): {
                "utility": float(utility_value),
                "harm": float(harm_value),
            }
            for row, utility_value, harm_value in zip(alternatives, utility, harm)
        }
        selected = {str(row["question_id"]): int(row["selected_chunk_id"]) for row in choices}
        output = {}
        for qid, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: (int(row["b0_rank"]), int(row["chunk_id"])))
            chunk_id = selected.get(qid, int(ordered[0]["chunk_id"]))
            by_id = {int(row["chunk_id"]): row for row in ordered}
            output[qid] = {
                "selected_chunk_id": chunk_id,
                "selected_row": by_id[chunk_id],
                "candidate_chunk_ids": [int(row["chunk_id"]) for row in ordered],
                "candidate_scores": {
                    str(cid): value
                    for (question, cid), value in diagnostics.items()
                    if question == qid
                },
                "fallback_reason": "NO_ALTERNATIVE_CANDIDATE" if qid not in selected else "NONE",
            }
        return output

    def refit_and_test(self) -> None:
        self.progress.stage(
            "refit-test",
            "choose shared lambda on TRAIN+VAL OOF and replay frozen TEST features",
        )
        summaries, questions = {}, {}
        for index, cell in enumerate(self.config.cells, 1):
            refit_rows = read_jsonl(self._source(f"refit/{cell}/oof_h2_rows.jsonl"))
            runtime = read_jsonl(self._source(f"test_predictions/_runtime_features/{cell}.jsonl"))
            policies, selection = self._fit_family("refit", cell, refit_rows)
            all_rows = []
            for seed in self.config.seeds:
                evaluated = self._evaluate_seed(
                    cell,
                    SYSTEM_SHARED,
                    seed,
                    runtime,
                    policies[seed],
                )
                all_rows.extend(evaluated)
                write_jsonl(
                    self.output
                    / "test_analysis"
                    / SYSTEM_SHARED
                    / f"{cell}_seed_{seed}.jsonl",
                    evaluated,
                )
            averaged = self._average_seeds(all_rows)
            questions[cell] = {SYSTEM_SHARED: averaged}
            summaries[cell] = {SYSTEM_SHARED: self._summary(averaged, len(all_rows))}
            write_jsonl(
                self.output / "test_analysis" / SYSTEM_SHARED / f"{cell}.jsonl",
                all_rows,
            )
            write_json(
                self.output / "test_analysis" / SYSTEM_SHARED / f"{cell}_selection.json",
                selection,
            )
            self.progress.count(index, len(self.config.cells), item="final datasets")
        write_json(self.output / "test_analysis/primary_metrics.json", summaries)
        write_json(self.output / "test_analysis/question_metrics.json", questions)
        write_json(self.output / "test_analysis/COMPLETE.json", {"status": "COMPLETE"})

    def compare(self) -> dict:
        result = super().compare()
        previous_summaries = read_json(self.config.previous_run / "test_analysis/primary_metrics.json")
        previous_questions = read_json(self.config.previous_run / "test_analysis/question_metrics.json")
        current_questions = read_json(self.output / "test_analysis/question_metrics.json")
        for cell in self.config.cells:
            result.setdefault("previous_seed_specific_blend", {})[cell] = previous_summaries[cell]["H3-BLEND"]
            result["paired_bootstrap"][cell][f"{SYSTEM_SHARED}_vs_H3-BLEND"] = paired_bootstrap(
                current_questions[cell][SYSTEM_SHARED],
                previous_questions[cell]["H3-BLEND"],
                self.config.bootstrap,
            )
        write_json(self.output / "comparison.json", result)
        return result

    def report(self, comparison: dict) -> Path:
        lines = [
            "# Dataset-level shared-lambda ablation",
            "",
            "> Post-hoc exploratory replay. One lambda is selected per dataset by mean "
            "document-grouped OOF Token F1 across seeds 42/43/44. TEST labels never enter selection.",
            "",
            "| Dataset | P2/H2 | P4/H3 | Previous seed-specific BLEND | Shared-lambda BLEND | Shared λ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cell in self.config.cells:
            source = comparison["source_baselines"][cell]
            previous = comparison["previous_seed_specific_blend"][cell]
            current = comparison["new_systems"][cell][SYSTEM_SHARED]
            selection = read_json(
                self.output / "refit" / cell / SYSTEM_SHARED / "shared_selection.json"
            )
            lines.append(
                f"| {cell} | {source['P2']['token_f1']:.4f} | {source['P4']['token_f1']:.4f} | "
                f"{previous['token_f1']:.4f} | {current['token_f1']:.4f} | "
                f"{selection['selected_lambda']:.2f} |"
            )
        lines.extend(["", "## Paired document bootstrap", ""])
        for cell in self.config.cells:
            lines.extend([f"### {cell}", "", "| Contrast | Delta Token F1 | 95% CI |", "|---|---:|---:|"])
            for label, value in comparison["paired_bootstrap"][cell].items():
                lines.append(
                    f"| {label} | {value['delta_token_f1']:+.4f} | "
                    f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
                )
            lines.append("")
        lines.extend(
            [
                "## Audit",
                "",
                "The final benchmark and previous seed-specific BLEND artifacts are read-only and hash-checked.",
            ]
        )
        path = self.output / "reports/shared_lambda_results.md"
        write_text(path, "\n".join(lines) + "\n")
        return path

    def _assert_source_unchanged(self) -> None:
        super()._assert_source_unchanged()
        after = self._previous_inputs()
        changed = {
            key: {"before": self.previous_hashes.get(key), "after": value}
            for key, value in after.items()
            if self.previous_hashes.get(key) != value
        }
        write_json(
            self.output / "previous_blend_immutability_check.json",
            {"status": "PASS" if not changed else "FAIL", "changed": changed, "hashes": after},
        )
        if changed:
            raise RuntimeError(f"previous seed-specific blend changed: {sorted(changed)}")
