"""Isolated replay of two H3 shrinkage hypotheses on frozen final features.

This module only reads the completed final benchmark.  It never imports or
overwrites the fitted P2/P4 policies and writes every artifact below its own
run directory.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from experiments.final_official_split_benchmark.analyze import paired_bootstrap
from experiments.final_official_split_benchmark.contract import CELL_SPECS
from experiments.final_official_split_benchmark.data import load_test_gold
from experiments.final_official_split_benchmark.io_guard import (
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
    write_text,
)
from experiments.final_official_split_benchmark.proposed import runtime_alternatives
from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import rouge_l_f1 as unicode_rouge_l
from experiments.qa_chunk_selection.v4_4_ru_ablation.lemma import RussianLemmatizer

from .models import APPLIERS, FITTERS, SYSTEM_BLEND, SYSTEM_GROUP


DEFAULT_CELLS = ("scirex", "scier_128", "scier_384", "russerrc", "scimdix")
DEFAULT_SYSTEMS = (SYSTEM_BLEND, SYSTEM_GROUP)


@dataclass(frozen=True)
class Config:
    source_run: Path
    run_dir: Path
    cells: tuple[str, ...] = DEFAULT_CELLS
    systems: tuple[str, ...] = DEFAULT_SYSTEMS
    seeds: tuple[int, ...] = (42, 43, 44)
    inner_folds: int = 4
    bootstrap: int = 10_000
    resume: bool = False

    def validate(self) -> None:
        if not (self.source_run / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"completed source benchmark missing: {self.source_run}")
        unknown_cells = set(self.cells) - set(DEFAULT_CELLS)
        unknown_systems = set(self.systems) - set(DEFAULT_SYSTEMS)
        if unknown_cells or unknown_systems:
            raise ValueError(f"unknown cells={sorted(unknown_cells)} systems={sorted(unknown_systems)}")
        if not self.seeds or self.inner_folds < 2 or self.bootstrap < 100:
            raise ValueError("seeds must be non-empty, inner_folds >= 2, bootstrap >= 100")

    def payload(self) -> dict:
        return {
            "source_run": str(self.source_run.resolve()),
            "run_dir": str(self.run_dir.resolve()),
            "cells": list(self.cells),
            "systems": list(self.systems),
            "seeds": list(self.seeds),
            "inner_folds": self.inner_folds,
            "bootstrap": self.bootstrap,
            "resume": self.resume,
        }


class Pipeline:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.output = config.run_dir
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time()
        self.input_hashes: dict[str, str] = {}

    def _status(self, status: str, **extra) -> None:
        write_json(
            self.output / "status.json",
            {"status": status, "elapsed_s": time.time() - self.started, **extra},
        )

    def _heartbeat(self, stop: threading.Event) -> None:
        while not stop.wait(30):
            self.progress.log(
                "still running",
                elapsed_seconds=round(time.time() - self.started, 1),
            )

    def _source(self, relative: str) -> Path:
        path = self.config.source_run / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen source artifact missing: {path}")
        return path

    def _hash_inputs(self) -> dict[str, str]:
        paths = []
        for cell in self.config.cells:
            paths.extend(
                [
                    self._source(f"development/{cell}/oof_h2_rows.jsonl"),
                    self._source(f"development/{cell}/val_h2_rows.jsonl"),
                    self._source(f"refit/{cell}/oof_h2_rows.jsonl"),
                    self._source(f"test_predictions/_runtime_features/{cell}.jsonl"),
                    self._source(f"test_analysis/P2/{cell}.jsonl"),
                    self._source(f"test_analysis/P4/{cell}.jsonl"),
                ]
            )
        paths.extend(
            [
                self._source("test_analysis/primary_metrics.json"),
                self._source("test_analysis/question_metrics.json"),
                self._source("manifests/evidence_label_availability.json"),
            ]
        )
        return {str(path.relative_to(self.config.source_run)): sha256(path) for path in sorted(set(paths))}

    def _manifest(self) -> None:
        current = self._hash_inputs()
        path = self.output / "manifest.json"
        stable = {key: value for key, value in self.config.payload().items() if key not in {"resume", "run_dir"}}
        payload = {
            "experiment": "H3 shrinkage ablation: convex H2/H3 blend and agreement group ridge",
            "config": stable,
            "source_input_hashes": current,
            "source_is_read_only": True,
            "test_status": "post-hoc replay on already exposed final TEST; exploratory, not a new blind confirmation",
        }
        if path.is_file() and self.config.resume:
            old = read_json(path)
            if old != payload:
                raise RuntimeError("resume fingerprint mismatch")
        else:
            write_json(path, payload, immutable=True)
        self.input_hashes = current

    @staticmethod
    def _seeded(rows: list[dict], seed: int) -> list[dict]:
        values = copy.deepcopy(rows)
        for row in values:
            row["seed"] = int(seed)
        return values

    def _progress_callback(self, *, phase: str, cell: str, system: str, seed: int):
        def callback(message: str, **fields) -> None:
            self.progress.log(
                message,
                phase=phase,
                cell=cell,
                system=system,
                seed=seed,
                **fields,
            )

        return callback

    def _fit_or_load(self, phase: str, cell: str, system: str, seed: int, rows: list[dict]):
        root = self.output / phase / cell / system / f"seed_{seed}"
        model_path = root / "policy.joblib"
        selection_path = root / "selection.json"
        if self.config.resume and model_path.is_file() and selection_path.is_file():
            self.progress.log("policy cache hit", phase=phase, cell=cell, system=system, seed=seed)
            return joblib.load(model_path)
        self.progress.log(
            "policy fit started",
            phase=phase,
            cell=cell,
            system=system,
            seed=seed,
            rows=len(rows),
        )
        policy = FITTERS[system](
            self._seeded(rows, seed),
            seed,
            self.config.inner_folds,
            self._progress_callback(phase=phase, cell=cell, system=system, seed=seed),
        )
        root.mkdir(parents=True, exist_ok=True)
        joblib.dump(policy, model_path)
        write_json(
            selection_path,
            {
                "phase": phase,
                "cell": cell,
                "system": system,
                "seed": seed,
                "selection": policy.selection,
            },
        )
        self.progress.log("policy fit complete", phase=phase, cell=cell, system=system, seed=seed)
        return policy

    @staticmethod
    def _choice_metrics(choices: list[dict]) -> dict:
        mean = lambda key: float(np.mean([row[key] for row in choices])) if choices else 0.0
        return {
            "questions": len(choices),
            "token_f1": mean("f1"),
            "baseline_token_f1": mean("base_f1"),
            "delta_token_f1": mean("f1") - mean("base_f1"),
            "hit@1": mean("hit1"),
            "hn@1": mean("hn1"),
            "switch_rate": mean("switch"),
        }

    def development(self) -> None:
        self.progress.stage("development", "TRAIN OOF selection followed by untouched VAL evaluation")
        summary = {}
        total = len(self.config.cells) * len(self.config.systems) * len(self.config.seeds)
        current = 0
        for cell in self.config.cells:
            train_rows = read_jsonl(self._source(f"development/{cell}/oof_h2_rows.jsonl"))
            val_rows = read_jsonl(self._source(f"development/{cell}/val_h2_rows.jsonl"))
            summary[cell] = {}
            for system in self.config.systems:
                seed_metrics = []
                for seed in self.config.seeds:
                    policy = self._fit_or_load("development", cell, system, seed, train_rows)
                    choices, _ = APPLIERS[system](self._seeded(val_rows, seed), policy)
                    metric = {"seed": seed, **self._choice_metrics(choices)}
                    seed_metrics.append(metric)
                    write_json(
                        self.output / "development" / cell / system / f"seed_{seed}" / "validation_metrics.json",
                        metric,
                    )
                    current += 1
                    self.progress.count(current, total, item="development policies")
                keys = [key for key in seed_metrics[0] if key != "seed"]
                summary[cell][system] = {
                    "per_seed": seed_metrics,
                    "mean": {key: float(np.mean([row[key] for row in seed_metrics])) for key in keys},
                }
        write_json(self.output / "development/validation_summary.json", summary)
        write_json(self.output / "development/COMPLETE.json", {"status": "COMPLETE"})

    @staticmethod
    def _runtime_decisions(runtime: list[dict], policy, system: str) -> dict[str, dict]:
        grouped = defaultdict(list)
        for row in runtime:
            grouped[str(row["question_id"])].append(row)
        alternatives = runtime_alternatives(runtime)
        choices, predicted = APPLIERS[system](alternatives, policy) if alternatives else ([], ([], []))
        utility, harm = predicted
        diagnostics = {
            (str(row["question_id"]), int(row["chunk_id"])): {"utility": float(u), "harm": float(h)}
            for row, u, h in zip(alternatives, utility, harm)
        }
        selected = {str(row["question_id"]): int(row["selected_chunk_id"]) for row in choices}
        output = {}
        for qid, rows in grouped.items():
            ordered = sorted(rows, key=lambda row: (int(row["b0_rank"]), int(row["chunk_id"])))
            chunk_id = selected.get(qid, int(ordered[0]["chunk_id"]))
            by_id = {int(row["chunk_id"]): row for row in ordered}
            if chunk_id not in by_id:
                raise RuntimeError(f"{qid}: policy selected unknown chunk {chunk_id}")
            output[qid] = {
                "selected_chunk_id": chunk_id,
                "selected_row": by_id[chunk_id],
                "candidate_chunk_ids": [int(row["chunk_id"]) for row in ordered],
                "candidate_scores": {
                    str(cid): value for (question, cid), value in diagnostics.items() if question == qid
                },
                "fallback_reason": "NO_ALTERNATIVE_CANDIDATE" if qid not in selected else "NONE",
            }
        return output

    def _evaluate_seed(self, cell: str, system: str, seed: int, runtime: list[dict], policy) -> list[dict]:
        decisions = self._runtime_decisions(self._seeded(runtime, seed), policy, system)
        gold = {str(item.question_id): item for item in load_test_gold(cell, unlocked=True)}
        availability = read_json(self._source("manifests/evidence_label_availability.json"))[cell]
        lemmatizer = RussianLemmatizer(smoke=False)
        missing = set(gold) - set(decisions)
        if missing:
            raise RuntimeError(
                f"{cell}/{system}/seed_{seed}: missing {len(missing)} canonical answerable questions"
            )
        rows = []
        for qid, item in gold.items():
            decision = decisions[qid]
            chosen = decision["selected_row"]
            answer = str(chosen.get("qa_answer") or "")
            metric = qa_row(answer, item.gold_answers)
            chunk = item.chunk_by_id(int(decision["selected_chunk_id"]))
            rows.append(
                {
                    "dataset_cell": cell,
                    "system_id": system,
                    "document_id": item.document_id,
                    "question_id": qid,
                    "seed": seed,
                    "selected_chunk_id": int(chunk.chunk_id),
                    "final_answer": answer,
                    "token_f1": metric["token_f1"],
                    "char_f1": metric["char_f1"],
                    "unicode_rouge_l": max(
                        (unicode_rouge_l(answer, value) for value in item.gold_answers), default=0.0
                    ),
                    "exact_match": metric["exact_match"],
                    "lemma_f1": lemmatizer.gold_f1(answer, item.gold_answers)
                    if CELL_SPECS[cell].language == "ru"
                    else None,
                    "hit@1": int(chunk.label == "positive") if availability["positive_labels"] else None,
                    "hit@k": int(
                        any(item.chunk_by_id(cid).label == "positive" for cid in decision["candidate_chunk_ids"])
                    )
                    if availability["positive_labels"]
                    else None,
                    "hn@1": int(chunk.label == "hard_negative")
                    if availability["hard_negative_labels"]
                    else None,
                    "safe_rate": int(chunk.label == "positive") if availability["positive_labels"] else None,
                    "unsafe_rate": int(chunk.label == "hard_negative")
                    if availability["hard_negative_labels"]
                    else None,
                    "f1_selected_positive": metric["token_f1"] if chunk.label == "positive" else None,
                    "selected_evidence": chunk.label,
                    "posthoc_policy": decision,
                }
            )
        return rows

    @staticmethod
    def _average_seeds(rows: list[dict]) -> list[dict]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["question_id"]].append(row)
        output = []
        for qid, values in grouped.items():
            first = values[0]
            result = {
                "question_id": qid,
                "document_id": first["document_id"],
                "seeds": len(values),
            }
            for key in (
                "token_f1",
                "char_f1",
                "unicode_rouge_l",
                "exact_match",
                "lemma_f1",
                "hit@1",
                "hit@k",
                "hn@1",
                "safe_rate",
                "unsafe_rate",
                "f1_selected_positive",
            ):
                clean = [row[key] for row in values if row.get(key) is not None]
                result[key] = float(np.mean(clean)) if clean else None
            output.append(result)
        return sorted(output, key=lambda row: row["question_id"])

    @staticmethod
    def _summary(rows: list[dict], prediction_rows: int) -> dict:
        def mean(key):
            clean = [row[key] for row in rows if row.get(key) is not None]
            return float(np.mean(clean)) if clean else None

        return {
            "questions": len(rows),
            "prediction_rows": prediction_rows,
            "seeds_per_question": sorted({row["seeds"] for row in rows}),
            **{
                key: mean(key)
                for key in (
                    "token_f1",
                    "char_f1",
                    "unicode_rouge_l",
                    "exact_match",
                    "lemma_f1",
                    "hit@1",
                    "hit@k",
                    "hn@1",
                    "safe_rate",
                    "unsafe_rate",
                    "f1_selected_positive",
                )
            },
        }

    def refit_and_test(self) -> None:
        self.progress.stage("refit-test", "TRAIN+VAL OOF refit and post-hoc replay on frozen TEST features")
        summaries, questions = {}, {}
        total = len(self.config.cells) * len(self.config.systems) * len(self.config.seeds)
        current = 0
        for cell in self.config.cells:
            refit_rows = read_jsonl(self._source(f"refit/{cell}/oof_h2_rows.jsonl"))
            runtime = read_jsonl(self._source(f"test_predictions/_runtime_features/{cell}.jsonl"))
            summaries[cell], questions[cell] = {}, {}
            for system in self.config.systems:
                all_rows = []
                for seed in self.config.seeds:
                    policy = self._fit_or_load("refit", cell, system, seed, refit_rows)
                    evaluated = self._evaluate_seed(cell, system, seed, runtime, policy)
                    all_rows.extend(evaluated)
                    write_jsonl(
                        self.output / "test_analysis" / system / f"{cell}_seed_{seed}.jsonl", evaluated
                    )
                    current += 1
                    self.progress.count(current, total, item="final policies")
                averaged = self._average_seeds(all_rows)
                questions[cell][system] = averaged
                summaries[cell][system] = self._summary(averaged, len(all_rows))
                write_jsonl(self.output / "test_analysis" / system / f"{cell}.jsonl", all_rows)
        write_json(self.output / "test_analysis/primary_metrics.json", summaries)
        write_json(self.output / "test_analysis/question_metrics.json", questions)
        write_json(self.output / "test_analysis/COMPLETE.json", {"status": "COMPLETE"})

    def compare(self) -> dict:
        self.progress.stage("compare", "compare both variants with frozen H2/P2 and H3/P4")
        summaries = read_json(self.output / "test_analysis/primary_metrics.json")
        questions = read_json(self.output / "test_analysis/question_metrics.json")
        source_summaries = read_json(self._source("test_analysis/primary_metrics.json"))
        source_questions = read_json(self._source("test_analysis/question_metrics.json"))
        contrasts = {}
        for cell in self.config.cells:
            contrasts[cell] = {}
            for system in self.config.systems:
                for baseline in ("P2", "P4"):
                    contrasts[cell][f"{system}_vs_{baseline}"] = paired_bootstrap(
                        questions[cell][system],
                        source_questions[cell][baseline],
                        self.config.bootstrap,
                    )
        result = {
            "new_systems": summaries,
            "source_baselines": {
                cell: {baseline: source_summaries[cell][baseline] for baseline in ("P2", "P4")}
                for cell in self.config.cells
            },
            "paired_bootstrap": contrasts,
            "interpretation_scope": "post-hoc exploratory comparison; hyperparameters use TRAIN OOF only",
        }
        write_json(self.output / "comparison.json", result)
        return result

    def report(self, comparison: dict) -> Path:
        lines = [
            "# H3 shrinkage ablation: results",
            "",
            "> This is a post-hoc exploratory replay over the already exposed final TEST features. "
            "All lambda, penalty, model and threshold choices are made from TRAIN/TRAIN+VAL OOF rows, never TEST labels.",
            "",
            "## Systems",
            "",
            "- **H3-BLEND**: convex blend of separately fitted H2 and H3 utility/harm predictions; "
            "lambda is selected from `0, .25, .5, .75, 1` inside grouped OOF.",
            "- **H3-GROUP-RIDGE**: the H3 feature schema with a separate L2 multiplier for the three "
            "agreement fields; multiplier is selected from `1, 2, 4, 8, 16` inside grouped OOF.",
            "",
            "## TEST Token F1",
            "",
            "| Dataset | P2/H2 | P4/H3 | H3-BLEND | H3-GROUP-RIDGE |",
            "|---|---:|---:|---:|---:|",
        ]
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        for cell in self.config.cells:
            base = comparison["source_baselines"][cell]
            new = comparison["new_systems"][cell]
            blend = new.get(SYSTEM_BLEND, {}).get("token_f1")
            group = new.get(SYSTEM_GROUP, {}).get("token_f1")
            lines.append(
                f"| {cell} | {fmt(base['P2']['token_f1'])} | {fmt(base['P4']['token_f1'])} | "
                f"{fmt(blend)} | {fmt(group)} |"
            )
        lines.extend(["", "## Paired document bootstrap", ""])
        for cell in self.config.cells:
            lines.append(f"### {cell}")
            lines.append("")
            lines.append("| Contrast | Delta Token F1 | 95% CI |")
            lines.append("|---|---:|---:|")
            for label, value in comparison["paired_bootstrap"][cell].items():
                lines.append(
                    f"| {label} | {value['delta_token_f1']:+.4f} | "
                    f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] |"
                )
            lines.append("")
        lines.extend(
            [
                "## Selection audit",
                "",
                "Every seed/cell policy directory contains `selection.json` with the complete OOF grid, "
                "selected lambda or agreement penalty, alpha/C, thresholds, and safety metrics.",
                "",
                "Source benchmark files are hash-checked before and after the run; the experiment aborts "
                "if any frozen source input changes.",
            ]
        )
        path = self.output / "reports/h3_shrinkage_results.md"
        write_text(path, "\n".join(lines) + "\n")
        return path

    def _assert_source_unchanged(self) -> None:
        after = self._hash_inputs()
        changed = {
            key: {"before": self.input_hashes.get(key), "after": value}
            for key, value in after.items()
            if self.input_hashes.get(key) != value
        }
        write_json(
            self.output / "source_immutability_check.json",
            {"status": "PASS" if not changed else "FAIL", "changed": changed, "hashes": after},
        )
        if changed:
            raise RuntimeError(f"frozen source benchmark changed during replay: {sorted(changed)}")

    def run(self) -> int:
        stop = threading.Event()
        thread = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True)
        thread.start()
        try:
            self._manifest()
            self._status("RUNNING")
            if not (self.config.resume and (self.output / "development/COMPLETE.json").is_file()):
                self.development()
            else:
                self.progress.log("development already complete")
            if not (self.config.resume and (self.output / "test_analysis/COMPLETE.json").is_file()):
                self.refit_and_test()
            else:
                self.progress.log("refit and TEST replay already complete")
            comparison = self.compare()
            report_path = self.report(comparison)
            self._assert_source_unchanged()
            write_json(
                self.output / "COMPLETED.json",
                {
                    "status": "COMPLETED",
                    "report": str(report_path.resolve()),
                    "source_benchmark_unchanged": True,
                    "scope": "post-hoc exploratory ablation",
                },
            )
            self._status("COMPLETED")
            self.progress.log("experiment complete", report=str(report_path.resolve()))
            return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc))
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            thread.join(1)
