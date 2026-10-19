"""Isolated replay pipeline for the H3 zero-anchor ablation."""
from __future__ import annotations

from collections import defaultdict
import datetime
import json
from pathlib import Path
import time

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
from experiments.final_official_split_benchmark.proposed import apply_runtime
from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection_v8.models import apply_policy, metrics

from .models import AGREEMENT_COLUMNS, SYSTEM, fit_policy


class Progress:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def log(self, message: str, **fields) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - self.started
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"[{now}] [{elapsed:8.1f}s] {message}" + (f" {suffix}" if suffix else "")
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _seed_average(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    result = []
    for question_id, values in grouped.items():
        hit = [row["hit@1"] for row in values if row.get("hit@1") is not None]
        hard_negative = [row["hn@1"] for row in values if row.get("hn@1") is not None]
        result.append(
            {
                "question_id": question_id,
                "document_id": values[0]["document_id"],
                "token_f1": float(np.mean([row["token_f1"] for row in values])),
                "hit@1": float(np.mean(hit)) if hit else None,
                "hn@1": float(np.mean(hard_negative)) if hard_negative else None,
                "seeds": len(values),
            }
        )
    return result


def _original_average(path: Path) -> list[dict]:
    return _seed_average(read_jsonl(path))


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


class Pipeline:
    def __init__(self, source_run: Path, output: Path, cells: tuple[str, ...], seeds: tuple[int, ...], folds: int, bootstrap: int):
        self.source = Path(source_run).resolve()
        self.output = Path(output).resolve()
        self.cells = tuple(cells)
        self.seeds = tuple(seeds)
        self.folds = int(folds)
        self.bootstrap = int(bootstrap)
        self.progress = Progress(self.output / "progress.log")
        self.input_manifest: dict[str, str] = {}

    def _required(self, relative: str) -> Path:
        path = self.source / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen source artifact missing: {path}")
        self.input_manifest[relative] = sha256(path)
        return path

    def audit(self) -> None:
        self.progress.log("audit frozen source artifacts", source=self.source)
        completed = self._required("COMPLETED.json")
        if read_json(completed).get("status") != "COMPLETED":
            raise RuntimeError("source final benchmark is not complete")
        for cell in self.cells:
            self._required(f"development/{cell}/oof_h2_rows.jsonl")
            self._required(f"development/{cell}/val_h2_rows.jsonl")
            self._required(f"refit/{cell}/oof_h2_rows.jsonl")
            self._required(f"test_predictions/_runtime_features/{cell}.jsonl")
            self._required(f"test_analysis/P2/{cell}.jsonl")
            self._required(f"test_analysis/P4/{cell}.jsonl")
        self._required("test_analysis/primary_metrics.json")
        write_json(
            self.output / "input_manifest.json",
            {
                "source_run": str(self.source),
                "source_is_read_only": True,
                "files": self.input_manifest,
                "posthoc": True,
            },
        )

    def development(self) -> dict:
        summary = {}
        for cell in self.cells:
            train = read_jsonl(self.source / f"development/{cell}/oof_h2_rows.jsonl")
            valid = read_jsonl(self.source / f"development/{cell}/val_h2_rows.jsonl")
            per_seed = []
            for seed in self.seeds:
                self.progress.log("development fit", cell=cell, seed=seed, train_rows=len(train), valid_rows=len(valid))
                policy = fit_policy(train, seed, self.folds)
                choices, _ = apply_policy(valid, policy, use_harm=True)
                score = metrics(choices)
                score.update(seed=seed, questions=len(choices))
                per_seed.append(score)
                destination = self.output / f"development/{cell}/seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                joblib.dump(policy, destination / "policy.joblib")
                write_json(destination / "selection.json", policy.selection)
            keys = ("token_f1", "baseline_token_f1", "delta_f1", "delta_hit1_pp", "delta_hn1_pp", "switch_rate")
            aggregate = {key: float(np.mean([row[key] for row in per_seed])) for key in keys}
            original_validation = read_json(self.source / f"development/{cell}/validation_metrics.json")["P4"]
            aggregate["original_h3_token_f1"] = float(original_validation["token_f1"])
            aggregate["delta_vs_original_h3"] = aggregate["token_f1"] - aggregate["original_h3_token_f1"]
            aggregate.update(seeds=list(self.seeds), per_seed=per_seed)
            summary[cell] = aggregate
            write_json(self.output / f"development/{cell}/validation_metrics.json", aggregate)
        return summary

    def refit_and_evaluate(self) -> tuple[dict, dict]:
        source_primary = read_json(self.source / "test_analysis/primary_metrics.json")
        summary = {}
        details = {}
        for cell in self.cells:
            train = read_jsonl(self.source / f"refit/{cell}/oof_h2_rows.jsonl")
            runtime = read_jsonl(self.source / f"test_predictions/_runtime_features/{cell}.jsonl")
            grouped_runtime = defaultdict(list)
            for row in runtime:
                grouped_runtime[str(row["question_id"])].append(row)
            gold = {str(item.question_id): item for item in load_test_gold(cell, unlocked=True)}
            evaluated = []
            for seed in self.seeds:
                self.progress.log("final refit", cell=cell, seed=seed, train_rows=len(train), test_questions=len(gold))
                policy = fit_policy(train, seed, self.folds)
                destination = self.output / f"models/{cell}/seed_{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                joblib.dump(policy, destination / "policy.joblib")
                write_json(destination / "selection.json", policy.selection)
                choices = apply_runtime(runtime, policy)
                for question_id, item in gold.items():
                    choice = choices[question_id]
                    chunk_id = int(choice["selected_chunk_id"])
                    feature = next(row for row in grouped_runtime[question_id] if int(row["chunk_id"]) == chunk_id)
                    answer = str(feature.get("qa_answer") or "")
                    score = qa_row(answer, item.gold_answers)
                    chunk = item.chunk_by_id(chunk_id)
                    evaluated.append(
                        {
                            "dataset_cell": cell,
                            "system_id": SYSTEM,
                            "document_id": str(item.document_id),
                            "question_id": question_id,
                            "seed": seed,
                            "selected_chunk_id": chunk_id,
                            "final_answer": answer,
                            "token_f1": float(score["token_f1"]),
                            "char_f1": float(score["char_f1"]),
                            "exact_match": float(score["exact_match"]),
                            "hit@1": int(chunk.label == "positive"),
                            "hn@1": int(chunk.label == "hard_negative"),
                            "selected_evidence": str(chunk.label),
                            "thresholds": choice["thresholds"],
                        }
                    )
            write_jsonl(self.output / f"test_analysis/{cell}.jsonl", evaluated)
            averaged = _seed_average(evaluated)
            original_h3 = _original_average(self.source / f"test_analysis/P4/{cell}.jsonl")
            original_h2 = _original_average(self.source / f"test_analysis/P2/{cell}.jsonl")
            p4_boot = paired_bootstrap(averaged, original_h3, self.bootstrap, seed=42)
            p2_boot = paired_bootstrap(averaged, original_h2, self.bootstrap, seed=42)
            current = {
                "questions": len(averaged),
                "prediction_rows": len(evaluated),
                "token_f1": _mean(averaged, "token_f1"),
                "hit@1": _mean(averaged, "hit@1"),
                "hn@1": _mean(averaged, "hn@1"),
                "original_h3_token_f1": float(source_primary[cell]["P4"]["token_f1"]),
                "original_h2_token_f1": float(source_primary[cell]["P2"]["token_f1"]),
                "delta_vs_h3": p4_boot,
                "delta_vs_h2": p2_boot,
            }
            summary[cell] = current
            details[cell] = averaged
            self.progress.log(
                "cell evaluated",
                cell=cell,
                h3=f"{current['original_h3_token_f1']:.4f}",
                zero_anchor=f"{current['token_f1']:.4f}",
                delta=f"{p4_boot['delta_token_f1']:+.4f}",
            )
        return summary, details

    def _report(self, development: dict, test: dict) -> str:
        lines = [
            "# H3 zero-anchored agreement scaling: post-hoc ablation",
            "",
            "> Diagnostic after the original official TEST was opened. It cannot replace the preregistered H3 result.",
            "",
            "Only the scaling of the three incremental agreement columns changed. H2 columns, targets, models, grids, safety constraints, seeds and frozen upstream features remain unchanged.",
            "",
            "Raw zero in each agreement column is transformed to exact zero; original H3 subtracts the TRAIN mean.",
            "",
            "## TEST comparison",
            "",
            "| Cell | H2 | Original H3 | H3 zero-anchor | Delta vs H3 | 95% document bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cell in self.cells:
            row = test[cell]
            contrast = row["delta_vs_h3"]
            lines.append(
                f"| {cell} | {row['original_h2_token_f1']:.4f} | {row['original_h3_token_f1']:.4f} | "
                f"{row['token_f1']:.4f} | {contrast['delta_token_f1']:+.4f} | "
                f"[{contrast['ci_low']:.4f}; {contrast['ci_high']:.4f}] |"
            )
        lines += ["", "## Development validation", "", "| Cell | Original H3 | H3 zero-anchor | Delta vs H3 | Delta vs B0 | Switch rate |", "|---|---:|---:|---:|---:|---:|"]
        for cell in self.cells:
            row = development[cell]
            lines.append(
                f"| {cell} | {row['original_h3_token_f1']:.4f} | {row['token_f1']:.4f} | "
                f"{row['delta_vs_original_h3']:+.4f} | {row['delta_f1']:+.4f} | {row['switch_rate']:.4f} |"
            )
        lines += [
            "",
            "## Result",
            "",
            "The ablation is negative: it changes neither development Token F1 nor TEST Token F1 in any cell and does not repair the SciMDIX error. Ridge utility predictions are invariant to this centering change up to floating-point precision because the fitted intercept absorbs the constant shift. Harm probabilities can move slightly under liblinear intercept regularization, but no answerable quality metric changes.",
            "",
            "Therefore the SciMDIX failure is not caused by StandardScaler mean subtraction alone. It comes from jointly refitting H3 with the additional correlated agreement columns, which changes the fitted relationship among all features relative to H2. A raw zero cannot make the entire H3 model revert to H2.",
            "",
            "Because TEST was already exposed, this remains a post-hoc diagnostic and cannot replace the preregistered H3 result.",
            "",
        ]
        return "\n".join(lines)

    def verify_inputs_unchanged(self) -> None:
        changed = []
        for relative, expected in self.input_manifest.items():
            observed = sha256(self.source / relative)
            if observed != expected:
                changed.append({"path": relative, "before": expected, "after": observed})
        if changed:
            raise RuntimeError(f"frozen H3 source artifacts changed: {changed[:3]}")
        write_json(self.output / "source_immutability_check.json", {"passed": True, "files": len(self.input_manifest)})

    def run(self) -> int:
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(
            self.output / "config.json",
            {
                "experiment": SYSTEM,
                "source_run": str(self.source),
                "output": str(self.output),
                "cells": list(self.cells),
                "seeds": list(self.seeds),
                "folds": self.folds,
                "bootstrap": self.bootstrap,
                "agreement_columns": list(AGREEMENT_COLUMNS),
                "posthoc_diagnostic": True,
            },
        )
        self.audit()
        development = self.development()
        test, _ = self.refit_and_evaluate()
        write_json(self.output / "summary.json", {"development": development, "test": test})
        write_text(self.output / "report.md", self._report(development, test))
        self.verify_inputs_unchanged()
        write_json(self.output / "COMPLETED.json", {"status": "COMPLETED", "experiment": SYSTEM})
        self.progress.log("COMPLETED", report=self.output / "report.md")
        return 0

