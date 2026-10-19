from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import rouge_l_f1 as unicode_rouge_l
from experiments.final_official_split_benchmark.analyze import paired_bootstrap, family_paired_bootstrap
from experiments.final_official_split_benchmark.config import Config as SourceConfig
from experiments.final_official_split_benchmark.contract import CELL_SPECS
from experiments.final_official_split_benchmark.data import load_test_gold, load_test_runtime
from experiments.final_official_split_benchmark.engine import CellEngine, release_cuda
from experiments.final_official_split_benchmark.io_guard import read_json, sha256, write_json, write_jsonl, write_text

from .reader import FrozenSquadReader


SYSTEMS = ("B5S-B0K", "B5S-ALL")


def _resolve_device(value: str) -> str:
    import torch
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _prediction(cell, system, item, chunk, answer, candidate_ids, model_id, revision):
    return {"runtime": {
        "dataset_cell": cell, "dataset": CELL_SPECS[cell].family, "split": "TEST",
        "language": "en", "system_id": system, "document_id": item.document_id,
        "canonical_source_document_id": item.document_id, "question_id": item.question_id,
        "aspect": item.aspect, "question": item.question, "selected_chunk_id": int(chunk.chunk_id),
        "final_answer": str(answer["answer"]), "model_seed": 0,
        "qa_confidence": float(answer["confidence"]), "answer_confidence": float(answer["confidence"]),
        "candidate_chunk_ids": [int(value) for value in candidate_ids], "candidate_k": len(candidate_ids),
        "reader_model": model_id, "reader_revision": revision, "reader_training": "SQuAD2 only",
        "posthoc": True, "failure_reason": None,
    }}


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.source = Path(args.source_run)
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time()

    def _status(self, status, **extra):
        write_json(self.output / "status.json", {"status": status, "elapsed_s": time.time() - self.started, **extra})

    def _resolve_snapshot(self):
        from huggingface_hub import snapshot_download
        self.progress.stage("model", "resolve pinned frozen ModernBERT-SQuAD2 snapshot")
        path = Path(snapshot_download(
            repo_id=self.args.model_id, revision=self.args.model_revision,
            local_files_only=not self.args.allow_downloads,
            allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.py"],
        ))
        required = [path / "config.json", path / "model.safetensors", path / "tokenizer.json"]
        missing = [str(value) for value in required if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete SQuAD2 snapshot: {missing}")
        return path

    def _assert_source(self):
        if read_json(self.source / "COMPLETED.json").get("status") != "COMPLETED":
            raise RuntimeError("source benchmark is not complete")
        for cell in self.args.cells:
            if CELL_SPECS[cell].language != "en":
                raise ValueError(f"English-only SQuAD2 baseline cannot run on {cell}")
            for path in (self.source / "models" / cell / "B0_final" / "model", self.source / "development" / cell / "frozen_selection.json"):
                if not path.exists():
                    raise FileNotFoundError(path)

    def _predict_cell(self, cell: str, reader: FrozenSquadReader):
        marker = self.output / "predictions" / f"{cell}_COMPLETE.json"
        if self.args.resume and marker.is_file():
            self.progress.log("prediction cell already complete", cell=cell)
            return
        self.progress.stage(f"predict:{cell}", "frozen SQuAD2 reader on B0 top-K and all chunks")
        source_config = SourceConfig(run_dir=self.source, cells=(cell,), device=self.args.device, resume=True)
        engine = CellEngine(source_config, cell, self.progress)
        selection = read_json(self.source / "development" / cell / "frozen_selection.json")
        engine.select_questions((), frozen=selection["question_mapping"])
        items = engine.apply_questions(load_test_runtime(cell))
        scorer = engine.load_b0(self.source / "models" / cell / "B0_final")
        outputs = {system: [] for system in SYSTEMS}
        for index, item_runtime in enumerate(items, 1):
            from experiments.final_official_split_benchmark.engine import runtime_as_unlabeled
            item = runtime_as_unlabeled(item_runtime)
            scores = engine.scores(scorer, item)
            b0_top = sorted(item.chunks, key=lambda value: (-scores[value.chunk_id], value.chunk_id))[: engine.spec.k]
            all_chunks = [value for value in item.chunks if value.eligible_for_qa]
            union = {int(value.chunk_id): value for value in (*all_chunks, *b0_top)}
            answers = reader.answer_chunks(item.question, union.values())
            for system, candidates in (("B5S-B0K", b0_top), ("B5S-ALL", all_chunks)):
                chosen = max(candidates, key=lambda value: (answers[value.chunk_id]["confidence"], -int(value.chunk_id)))
                outputs[system].append(_prediction(
                    cell, system, item, chosen, answers[chosen.chunk_id], [value.chunk_id for value in candidates],
                    self.args.model_id, self.args.model_revision,
                ))
            self.progress.count(index, len(items), item=f"{cell} questions", extra={"all_chunks": len(all_chunks)})
        hashes = {}
        for system, rows in outputs.items():
            path = self.output / "predictions" / system / f"{cell}.jsonl"
            write_jsonl(path, rows, immutable=True)
            hashes[system] = {"rows": len(rows), "sha256": sha256(path)}
        write_json(marker, {"status": "COMPLETE", "cell": cell, "systems": hashes}, immutable=True)
        release_cuda(scorer, engine)

    def _evaluate(self):
        self.progress.stage("evaluate", "canonical answerable TEST metrics; explicitly post-hoc")
        summaries, questions = {}, {}
        for cell in self.args.cells:
            gold = {str(value.question_id): value for value in load_test_gold(cell, unlocked=True)}
            summaries[cell], questions[cell] = {}, {}
            for system in SYSTEMS:
                predictions = [json.loads(line) for line in (self.output / "predictions" / system / f"{cell}.jsonl").read_text().splitlines() if line.strip()]
                by_qid = {str(value["runtime"]["question_id"]): value["runtime"] for value in predictions}
                missing = set(gold) - set(by_qid)
                if missing:
                    raise RuntimeError(f"{cell}/{system}: missing evaluation questions: {sorted(missing)[:5]}")
                rows = []
                for qid, item in gold.items():
                    runtime = by_qid[qid]; answer = str(runtime.get("final_answer") or "")
                    metric = qa_row(answer, item.gold_answers); chunk = item.chunk_by_id(int(runtime["selected_chunk_id"]))
                    rows.append({
                        "dataset_cell": cell, "system_id": system, "document_id": item.document_id,
                        "question_id": qid, "seed": 0, "selected_chunk_id": int(chunk.chunk_id),
                        "final_answer": answer, "token_f1": metric["token_f1"], "char_f1": metric["char_f1"],
                        "unicode_rouge_l": max((unicode_rouge_l(answer, value) for value in item.gold_answers), default=0.0),
                        "exact_match": metric["exact_match"], "hit@1": int(chunk.label == "positive"),
                    })
                write_jsonl(self.output / "evaluation" / system / f"{cell}.jsonl", rows, immutable=True)
                questions[cell][system] = rows
                summaries[cell][system] = {
                    "questions": len(rows),
                    **{key: float(np.mean([value[key] for value in rows])) for key in ("token_f1", "char_f1", "unicode_rouge_l", "exact_match", "hit@1")},
                }
                self.progress.log("evaluated", cell=cell, system=system, token_f1=round(summaries[cell][system]["token_f1"], 4))
        write_json(self.output / "evaluation" / "primary_metrics.json", summaries, immutable=True)
        write_json(self.output / "evaluation" / "question_metrics.json", questions, immutable=True)
        return summaries, questions

    @staticmethod
    def _english_balanced(summaries, system):
        scier = float(np.mean([summaries["scier_128"][system]["token_f1"], summaries["scier_384"][system]["token_f1"]]))
        scirex = float(summaries["scirex"][system]["token_f1"])
        return {"SciREX": scirex, "SciER": scier, "EN_BALANCED": float(np.mean([scirex, scier]))}

    def _statistics(self, summaries, questions):
        original_questions = read_json(self.source / "test_analysis" / "question_metrics.json")
        original_metrics = read_json(self.source / "test_analysis" / "primary_metrics.json")
        merged = {}
        per_cell = {}
        comparisons = (("B5S-B0K", "B5-EN"), ("B5S-ALL", "B5-EN"), ("B5S-B0K", "P2"),
                       ("B5S-ALL", "P2"), ("B5S-ALL", "B5S-B0K"), ("B5S-ALL", "B1"))
        for cell in self.args.cells:
            merged[cell] = {**original_questions[cell], **questions[cell]}
            per_cell[cell] = {}
            for left, right in comparisons:
                per_cell[cell][f"{left}_vs_{right}"] = paired_bootstrap(merged[cell][left], merged[cell][right], self.args.bootstrap)
        family = {}
        for left, right in comparisons:
            family[f"{left}_vs_{right}"] = family_paired_bootstrap(merged, left, right, self.args.bootstrap, list(self.args.cells))
        balanced = {system: self._english_balanced(summaries, system) for system in SYSTEMS}
        for system in ("B1", "B2", "B3", "B4b", "B5-EN", "P2", "P4"):
            combined = {cell: {system: original_metrics[cell][system]} for cell in self.args.cells}
            balanced[system] = self._english_balanced(combined, system)
        result = {"posthoc": True, "per_cell_contrasts": per_cell, "english_family_contrasts": family,
                  "english_balanced": balanced, "bootstrap_iterations": self.args.bootstrap}
        write_json(self.output / "statistics.json", result, immutable=True)
        return result

    def _report(self, summaries, statistics):
        lines = [
            "# Post-hoc frozen ModernBERT-SQuAD2 baselines", "",
            "> These systems were specified after the original TEST results were visible. They are post-hoc diagnostics and do not alter the confirmatory benchmark.", "",
            f"- Source run: `{self.source}`", f"- Reader: `{self.args.model_id}`", f"- Revision: `{self.args.model_revision}`",
            "- Reader training: public SQuAD 2.0 only; no SciREX or SciER fine-tuning.",
            "- B5S-B0K: frozen SQuAD2 reader on the source run's B0 top-K, selected by QA confidence.",
            "- B5S-ALL: the same reader on every eligible canonical chunk, selected only by QA confidence.", "",
            "## Token F1", "", "| System | SciREX | SciER384 | SciER128 | EN balanced |", "|---|---:|---:|---:|---:|",
        ]
        for system in ("B5S-B0K", "B5S-ALL", "B5-EN", "P2", "P4"):
            source_metrics = read_json(self.source / "test_analysis" / "primary_metrics.json") if system not in SYSTEMS else summaries
            b = statistics["english_balanced"][system]["EN_BALANCED"]
            lines.append(f"| {system} | {source_metrics['scirex'][system]['token_f1']:.4f} | {source_metrics['scier_384'][system]['token_f1']:.4f} | {source_metrics['scier_128'][system]['token_f1']:.4f} | {b:.4f} |")
        lines.extend(["", "## Paired bootstrap contrasts", "", "Positive delta favors the first system.", "",
                      "| Scope | Contrast | Delta | 95% CI |", "|---|---|---:|---:|"])
        for scope, values in [("EN balanced", statistics["english_family_contrasts"]), *[(cell, statistics["per_cell_contrasts"][cell]) for cell in self.args.cells]]:
            for name, value in values.items():
                lines.append(f"| {scope} | {name.replace('_vs_', ' − ')} | {value['delta_token_f1']:+.4f} | [{value['ci_low']:.4f}; {value['ci_high']:.4f}] |")
        lines.extend(["", "## Interpretation guard", "", "The original immutable predictions and generated report were not modified. Any combined table must label B5S-B0K and B5S-ALL as post-hoc.", ""])
        path = self.output / "report.md"; write_text(path, "\n".join(lines), immutable=True); return path

    def run(self):
        try:
            self._status("RUNNING"); self._assert_source(); snapshot = self._resolve_snapshot()
            manifest = {
                "status": "POSTHOC", "source_run": str(self.source.resolve()), "cells": list(self.args.cells),
                "systems": list(SYSTEMS), "model_id": self.args.model_id, "model_revision": self.args.model_revision,
                "base_model": self.args.base_model, "snapshot": str(snapshot),
                "model_sha256": sha256(snapshot / "model.safetensors"),
                "remote_code_executed": False, "test_results_visible_when_specified": True,
            }
            write_json(self.output / "manifest.json", manifest, immutable=self.args.resume)
            reader = FrozenSquadReader(snapshot, _resolve_device(self.args.device), base_model=self.args.base_model,
                                       max_length=self.args.max_length, batch_size=self.args.batch_size)
            for cell in self.args.cells:
                self._predict_cell(cell, reader)
            summaries, questions = self._evaluate(); statistics = self._statistics(summaries, questions); report = self._report(summaries, statistics)
            self._status("COMPLETED", report=str(report)); write_json(self.output / "COMPLETED.json", {"status": "COMPLETED", "report": str(report)}, immutable=True)
            self.progress.log("post-hoc SQuAD2 baselines complete", report=str(report)); return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc)); self.progress.warn(f"FAILED {type(exc).__name__}: {exc}"); raise

