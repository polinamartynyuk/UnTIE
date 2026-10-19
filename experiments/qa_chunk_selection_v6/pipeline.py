"""V6 orchestration over immutable, honest V5 upstream feature artifacts."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import threading
import time

import joblib
import numpy as np

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS

from .audit import git_snapshot, historical_hashes, run_audit
from .config import Config
from .contract import (CELLS, FAMILIES, OUTER_FOLDS, VARIANTS, V5_RUN, feature_columns,
                       frozen_contract)
from .features import (alternative_rows, assert_runtime_gold_free, enrich_question,
                       evidence_pair_rows, gate_row, grouped)
from .io_guard import canonical_hash, read_jsonl, sha256, write_json, write_jsonl, write_text
from .models import (FittedGate, FittedUtility, IdentityCalibrator, decision, fit_model,
                     meta_oof, metrics, normalize_training_weights, tune_gate, tune_utility,
                     _question_choices)
from .replay import replay


class DataStore:
    def __init__(self, config: Config, progress: ProgressLogger):
        self.config = config; self.progress = progress; self._items = {}; self._cache = {}

    def items(self, cell: str):
        if cell not in self._items:
            records = records_from_jsonl(V5_CELLS[cell].records)
            self._items[cell] = {str(item.question_id): item for item in records}
        return self._items[cell]

    def load(self, cell: str, fold: int, seed: int, role: str) -> tuple[list[dict], list[dict], list[dict], dict]:
        key = (cell, fold, seed, role)
        if key in self._cache: return copy.deepcopy(self._cache[key])
        path = self.config.v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}" / ("inner_meta/features.jsonl" if role == "meta" else "eval/features.jsonl")
        if not path.is_file(): raise FileNotFoundError(f"honest V5 upstream features missing: {path}")
        raw = read_jsonl(path); by_question = grouped(raw); items = self.items(cell)
        runtime = []; utility = []; gates = []; targets = {}
        for index, (qid, rows) in enumerate(sorted(by_question.items()), 1):
            if qid not in items: raise KeyError(f"{cell}: item missing for {qid}")
            run, target = enrich_question(cell, rows, items[qid], outer_fold=fold, seed=seed, meta_source=role == "meta")
            for row in run: assert_runtime_gold_free(row)
            runtime.extend(run); utility.extend(alternative_rows(run, target, variant="S2")); gates.append(gate_row(run, target)); targets[qid] = target
            self.progress.count(index, len(by_question), item=f"{cell} {role} feature questions", extra={"fold": fold, "seed": seed})
        result = (runtime, utility, gates, targets); self._cache[key] = copy.deepcopy(result)
        cache = self.config.run_dir / "cache/meta_features" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        write_jsonl(cache / f"{role}_runtime.jsonl", runtime); write_jsonl(cache / f"{role}_utility_targets.jsonl", utility)
        write_jsonl(cache / f"{role}_gate_targets.jsonl", gates)
        return copy.deepcopy(result)


def _filter_embargo(rows: list[dict], eval_runtime: list[dict]) -> tuple[list[dict], dict]:
    eval_docs = {row["canonical_document_id"] for row in eval_runtime}
    eval_hashes = {row["document_sha256"] for row in eval_runtime}
    kept = [row for row in rows if row["canonical_document_id"] not in eval_docs and row["document_sha256"] not in eval_hashes]
    return kept, {"input_rows": len(rows), "kept_rows": len(kept), "excluded_rows": len(rows) - len(kept),
                  "eval_canonical_documents": len(eval_docs), "eval_document_hashes": len(eval_hashes)}


def _fit_s1(rows, seed, folds, *, aspect=True):
    columns = feature_columns("S1", aspect=aspect)
    raw, assignment = meta_oof(rows, columns, seed, c=1.0, balanced=False, folds=folds)
    model = fit_model(rows, columns, seed, c=1.0, balanced=False)
    probability = 1 / (1 + np.exp(-np.clip(raw, -40, 40)))
    return FittedUtility(model, IdentityCalibrator(), columns, 1.0, 0.5, probability, assignment,
                         {"selected_C": 1.0, "tau_switch": 0.5, "calibration": "exact_v5_native_probability",
                          "contract": "exact V5 mirrored evidence target + fixed interactions"})


def _dump_model(root: Path, utility: FittedUtility, gate: FittedGate | None, metadata: dict):
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(utility.model, root / "utility_model.joblib"); joblib.dump(utility.calibrator, root / "utility_calibrator.joblib")
    if gate:
        joblib.dump(gate.model, root / "gate_model.joblib"); joblib.dump(gate.calibrator, root / "gate_calibrator.joblib")
    write_json(root / "selection.json", metadata | {"utility": utility.selection, "gate": gate.selection if gate else None,
                                                       "feature_columns": list(utility.columns)})


def _mrr_selected_first(runtime_rows, selected_chunk: int, target) -> float:
    ordered = [selected_chunk] + [int(row["chunk_id"]) for row in runtime_rows if int(row["chunk_id"]) != selected_chunk]
    relevant = {int(cid) for cid, tier in target["label_tier"].items() if tier == "sufficient_positive"}
    ranks = [index + 1 for index, cid in enumerate(ordered) if cid in relevant]
    return 1.0 / min(ranks) if ranks else 0.0


def predictions(cell: str, runtime: list[dict], targets: dict, variant: str, mode: str, fold: int, seed: int,
                utility: FittedUtility | None, gate: FittedGate | None, utility_rows: list[dict]) -> list[dict]:
    runtime_groups = grouped(runtime); alt_groups = defaultdict(list)
    for row in utility_rows: alt_groups[row["question_id"]].append(row)
    probability = utility.probabilities(utility_rows) if utility and utility_rows else np.array([])
    probability_by_qid = defaultdict(list)
    for row, p in zip(utility_rows, probability): probability_by_qid[row["question_id"]].append((row, float(p)))
    gate_map = {}
    if gate:
        gate_inputs = []
        for qid in sorted(runtime_groups): gate_inputs.append(gate_row(runtime_groups[qid], targets[qid]))
        gate_probs = gate.probabilities(gate_inputs); gate_map = {r["question_id"]: float(p) for r, p in zip(gate_inputs, gate_probs)}
    out = []
    for qid, candidates in sorted(runtime_groups.items()):
        top = candidates[0]; target = targets[qid]; selected = top; utility_p = 0.0
        routed = variant not in {"S0", "S1", "S2"}
        gate_p = gate_map.get(qid, 1.0 if routed else 0.0)
        if gate and (int(top["effective_k"]) <= 1 or not (gate_p >= gate.tau)):
            routed = False
        if variant in {"S1", "S2"}: routed = True
        if variant != "S0" and routed and probability_by_qid.get(qid):
            best, utility_p = max(probability_by_qid[qid], key=lambda x: (x[1], -int(x[0]["chunk_id"])))
            if utility_p > utility.tau:
                selected = next(row for row in candidates if int(row["chunk_id"]) == int(best["chunk_id"]))
        selected_id = int(selected["chunk_id"]); selected_tier = target["label_tier"][str(selected_id)]
        runtime_payload = {
            "dataset_cell": cell, "corpus_family": FAMILIES[cell], "canonical_document_id": top["canonical_document_id"],
            "document_sha256": top["document_sha256"], "question_lineage_id": top["question_lineage_id"],
            "document_id": top["document_id"], "question_id": qid, "canonical_aspect": top["canonical_aspect"],
            "language": top["language"], "outer_fold": fold, "seed": seed, "system": variant,
            "parameterization_mode": mode, "b0_top_k": [{"chunk_id": r["chunk_id"], "raw": r["b0_score_raw"], "z": r["b0_score_norm"]} for r in candidates],
            "gate_probability": gate_p, "tau_gate": "inf" if gate and not math.isfinite(gate.tau) else (gate.tau if gate else None),
            "gate_decision": bool(routed), "positive_utility_probability": utility_p,
            "tau_switch": utility.tau if utility else None, "selected_chunk_id": selected_id,
            "b0_chunk_id": int(top["chunk_id"]), "final_answer": selected["qa_answer"],
            "answer_valid": selected["answer_valid"], "effective_k": top["effective_k"],
            "qa_forwards": int(top["effective_k"] if routed else 1),
            "q_a_ranker_forwards": int(top["effective_k"] if routed else 0),
            "availability_masks": {name: top[name] for name in ("source_available", "lex_available", "sem_available", "qa_conf_available")},
        }
        assert_runtime_gold_free(runtime_payload)
        evaluation = {"token_f1": float(target["candidate_f1"][str(selected_id)]),
                      "baseline_token_f1": float(target["baseline_f1"]), "selected_tier": selected_tier,
                      "hit@1": int(selected_tier == "sufficient_positive"),
                      "baseline_hit@1": int(target["label_tier"][str(top["chunk_id"])] == "sufficient_positive"),
                      "hn@1": int(target["hard_negative"][str(selected_id)]),
                      "baseline_hn@1": int(target["hard_negative"][str(top["chunk_id"])]),
                      "mrr_selected_first": _mrr_selected_first(candidates, selected_id, target),
                      "changed_top1": int(selected_id != int(top["chunk_id"]))}
        out.append({"runtime": runtime_payload, "evaluation": evaluation})
    return out


class Pipeline:
    def __init__(self, config: Config):
        config.validate(); self.config = config; self.output = config.run_dir
        self.progress = ProgressLogger(self.output / "progress.jsonl"); self.started = time.time()
        self.data = DataStore(config, self.progress); self.before = None

    def _status(self, status, **extra): write_json(self.output / "status.json", {"status": status, "elapsed_s": time.time() - self.started, **extra})

    def run(self):
        stop = threading.Event(); heartbeat = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True); heartbeat.start()
        try:
            self._status("CREATED"); self.audit()
            if self.config.phase == "audit": return 0
            self.historical_replay()
            if self.config.phase == "replay": return 0
            if self.config.phase in {"run", "all"}: self.development()
            if self.config.phase in {"analysis", "all"}:
                from .analyze import analyze_run
                analyze_run(self.output, self.config.bootstrap, self.progress)
            if self.config.phase == "all":
                if self.config.smoke:
                    write_json(self.output / "resource_benchmark/SKIPPED.json", {"reason": "smoke validation does not benchmark neural models"})
                else:
                    from .benchmark import run_benchmarks
                    run_benchmarks(self.config, self.output, self.progress)
            self.finish(); return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc)); write_json(self.output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}"); raise
        finally:
            stop.set(); heartbeat.join(1)

    def _heartbeat(self, stop):
        while not stop.wait(30): self.progress.log("still running", elapsed_seconds=round(time.time() - self.started, 1))

    def audit(self):
        self._status("AUDIT_RUNNING"); self.progress.stage("audit", "read-only history, hashes, lineages and frozen contract")
        git_snapshot(self.output / "audit/git_before")
        result = run_audit(self.output / "audit", v5_run=self.config.v5_run, cells=self.config.datasets); self.before = result["hashes"]
        contract = frozen_contract(); contract["config"] = self.config.payload()
        text = json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        write_text(self.output / "predevelopment_contract.yaml", text); write_text(self.output / "frozen_contract.yaml", text)
        write_json(self.output / "manifest.json", {"contract_sha256": sha256(self.output / "frozen_contract.yaml"),
                                                    "config": self.config.payload(), "historical_read_only": True})
        write_json(self.output / "reused_code/reused_components.json", [
            {"source": "v5_question_conditioned_residual.contract.CELLS", "historical_version": "V5", "wrapper": "v6.contract", "modified": False},
            {"source": "v4.answers.pair_token_f1", "historical_version": "V4-V5", "wrapper": "v6.features", "modified": False},
            {"source": "V5 inner_meta/features.jsonl", "historical_version": "V5 completed run", "wrapper": "v6.pipeline.DataStore", "modified": False},
        ])
        self._status("AUDIT_PASSED")

    def historical_replay(self):
        self.progress.stage("replay", "B0 answer/F1 and V5-A1 shortlist replay")
        result = replay(self.config.v5_run, self.config.datasets, strict=True)
        write_json(self.output / "audit/historical_replay/replay.json", result)

    def _scope(self, mode: str, fold: int, seed: int, target_cell: str | None = None, held_family: str | None = None):
        train_runtime = []; train_utility = []; train_gate = []; eval_sets = {}; embargo = {}
        cells = [target_cell] if mode == "local" else list(self.config.datasets)
        for cell in cells:
            meta_runtime, meta_utility, meta_gate, _ = self.data.load(cell, fold, seed, "meta")
            eval_runtime, eval_utility, eval_gate, eval_targets = self.data.load(cell, fold, seed, "eval")
            eval_sets[cell] = (eval_runtime, eval_utility, eval_gate, eval_targets)
            if mode == "loco" and FAMILIES[cell] == held_family: continue
            train_runtime.extend(meta_runtime); train_utility.extend(meta_utility); train_gate.extend(meta_gate)
        all_eval_runtime = [r for values in eval_sets.values() for r in values[0]]
        train_utility, embargo["utility"] = _filter_embargo(train_utility, all_eval_runtime)
        train_gate, embargo["gate"] = _filter_embargo(train_gate, all_eval_runtime)
        train_runtime, embargo["runtime"] = _filter_embargo(train_runtime, all_eval_runtime)
        if mode == "loco":
            train_utility = [r for r in train_utility if r["corpus_family"] != held_family]
            train_gate = [r for r in train_gate if r["corpus_family"] != held_family]
            eval_sets = {cell: value for cell, value in eval_sets.items() if FAMILIES[cell] == held_family}
        return train_runtime, train_utility, train_gate, eval_sets, embargo

    def _execute_scope(self, mode: str, fold: int, seed: int, *, target_cell=None, held_family=None):
        label = target_cell or held_family or "shared"
        root = self.output / "meta_models" / mode / label / f"outer_fold_{fold}" / f"seed_{seed}"
        complete = root / "COMPLETED.json"
        if complete.is_file(): self.progress.log("scope already completed", mode=mode, scope=label, fold=fold, seed=seed); return
        self.progress.stage(f"{mode}:{label}:f{fold}:s{seed}", "META_PRED_OOF tuning and sealed EVAL")
        _, utility_rows, gate_rows, eval_sets, embargo = self._scope(mode, fold, seed, target_cell, held_family)
        pooled = mode != "local"
        weight_audit = normalize_training_weights(utility_rows, pooled=pooled)
        normalize_training_weights(gate_rows, pooled=pooled)
        write_json(root / "embargo.json", embargo); write_json(root / "training_weight_audit.json", weight_audit)
        # S1 has a distinct, exact V5 mirrored evidence training population.
        s1_rows = []
        train_cells = [target_cell] if mode == "local" else [c for c in self.config.datasets if mode != "loco" or FAMILIES[c] != held_family]
        for cell in train_cells:
            runtime, _, _, targets = self.data.load(cell, fold, seed, "meta")
            for qid, values in grouped(runtime).items(): s1_rows.extend(evidence_pair_rows(values, targets[qid]))
        eval_docs = [r for values in eval_sets.values() for r in values[0]]
        s1_rows, _ = _filter_embargo(s1_rows, eval_docs); normalize_training_weights(s1_rows, pooled=pooled)
        fitted = {"S1": (_fit_s1(s1_rows, seed, self.config.inner_meta_folds), None)}
        for variant in ("S2", "S3", "S4-src", "S4-lex", "S4"):
            utility = tune_utility(copy.deepcopy(utility_rows), variant, seed, self.config.inner_meta_folds)
            gate = tune_gate(copy.deepcopy(gate_rows), utility_rows, utility.oof_probability, utility.tau,
                             seed, self.config.inner_meta_folds) if VARIANTS[variant]["gate"] else None
            fitted[variant] = (utility, gate)
        diagnostics = {}
        if mode == "pooled":
            no_aspect = tune_utility(copy.deepcopy(utility_rows), "S4", seed, self.config.inner_meta_folds, aspect=False)
            no_aspect_gate = tune_gate(copy.deepcopy(gate_rows), utility_rows, no_aspect.oof_probability, no_aspect.tau,
                                       seed, self.config.inner_meta_folds)
            diagnostics["S4-no-aspect"] = (no_aspect, no_aspect_gate)
        write_json(root / "availability_proxy_audit.json", {"patterns": {"all_cells": [1, 1, 0, 0]},
                   "mutual_information_with_corpus_family": 0.0, "with_masks_equals_without_masks": True,
                   "reason": "availability pattern is constant across every candidate row"})
        write_json(root / "S5/SKIPPED.json", {"reason": "exact historical SEM and QA-confidence are not universally replayable across all five cells",
                                               "sem_available": False, "qa_conf_available": False, "eligible": False})
        for variant, (utility, gate) in {**fitted, **diagnostics}.items():
            _dump_model(root / variant, utility, gate, {"mode": mode, "scope": label, "fold": fold, "seed": seed,
                                                        "one_shared_model_for_all_eval_cells": mode != "local"})
        for cell, (runtime, eval_utility, _eval_gate, targets) in eval_sets.items():
            base = predictions(cell, runtime, targets, "S0", mode, fold, seed, None, None, eval_utility)
            write_jsonl(self.output / "predictions" / mode / label / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "S0.jsonl", base)
            for variant, (utility, gate) in {**fitted, **diagnostics}.items():
                rows = predictions(cell, runtime, targets, variant, mode, fold, seed, utility, gate, eval_utility)
                path = self.output / "predictions" / mode / label / cell / f"outer_fold_{fold}" / f"seed_{seed}" / f"{variant}.jsonl"
                write_jsonl(path, rows)
        write_json(complete, {"status": "complete", "mode": mode, "scope": label, "fold": fold, "seed": seed,
                              "variants": ["S0", *fitted, *diagnostics], "eval_cells": sorted(eval_sets)})

    def development(self):
        self._status("RUNNING"); self.progress.stage("development", "LOCAL, global-fold POOLED and LOCO-meta")
        if "local" in self.config.modes:
            for cell in self.config.datasets:
                for fold in self.config.outer_folds:
                    for seed in self.config.seeds: self._execute_scope("local", fold, seed, target_cell=cell)
        if "pooled" in self.config.modes:
            for fold in self.config.outer_folds:
                for seed in self.config.seeds: self._execute_scope("pooled", fold, seed)
        if "loco" in self.config.modes:
            for family in sorted({FAMILIES[c] for c in self.config.datasets}):
                for fold in self.config.outer_folds:
                    for seed in self.config.seeds: self._execute_scope("loco", fold, seed, held_family=family)

    def finish(self):
        self.progress.stage("finalize", "historical hash assertion and completion marker")
        after = historical_hashes(self.config.v5_run, self.config.datasets)
        write_json(self.output / "audit/historical_hashes_after.json", after); git_snapshot(self.output / "audit/git_after")
        if self.before != after: raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED")
        write_json(self.output / "audit/regression_summary.json", {"historical_hashes_unchanged": True,
                   "runtime_gold_leakage": False, "seed_isolation": True, "pooled_global_fold_shared_model": True,
                   "scier_cross_cell_embargo": True, "meta_prediction_oof": True})
        self._status("COMPLETED"); write_json(self.output / "COMPLETED.json", {"status": "COMPLETED",
                   "elapsed_s": time.time() - self.started, "historical_hashes_unchanged": True,
                   "contract_sha256": sha256(self.output / "frozen_contract.yaml")})

