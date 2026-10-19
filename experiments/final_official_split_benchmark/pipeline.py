"""Phase-separated final benchmark orchestration with a one-way TEST firewall."""
from __future__ import annotations

import copy
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection_v8.models import apply_policy, fold_for

from .analyze import evaluate, report, statistics
from .audit import assert_history_unchanged, model_provenance, run_audit
from .config import Config
from .contract import CELL_SPECS, MODEL_REGISTRY, REAL_SYSTEMS, fairness_matrix, frozen_contract, system_registry
from .data import development_splits, final_training_pool, load_test_runtime
from .engine import CellEngine, release_cuda
from .io_guard import canonical_hash, read_json, sha256, write_json, write_jsonl, write_text
from .modernbert import evaluate_model, train_model
from .proposed import apply_labeled, apply_runtime, fit_final_family, fit_policy, save_policy
from .resources import benchmark
from .systems import run_blind_cell


class Pipeline:
    def __init__(self, config: Config):
        config.validate(); self.config = config; self.output = config.run_dir
        self.progress = ProgressLogger(self.output / "progress.jsonl"); self.started = time.time()

    def _status(self, status, **extra):
        write_json(self.output / "status.json", {"status": status, "elapsed_s": time.time() - self.started, **extra})

    def _heartbeat(self, stop):
        while not stop.wait(30): self.progress.log("still running", elapsed_seconds=round(time.time() - self.started, 1), phase=self.config.phase)

    def _assert_resume_fingerprint(self):
        path = self.output / "manifest.json"
        if not self.config.resume or not path.is_file(): return
        old = read_json(path)["config"]; current = self.config.payload(); stable = ("cells", "seeds", "inner_folds", "smoke", "allow_downloads")
        mismatch = {key: (old.get(key), current.get(key)) for key in stable if old.get(key) != current.get(key)}
        if mismatch: raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")

    def run(self):
        self._assert_resume_fingerprint()
        stop = threading.Event(); thread = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True); thread.start()
        try:
            if self.config.phase in {"audit", "all", "smoke"}:
                if not (self.config.resume and self.config.phase == "all" and (self.output / "audit/AUDIT_COMPLETE.json").is_file()): self.audit()
                else: self.progress.log("audit already complete; preserving frozen pre-TEST provenance")
            if self.config.phase in {"develop", "all", "smoke"}:
                if not (self.config.resume and self.config.phase == "all" and (self.output / "development/COMPLETE.json").is_file()): self.develop()
                else: self.progress.log("development already complete")
            if self.config.phase in {"freeze", "all", "smoke"}:
                if not (self.config.resume and self.config.phase == "all" and (self.output / "frozen/freeze_contract.json").is_file()): self.freeze()
                else: self.progress.log("freeze already complete")
            if self.config.phase in {"refit", "all", "smoke"}:
                if not (self.config.resume and self.config.phase == "all" and (self.output / "refit/COMPLETE.json").is_file()): self.refit()
                else: self.progress.log("refit already complete")
            if self.config.phase == "smoke":
                write_json(self.output / "SMOKE_COMPLETE.json", {"status": "PASS", "official_test_accessed": False}); self._status("SMOKE_COMPLETE"); return 0
            if self.config.phase == "all" and not self.config.unlock_test:
                self._status("READY_FOR_TEST", instruction="resume with --phase all --unlock-test"); return 0
            if self.config.phase in {"predict-test", "all"}:
                if not (self.config.resume and (self.output / "test_predictions/ALL_REAL_SYSTEMS_COMPLETE.json").is_file()): self.predict_test()
                else: self.progress.log("blind TEST predictions already complete; immutable artifacts preserved")
            if self.config.phase in {"evaluate", "all"}:
                if not (self.config.resume and (self.output / "test_analysis/COMPLETE.json").is_file()): self.evaluate_test()
                else: self.progress.log("TEST evaluation already complete; gold not reopened")
            if self.config.phase in {"resources", "all"}:
                if not (self.config.resume and (self.output / "resources/summary.json").is_file()): self.resources()
                else: self.progress.log("resource benchmark already complete")
            if self.config.phase in {"report", "all"}:
                if not (self.config.resume and (self.output / "COMPLETED.json").is_file()): self.report()
                else: self.progress.log("final report already complete")
            self._status("COMPLETED" if self.config.phase == "all" else f"PHASE_{self.config.phase.upper()}_COMPLETE")
            return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc)); write_json(self.output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}"); raise
        finally:
            stop.set(); thread.join(1)

    def _require(self, relative):
        path = self.output / relative
        if not path.exists(): raise RuntimeError(f"required prior phase artifact missing: {path}")
        return path

    def audit(self):
        self._status("AUDIT_RUNNING"); contract = frozen_contract(); contract_path = self.output / "frozen_contract.json"
        write_json(contract_path, contract, immutable=self.config.resume)
        manifest = {"config": self.config.payload(), "contract_sha256": sha256(contract_path), "created_before_test": True,
                    "historical_namespaces_read_only": True, "experiment_status": "FINAL_BENCHMARK"}
        path = self.output / "manifest.json"
        if path.exists() and self.config.resume:
            old = read_json(path); stable = ("cells", "seeds", "inner_folds", "smoke", "allow_downloads")
            mismatch = {k: (old["config"].get(k), manifest["config"].get(k)) for k in stable if old["config"].get(k) != manifest["config"].get(k)}
            if mismatch or old["contract_sha256"] != manifest["contract_sha256"]: raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")
        else: write_json(path, manifest, immutable=True)
        write_json(self.output / "manifests/system_registry.json", system_registry(), immutable=True)
        write_json(self.output / "audit/system_registry.json", system_registry(), immutable=True)
        write_json(self.output / "system_registry.json", system_registry(), immutable=True)
        write_json(self.output / "audit/fairness_matrix.json", fairness_matrix(), immutable=True)
        run_audit(self.config, self.progress); self._status("AUDIT_COMPLETE")

    @staticmethod
    def _calibration(records, held_fold):
        candidates = [x for x in records if fold_for(x.document_id, 1042, 5) == (held_fold % 5)]
        if not candidates: candidates = list(records)[: max(1, len(records) // 10)]
        docs = {x.document_id for x in candidates}; train = [x for x in records if x.document_id not in docs]
        return (train or list(records)), candidates

    def _oof_features(self, cell, records, engine, root, *, tag):
        folds = min(self.config.inner_folds, len({x.document_id for x in records})); folds = max(2, folds)
        assignment = {x.document_id: fold_for(x.document_id, 42, folds) for x in records}
        missing = set(range(folds)) - set(assignment.values())
        if missing: raise RuntimeError(f"{cell}: empty B0 OOF folds {sorted(missing)}")
        runtime, targets, rows, reports = [], {}, [], []
        for held in range(folds):
            held_rows = [x for x in records if assignment[x.document_id] == held]; remain = [x for x in records if assignment[x.document_id] != held]
            train, calibration = self._calibration(remain, held); work = root / f"b0_oof/fold_{held}"
            scorer, fit = engine.train_b0(train, calibration, 42, work)
            run, target, alternatives = engine.materialize(held_rows, scorer, 42, role=f"{tag}_OOF_{held}", with_targets=True)
            runtime.extend(run); targets.update(target); rows.extend(alternatives); reports.append({"fold": held, "train": len(train), "calibration": len(calibration), "held": len(held_rows), "fit": fit})
            engine.adapter.release(scorer, work, delete_model=True)
        write_json(root / "b0_oof_manifest.json", {"folds": folds, "assignment": assignment, "reports": reports})
        write_jsonl(root / "oof_runtime.jsonl", runtime); write_jsonl(root / "oof_h2_rows.jsonl", rows)
        return runtime, targets, rows

    @staticmethod
    def _clone_seed(rows, seed):
        values = copy.deepcopy(rows)
        for row in values: row["seed"] = int(seed)
        return values

    def _evaluate_policy(self, rows, policy, use_harm=True):
        choices, _ = apply_labeled(rows, policy, use_harm=use_harm)
        mean = lambda key: float(np.mean([x[key] for x in choices])) if choices else 0.0
        return {"questions": len(choices), "token_f1": mean("f1"), "baseline_token_f1": mean("base_f1"),
                "delta_token_f1": mean("f1") - mean("base_f1"), "hit@1": mean("hit1"), "hn@1": mean("hn1"), "switch_rate": mean("switch")}

    def _modernbert_development(self, cell, train, val, engine, scorer):
        if self.config.smoke or CELL_SPECS[cell].language != "en": return {"status": "NOT_RUN"}
        capabilities = read_json(self.output / "manifests/system_capability_matrix.json")
        status = next(x["status"] for x in capabilities["rows"] if x["cell"] == cell and x["system"] == "B5-EN")
        if status != "RUN": return {"status": status}
        provenance = read_json(self.output / "manifests/model_provenance.json")["MODERNBERT_EN"]
        trials = []
        for lr in (2e-5, 3e-5):
            for epochs in (2, 3):
                root = self.output / "development" / cell / "B5-EN" / f"lr_{lr}_epochs_{epochs}"
                train_model(train, engine, scorer, checkpoint=MODEL_REGISTRY["MODERNBERT_EN"]["model_id"], output=root, seed=42,
                            learning_rate=lr, epochs=epochs, device=engine.device, allow_downloads=self.config.allow_downloads, progress=self.progress, revision=provenance.get("resolved_revision"))
                metric = evaluate_model(val, engine, scorer, root); trials.append({"learning_rate": lr, "epochs": epochs, **metric, "path": str(root)})
        selected = max(trials, key=lambda x: (x["token_f1"], -x["epochs"], -x["learning_rate"]))
        return {"status": "SELECTED_ON_VAL", "selected": selected, "trials": trials}

    def develop(self):
        self._require("audit/AUDIT_COMPLETE.json"); self._status("DEVELOPMENT_RUNNING")
        for cell in self.config.cells:
            done = self.output / "development" / cell / "COMPLETE.json"
            if done.exists() and self.config.resume: self.progress.log("development cell already complete", cell=cell); continue
            self.progress.stage(f"develop:{cell}", "TRAIN-only B0 OOF, LOCAL H2-family tuning, VAL selection")
            train, val = development_splits(cell, smoke=self.config.smoke); engine = CellEngine(self.config, cell, self.progress)
            question_mapping = engine.select_questions(train); train = engine.apply_questions(train); val = engine.apply_questions(val)
            _, _, train_rows = self._oof_features(cell, train, engine, self.output / "development" / cell, tag="TRAIN")
            rows_by_seed = {seed: self._clone_seed(train_rows, seed) for seed in self.config.seeds}
            family = fit_final_family(rows_by_seed, self.config.seeds, self.config.inner_folds, self.output / "development" / cell / "proposed", self.progress)
            b0_work = self.output / "development" / cell / "B0_train_val_selection"; scorer, b0_report = engine.train_b0(train, val, 42, b0_work)
            val_runtime, val_targets, val_rows = engine.materialize(val, scorer, 42, role="VAL", with_targets=True)
            write_jsonl(self.output / "development" / cell / "val_runtime.jsonl", val_runtime); write_jsonl(self.output / "development" / cell / "val_h2_rows.jsonl", val_rows)
            metrics = {}
            for seed in self.config.seeds:
                seeded = self._clone_seed(val_rows, seed)
                for system in ("P1", "P2", "P3", "P4", "P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK"):
                    from .proposed import load_policy
                    policy = load_policy(self.output / "development" / cell / "proposed" / system / f"seed_{seed}")
                    metrics.setdefault(system, []).append(self._evaluate_policy(seeded, policy, use_harm=system != "P2-NO-RISK"))
            aggregated = {system: {k: float(np.mean([x[k] for x in values])) for k in values[0]} for system, values in metrics.items()}
            modern = self._modernbert_development(cell, train, val, engine, scorer)
            epoch = int(b0_report.get("best_epoch") or 1)
            training_report = b0_work / "model/training_report.json"
            if training_report.is_file(): epoch = int(read_json(training_report).get("best_epoch") or epoch)
            write_json(self.output / "development" / cell / "validation_metrics.json", aggregated)
            write_json(self.output / "validation/metrics" / f"{cell}.json", aggregated)
            write_json(self.output / "train/inner_cv" / f"{cell}.json", {"oof_runtime": str((self.output / "development" / cell / "oof_runtime.jsonl").resolve()), "oof_h2_rows": str((self.output / "development" / cell / "oof_h2_rows.jsonl").resolve()), "models": str((self.output / "development" / cell / "proposed").resolve())})
            write_json(self.output / "development" / cell / "frozen_selection.json", {"b0_epochs": epoch, "B5-EN": modern,
                "question_mapping": question_mapping, "question_selection_source": "TRAIN-only canonical historical Q5 policy",
                "Qwen": {"prompt_choice": "single predeclared language template", "candidate_count": 1, "decoding": "deterministic"},
                "P3_selected_config": family["P3_selected_config"], "selection_source": "TRAIN OOF plus official/deterministic VAL; TEST unavailable"})
            write_json(self.output / "validation/selections" / f"{cell}.json", read_json(self.output / "development" / cell / "frozen_selection.json"))
            write_json(done, {"status": "COMPLETE", "train_questions": len(train), "val_questions": len(val)})
            engine.adapter.release(scorer, b0_work, delete_model=True); release_cuda(engine)
        write_json(self.output / "development/COMPLETE.json", {"cells": list(self.config.cells)}); self._status("DEVELOPMENT_COMPLETE")

    def freeze(self):
        self._require("development/COMPLETE.json"); self.progress.stage("freeze", "freeze all systems, selections, prompts, models, and TEST unlock contract")
        selections = {cell: read_json(self.output / "development" / cell / "frozen_selection.json") for cell in self.config.cells}
        payload = {"contract_sha256": sha256(self.output / "frozen_contract.json"), "system_registry_sha256": sha256(self.output / "manifests/system_registry.json"),
                   "selections": selections, "selection_sha256": canonical_hash(selections),
                   "model_provenance_sha256": sha256(self.output / "manifests/model_provenance.json"), "test_metrics_seen": False, "status": "FROZEN_BEFORE_TEST"}
        write_json(self.output / "frozen/freeze_contract.json", payload, immutable=True)
        write_text(self.output / "frozen_contract.yaml", json.dumps({"scientific_contract": frozen_contract(), "freeze": payload}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", immutable=True)
        self._status("FROZEN")

    def refit(self):
        self._require("frozen/freeze_contract.json"); self._status("REFIT_RUNNING")
        for cell in self.config.cells:
            done = self.output / "refit" / f"{cell}_COMPLETE.json"
            if done.exists() and self.config.resume: continue
            self.progress.stage(f"refit:{cell}", "TRAIN+VAL OOF meta fit and full-pool neural refit")
            refit_started = time.time()
            pool = final_training_pool(cell, smoke=self.config.smoke); engine = CellEngine(self.config, cell, self.progress)
            frozen_selection = read_json(self.output / "development" / cell / "frozen_selection.json")
            engine.select_questions(pool, frozen=frozen_selection["question_mapping"]); pool = engine.apply_questions(pool)
            _, _, rows = self._oof_features(cell, pool, engine, self.output / "refit" / cell, tag="TRAIN_PLUS_VAL")
            rows_by_seed = {seed: self._clone_seed(rows, seed) for seed in self.config.seeds}
            family = fit_final_family(rows_by_seed, self.config.seeds, self.config.inner_folds, self.output / "models" / cell / "proposed", self.progress)
            selection = frozen_selection; epochs = int(selection["b0_epochs"])
            scorer = engine.refit_b0(pool, 42, epochs, self.output / "models" / cell / "B0_final")
            modern = selection["B5-EN"]
            if modern.get("status") == "SELECTED_ON_VAL":
                chosen = modern["selected"]
                for seed in self.config.seeds:
                    train_model(pool, engine, scorer, checkpoint=MODEL_REGISTRY["MODERNBERT_EN"]["model_id"], output=self.output / "models" / cell / "B5-EN" / f"seed_{seed}",
                                seed=seed, learning_rate=float(chosen["learning_rate"]), epochs=int(chosen["epochs"]), device=engine.device,
                                allow_downloads=self.config.allow_downloads, progress=self.progress,
                                revision=read_json(self.output / "manifests/model_provenance.json")["MODERNBERT_EN"].get("resolved_revision"))
            pool_hash = canonical_hash(sorted((x.document_id, x.question_id, x.question) for x in pool))
            cell_provenance = {"cell": cell, "training_split_hash": pool_hash, "frozen_contract_sha256": sha256(self.output / "frozen_contract.json"),
                "model_seeds": list(self.config.seeds), "b0_seed": 42, "b0_fixed_epochs": epochs,
                "P3_selected_config": family["P3_selected_config"], "B5-EN_selection": modern,
                "models_root": str((self.output / "models" / cell).resolve())}
            write_json(self.output / "refit_train_val/models" / f"{cell}_provenance.json", cell_provenance)
            write_json(done, {"status": "COMPLETE", "pool_questions": len(pool), "pool_documents": len({x.document_id for x in pool}),
                "training_split_hash": pool_hash, "b0_fixed_epochs": epochs, "P3_selected_config": family["P3_selected_config"],
                "training_wall_seconds": time.time() - refit_started, "training_gpu_hours_upper_bound_one_gpu": (time.time() - refit_started) / 3600.0})
            release_cuda(scorer, engine)
        model_files = sorted(path for path in (self.output / "models").rglob("*") if path.is_file())
        model_hashes = {str(path.relative_to(self.output)): sha256(path) for path in model_files}
        write_json(self.output / "refit_train_val/model_hashes.json", model_hashes)
        write_json(self.output / "refit_train_val/models/index.json", {"root": str((self.output / "models").resolve()), "files": len(model_hashes)})
        provenance = model_provenance(self.config); write_json(self.output / "manifests/model_provenance_after_refit.json", provenance)
        write_json(self.output / "refit/COMPLETE.json", {"cells": list(self.config.cells)}); self._status("READY_FOR_TEST")

    def _unlock_assertions(self):
        if not self.config.unlock_test: raise RuntimeError("TEST_FIREWALL: pass --unlock-test after reviewing frozen/refit artifacts")
        if self.config.smoke: raise RuntimeError("TEST_FIREWALL: smoke cannot unlock TEST")
        freeze = read_json(self._require("frozen/freeze_contract.json")); self._require("refit/COMPLETE.json")
        if freeze["contract_sha256"] != sha256(self.output / "frozen_contract.json"): raise RuntimeError("frozen contract changed")
        if freeze["model_provenance_sha256"] != sha256(self.output / "manifests/model_provenance.json"): raise RuntimeError("frozen model provenance changed")
        capabilities = read_json(self.output / "manifests/system_capability_matrix.json")
        unresolved = [x for x in capabilities["rows"] if x["status"] not in {"RUN", "BLOCKED_MISSING_MODEL", "NOT_APPLICABLE_LANGUAGE", "NOT_RUN_PREDECLARED"}]
        if unresolved: raise RuntimeError(f"unresolved system configs: {unresolved}")
        write_json(self.output / "test_unlock_assertion.json", {"status": "PASS", "frozen_hash": sha256(self.output / "frozen_contract.json"),
            "refit_complete": True, "all_system_statuses_resolved": True, "test_metrics_seen": False}, immutable=True)

    def predict_test(self):
        self._unlock_assertions(); self._status("BLIND_TEST_PREDICTION_RUNNING")
        for cell in self.config.cells:
            marker = self.output / "test_predictions" / f"{cell}_COMPLETE.json"
            if marker.exists() and self.config.resume: continue
            self.progress.stage(f"blind-test:{cell}", "generate all real-system predictions without loading TEST gold")
            items = load_test_runtime(cell); engine = CellEngine(self.config, cell, self.progress)
            selection = read_json(self.output / "development" / cell / "frozen_selection.json")
            engine.select_questions((), frozen=selection["question_mapping"]); items = engine.apply_questions(items)
            scorer = engine.load_b0(self.output / "models" / cell / "B0_final")
            runtime, target, rows = engine.materialize(items, scorer, 42, role="BLIND_TEST", with_targets=False)
            if target or rows: raise AssertionError("blind TEST feature materialization produced gold targets")
            write_jsonl(self.output / "test_predictions/_runtime_features" / f"{cell}.jsonl", runtime, immutable=True)
            run_blind_cell(self.config, cell, items, engine, scorer, runtime, self.progress); release_cuda(scorer, engine)
        write_json(self.output / "test_predictions/ALL_REAL_SYSTEMS_COMPLETE.json", {"cells": list(self.config.cells)}, immutable=True)
        self._status("BLIND_TEST_PREDICTIONS_COMPLETE")

    def evaluate_test(self):
        self._require("test_predictions/ALL_REAL_SYSTEMS_COMPLETE.json"); self._status("TEST_EVALUATION_RUNNING")
        evaluated = evaluate(self.config, self.progress); stats = statistics(self.config, evaluated)
        write_json(self.output / "test_analysis/COMPLETE.json", {"status": "COMPLETE", "cells": list(self.config.cells)}, immutable=True)
        self._status("TEST_EVALUATION_COMPLETE")

    def resources(self):
        self._require("test_predictions/ALL_REAL_SYSTEMS_COMPLETE.json"); self._status("RESOURCE_BENCHMARK_RUNNING")
        benchmark(self.config, self.progress); self._status("RESOURCE_BENCHMARK_COMPLETE")

    def report(self):
        self._require("test_analysis/COMPLETE.json"); evaluated = {"summaries": read_json(self.output / "test_analysis/primary_metrics.json"), "question_metrics": read_json(self.output / "test_analysis/question_metrics.json")}
        stats = read_json(self.output / "statistics/statistics.json"); path = report(self.config, evaluated, stats)
        assert_history_unchanged(self.output); write_json(self.output / "COMPLETED.json", {"status": "COMPLETED", "report": str(path), "historical_V1_V10_unchanged": True})
