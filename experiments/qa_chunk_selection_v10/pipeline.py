"""End-to-end V10 orchestration with sealed outer evaluation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.progress import ProgressLogger
from .audit import git_snapshot, historical_hashes, run_audit
from .config import Config
from .contract import ABLATIONS, FAMILIES, H2_COLUMNS, frozen_contract
from .data import DataStore
from .features import assert_feature_schema, assert_runtime_gold_free, inner_manifest, make_questions, runtime_question_view
from .io_guard import read_jsonl, remove_file, sha256, write_joblib, write_json, write_jsonl, write_text
from .models import policy_metrics, predict_all, simulate, train_srr
from .replay import replay


class Pipeline:
    def __init__(self, config: Config):
        config.validate(); self.config = config; self.output = config.run_dir
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time(); self.data = DataStore(config, self.progress); self.before = None

    def _status(self, status, **extra):
        write_json(self.output / "status.json", {"status": status, "elapsed_s": time.time() - self.started, **extra})

    def _heartbeat(self, stop):
        while not stop.wait(30): self.progress.log("still running", elapsed_seconds=round(time.time() - self.started, 1))

    def run(self):
        stop = threading.Event(); heartbeat = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True); heartbeat.start()
        try:
            self._status("CREATED")
            if self.config.phase in {"audit", "replay", "run", "all"}: self.audit()
            if self.config.phase == "audit": self._status("PHASE_AUDIT_COMPLETE"); return 0
            if self.config.phase in {"replay", "run", "all"}: self.historical_replay()
            if self.config.phase == "replay": self._status("PHASE_REPLAY_COMPLETE"); return 0
            if self.config.phase in {"run", "all"}: self.freeze(); self.development()
            if self.config.phase in {"analysis", "all"}:
                from .analyze import analyze_run
                analyze_run(self.output, self.config.bootstrap, self.progress)
            if self.config.phase in {"benchmark", "all"}:
                from .benchmark import benchmark
                benchmark(self.config, self.output, self.progress)
            if self.config.phase == "all": self.finish()
            else: self._status(f"PHASE_{self.config.phase.upper()}_COMPLETE")
            return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc))
            write_json(self.output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set(); heartbeat.join(1)

    def audit(self):
        self._status("AUDIT_RUNNING"); self.progress.stage("audit", "snapshot git, hash read-only V1-V9, freeze feature/evidence semantics")
        git_snapshot(self.output / "audit/git_before")
        self.before = run_audit(self.output / "audit", self.config)
        assert_feature_schema()
        contract_path = self.output / "frozen_contract.yaml"
        write_text(contract_path, json.dumps(frozen_contract(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        manifest = {"config": self.config.payload(), "contract_sha256": sha256(contract_path), "historical_read_only": True, "estimate": "DEVELOPMENT"}
        path = self.output / "manifest.json"
        if path.is_file() and self.config.resume:
            old = json.loads(path.read_text()); stable = ("v5_run", "v8_run", "v9_run", "datasets", "seeds", "outer_folds", "inner_folds", "smoke")
            mismatch = {key: (old["config"].get(key), manifest["config"].get(key)) for key in stable if old["config"].get(key) != manifest["config"].get(key)}
            if mismatch: raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")
            if old["contract_sha256"] != manifest["contract_sha256"]: raise RuntimeError("frozen V10 contract hash mismatch")
        else: write_json(path, manifest)
        self._status("AUDIT_PASSED")

    def historical_replay(self):
        done = self.output / "audit/replay_COMPLETED.json"
        if done.is_file(): self.progress.log("S0/H2 replay already complete"); return
        self._status("REPLAY_RUNNING"); self.progress.stage("replay", "exact V8↔V9 S0/H2 candidate, answer, selection and F1 replay")
        replay(self.config, self.output / "audit", self.progress)
        write_json(done, {"status": "PASS", "systems": ["S0", "S1/H2"]}); self._status("REPLAY_PASSED")

    def freeze(self):
        path = self.output / "frozen_contract.yaml"; expected = json.loads((self.output / "manifest.json").read_text())["contract_sha256"]
        if not path.is_file() or sha256(path) != expected: raise RuntimeError("V10 contract not frozen before outer evaluation")
        write_json(self.output / "audit/freeze.json", {"status": "FROZEN_BEFORE_OUTER_EVAL", "sha256": expected})

    def _historical_prediction(self, system, cell, fold, seed):
        path = self.config.v9_run / "predictions" / system / cell / f"outer_fold_{fold}" / f"seed_{seed}.jsonl"
        rows = copy.deepcopy(read_jsonl(path))
        for row in rows:
            row["runtime"]["system"] = system
            row["runtime"]["historical_exact_replay"] = "V9-S0" if system == "S0" else "V9-exact-H2"
        return rows

    def _save_models(self, root: Path, models, selection):
        values = {"switch": models.switch, "rank": models.rank, "answer_risk": models.answer_risk, "evidence_risk": models.evidence_risk}
        hashes = {}
        for name, model in values.items():
            path = root / f"{name}.joblib"; write_joblib(path, model); hashes[name] = sha256(path)
        write_json(root / "selection.json", selection)
        return hashes

    def _new_predictions(self, cell, fold, seed, questions, module_predictions, selection, hashes, manifest_hash):
        outputs = {system: [] for system in ABLATIONS}
        feature_hash = sha256(self.output / "audit/h2_feature_schema.json")
        for system, (use_answer, use_evidence) in ABLATIONS.items():
            policy = selection["policies"][system]
            simulated = simulate(
                questions, module_predictions, policy["thresholds"], use_answer, use_evidence,
                fold_fallback=policy["fold_policy"] == "B0_FALLBACK",
            )
            for item in simulated:
                question = item["question"]; selected = item["selected"]; base = item["base"]; pred = item["prediction"]
                runtime = runtime_question_view(question)
                runtime.update({
                    "corpus_family": FAMILIES[cell], "canonical_aspect": question["canonical_aspect"],
                    "language": question["language"], "system": system, "model_seed": seed,
                    "p_switch": pred["p_switch"], "pairwise_probabilities": pred["pairs"],
                    "rank_scores": {str(k): v for k, v in pred["rank_scores"].items()}, "rank_order": pred["rank_order"],
                    "p_answer_harm": {str(k): v for k, v in pred["p_answer_harm"].items()},
                    "p_evidence_harm": {str(k): v for k, v in pred["p_evidence_harm"].items()},
                    "thresholds": policy["thresholds"], "fold_policy": policy["fold_policy"],
                    "selected_candidate": selected["chunk_id"], "selected_chunk_id": selected["chunk_id"],
                    "fallback_reason": item["reason"], "final_answer": selected["answer"],
                    "qa_forwards": len(question["candidates"]), "q_a_ranker_forwards": len(question["candidates"]),
                    "semantic_encoder_forwards": 0, "pair_comparisons": len(pred["pairs"]),
                    "provenance": {"inner_split_hash": manifest_hash, "model_hashes": hashes, "feature_schema_hash": feature_hash},
                })
                assert_runtime_gold_free(runtime)
                metrics = qa_row(selected["answer"], question["evaluation"]["gold_answers"])
                base_eval = base["evaluation"]; selected_eval = selected["evaluation"]
                alternatives = question["candidates"][1:]
                best = max(alternatives, key=lambda x: x["evaluation"]["f1"], default=base)
                evaluation = {
                    "gold_answers": question["evaluation"]["gold_answers"], **metrics,
                    "baseline_token_f1": base_eval["f1"], "selected_tier": selected_eval["tier"],
                    "selected_evidence_class": selected_eval["evidence_class"],
                    "hit@1": int(selected_eval["tier"] == "sufficient_positive"),
                    "baseline_hit@1": int(base_eval["tier"] == "sufficient_positive"),
                    "hn@1": selected_eval["hn"], "baseline_hn@1": base_eval["hn"],
                    "hit@k": int(any(x["evaluation"]["tier"] == "sufficient_positive" for x in question["candidates"])),
                    "hn@k": int(any(x["evaluation"]["hn"] for x in question["candidates"])),
                    "selected_safe_evidence": int(selected_eval["evidence_class"] == "SAFE_EVIDENCE"),
                    "selected_unsafe_evidence": int(selected_eval["evidence_class"] == "UNSAFE_EVIDENCE"),
                    "changed_top1": int(selected["chunk_id"] != base["chunk_id"]),
                    "actual_switch_utility": selected_eval["f1"] - base_eval["f1"],
                    "positive_opportunity": question["evaluation"]["positive_opportunity"],
                    "positive_capture": int(selected_eval["f1"] > base_eval["f1"]),
                    "harmful_switch": int(selected["chunk_id"] != base["chunk_id"] and selected_eval["f1"] < base_eval["f1"]),
                    "oracle_candidate": best["chunk_id"], "oracle_f1": best["evaluation"]["f1"],
                    "oracle_rank": best["rank"], "rank_regret": best["evaluation"]["f1"] - (next((x for x in question["candidates"] if x["chunk_id"] == pred["rank_order"][0]), base)["evaluation"]["f1"] if pred["rank_order"] else base_eval["f1"]),
                    "candidate_f1": {str(x["chunk_id"]): x["evaluation"]["f1"] for x in question["candidates"]},
                    "candidate_evidence": {str(x["chunk_id"]): x["evaluation"]["evidence_class"] for x in question["candidates"]},
                    "candidate_hn": {str(x["chunk_id"]): x["evaluation"]["hn"] for x in question["candidates"]},
                }
                outputs[system].append({"runtime": runtime, "evaluation": evaluation})
        return outputs

    def _execute(self, cell, fold, seed):
        state = self.output / "state" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        if (state / "COMPLETED.json").is_file(): self.progress.log("outer unit already complete", cell=cell, fold=fold, seed=seed); return
        self.progress.stage(f"{cell}:f{fold}:s{seed}", "load exact H2 features, inner OOF tune four modules, calibrate thresholds, sealed outer prediction")
        train_runtime, train_rows, train_targets, eval_runtime, eval_rows, eval_targets, embargo = self.data.scope(cell, fold, seed)
        train_questions = make_questions(train_runtime, train_rows, train_targets)
        eval_questions = make_questions(eval_runtime, eval_rows, eval_targets)
        manifest_path = self.output / "audit/inner_split_manifests" / cell / f"outer_fold_{fold}.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            current_documents = {q["canonical_document_id"] for q in train_questions}
            if current_documents != set(manifest["assignment"]):
                raise RuntimeError("inner split document universe differs across seeds")
        else:
            if seed != 42: raise RuntimeError("seed-42 inner split reference must be created first")
            manifest = inner_manifest(train_questions, self.config.inner_folds)
            write_json(manifest_path, manifest)
        write_json(state / "embargo.json", embargo)
        models, selection, oof = train_srr(train_questions, seed, manifest, self.progress)
        model_root = self.output / "models" / cell / f"outer_fold_{fold}" / f"seed_{seed}"
        hashes = self._save_models(model_root, models, selection)
        write_jsonl(self.output / "oof/modules" / cell / f"outer_fold_{fold}" / f"seed_{seed}.jsonl", [
            {"question_id": qid, **value} for qid, value in sorted(oof.items())
        ])
        write_json(self.output / "oof/threshold_search" / cell / f"outer_fold_{fold}" / f"seed_{seed}.json", selection["policies"])
        module_predictions = predict_all(models, eval_questions)
        generated = self._new_predictions(cell, fold, seed, eval_questions, module_predictions, selection, hashes, manifest["sha256"])
        historical = {"S0": self._historical_prediction("S0", cell, fold, seed), "S1": self._historical_prediction("S1", cell, fold, seed)}
        for system, rows in {**historical, **generated}.items():
            write_jsonl(self.output / "predictions" / system / cell / f"outer_fold_{fold}" / f"seed_{seed}.jsonl", rows)
        # Runtime resource equality is asserted against exact H2 question by question.
        s1 = {x["runtime"]["question_id"]: x for x in historical["S1"]}
        for row in generated["S2"]:
            old = s1[row["runtime"]["question_id"]]["runtime"]
            if row["runtime"]["qa_forwards"] != old["qa_forwards"] or row["runtime"]["q_a_ranker_forwards"] != old["q_a_ranker_forwards"]:
                raise AssertionError("S2/S1 neural forward equality failed")
        write_json(state / "COMPLETED.json", {"status": "complete", "cell": cell, "fold": fold, "seed": seed, "inner_split_sha256": manifest["sha256"], "policies": {k: v["fold_policy"] for k, v in selection["policies"].items()}})

    def development(self):
        self._status("RUNNING"); self.progress.stage("development", "V10 LOCAL five-cell development evaluation")
        units = [(cell, fold, seed) for cell in self.config.datasets for fold in self.config.outer_folds for seed in self.config.seeds]
        for index, (cell, fold, seed) in enumerate(units, 1):
            self.progress.count(index, len(units), item="outer cell/fold/seed", extra={"cell": cell, "fold": fold, "seed": seed})
            try:
                self._execute(cell, fold, seed)
            except Exception as exc:
                write_json(self.output / "state" / cell / f"outer_fold_{fold}" / f"seed_{seed}" / "FAILED.json", {
                    "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                    "silent_surrogate_used": False,
                })
                raise
        hashes = {}
        for path in sorted((self.output / "audit/inner_split_manifests").glob("*/outer_fold_*.json")):
            hashes[str(path.relative_to(self.output))] = json.loads(path.read_text())["sha256"]
        write_json(self.output / "audit/inner_split_hashes.json", hashes)
        self._status("RUN_COMPLETE", completed_outer_units=len(units))

    def finish(self):
        self.progress.stage("finalize", "verify V1-V9 history unchanged and enforce complete 5×5×3 matrix")
        expected = len(self.config.datasets) * len(self.config.outer_folds) * len(self.config.seeds)
        completed = list((self.output / "state").glob("*/outer_fold_*/seed_*/COMPLETED.json"))
        if len(completed) != expected: raise RuntimeError(f"incomplete V10 run: {len(completed)} != {expected}")
        after = historical_hashes(self.config); write_json(self.output / "audit/historical_hashes_after.json", after)
        git_snapshot(self.output / "audit/git_after")
        if self.before is None: self.before = json.loads((self.output / "audit/historical_hashes_before.json").read_text())
        if self.before != after: raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED")
        write_json(self.output / "audit/regression_summary.json", {
            "historical_hashes_unchanged": True, "runtime_gold_leakage": False,
            "outer_eval_tuning": False, "same_inner_splits_all_modules_and_seeds": True,
            "scier_cross_cell_embargo": True, "manual_dataset_language_aspect_rules": False,
            "same_neural_forwards_s1_s2": True,
        })
        failed = self.output / "FAILED.json"
        if failed.is_file(): write_json(self.output / "audit/recovered_failure.json", json.loads(failed.read_text())); remove_file(failed)
        self._status("COMPLETED")
        write_json(self.output / "COMPLETED.json", {"status": "COMPLETED", "elapsed_s": time.time() - self.started, "outer_units": expected, "historical_hashes_unchanged": True, "estimate": "DEVELOPMENT"})
