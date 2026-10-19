"""End-to-end V9 orchestration with three logically separated CV levels."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time


from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection_v8.features import normalize_weights
from experiments.qa_chunk_selection_v8.models import tune_policy
from experiments.qa_chunk_selection_v8.pipeline import _embargo, build_predictions
from .audit import git_snapshot, historical_hashes, run_audit
from .config import Config
from .contract import CONFIGS, CONFIG_COLUMNS, DIAGNOSTIC_SYSTEMS, SYSTEM_TO_CONFIG, frozen_contract
from .data import DataStore
from .io_guard import read_jsonl, remove_file, sha256, write_joblib, write_json, write_jsonl, write_text
from .replay import replay
from .selection import make_select_split_manifest, select_configuration, split_rows, split_runtime


def _same_predictions(left: list[dict], right: list[dict]) -> bool:
    def signature(rows):
        return sorted(
            (
                str(x["runtime"]["question_id"]),
                int(x["runtime"]["seed"]),
                int(x["runtime"]["selected_chunk_id"]),
                str(x["runtime"]["final_answer"]),
            )
            for x in rows
        )
    return signature(left) == signature(right)


class Pipeline:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.output = config.run_dir
        self.progress = ProgressLogger(self.output / "progress.jsonl")
        self.started = time.time()
        self.data = DataStore(config, self.progress)
        self.before = None

    def _status(self, status: str, **extra) -> None:
        write_json(self.output / "status.json", {"status": status, "elapsed_s": time.time() - self.started, **extra})

    def _heartbeat(self, stop: threading.Event) -> None:
        while not stop.wait(30):
            self.progress.log("still running", elapsed_seconds=round(time.time() - self.started, 1))

    def run(self) -> int:
        stop = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True)
        heartbeat.start()
        try:
            self._status("CREATED")
            if self.config.phase in {"audit", "replay", "run", "all"}:
                self.audit()
            if self.config.phase == "audit":
                return 0
            if self.config.phase in {"replay", "run", "all"}:
                self.historical_replay()
            if self.config.phase == "replay":
                return 0
            if self.config.phase in {"run", "all"}:
                self.freeze()
                self.development()
            if self.config.phase in {"analysis", "all"}:
                from .analyze import analyze_run
                analyze_run(self.output, self.config.bootstrap, self.progress)
            if self.config.phase in {"benchmark", "all"}:
                if self.config.smoke:
                    write_json(self.output / "resource_benchmark/SKIPPED.json", {"reason": "smoke mode"})
                else:
                    from .benchmark import run_benchmarks
                    run_benchmarks(self.config, self.output, self.progress)
            if self.config.phase == "all":
                self.finish()
            else:
                self._status(f"PHASE_{self.config.phase.upper()}_COMPLETE")
            return 0
        except Exception as exc:
            self._status("FAILED", error_type=type(exc).__name__, error=str(exc))
            write_json(self.output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc)})
            self.progress.warn(f"FAILED {type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            heartbeat.join(1)

    def audit(self) -> None:
        self._status("AUDIT_RUNNING")
        self.progress.stage("audit", "snapshot git state, freeze provenance and hash read-only V1-V8 history")
        git_snapshot(self.output / "audit/git_before")
        self.before = run_audit(self.output / "audit", self.config)
        contract = frozen_contract()
        contract_path = self.output / "frozen_contract.yaml"
        write_text(contract_path, json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        manifest = {
            "config": self.config.payload(),
            "contract_sha256": sha256(contract_path),
            "historical_read_only": True,
            "estimate": "DEVELOPMENT",
        }
        path = self.output / "manifest.json"
        if path.is_file() and self.config.resume:
            old = json.loads(path.read_text())
            stable = (
                "v4_4_run", "v4_4_external_run", "v4_5_run", "v5_run",
                "v6_run", "v7_run", "v8_run", "datasets", "seeds",
                "outer_folds", "inner_folds", "select_folds", "smoke",
            )
            mismatch = {
                key: (old["config"].get(key), manifest["config"].get(key))
                for key in stable if old["config"].get(key) != manifest["config"].get(key)
            }
            if mismatch:
                raise RuntimeError(f"resume fingerprint mismatch: {mismatch}")
            if old["contract_sha256"] != manifest["contract_sha256"]:
                raise RuntimeError("frozen V9 contract hash mismatch")
        else:
            write_json(path, manifest)
        self._status("AUDIT_PASSED")

    def historical_replay(self) -> None:
        done = self.output / "audit/replay_COMPLETED.json"
        if done.is_file():
            self.progress.log("exact V8 replay already complete")
            return
        self._status("REPLAY_RUNNING")
        self.progress.stage("replay", "exact candidate/answer/switch replay for H0, H1, H2 and H3")
        replay(self.config, self.output / "audit", self.progress)
        write_json(done, {"status": "PASS", "systems": ["H0", "H1", "H2", "H3"]})
        self._status("REPLAY_PASSED")

    def freeze(self) -> None:
        self.progress.stage("freeze", "verify frozen contract precedes all V9 outer evaluation")
        contract_path = self.output / "frozen_contract.yaml"
        if not contract_path.is_file():
            raise RuntimeError("V9 frozen contract missing")
        expected = json.loads((self.output / "manifest.json").read_text())["contract_sha256"]
        if sha256(contract_path) != expected:
            raise RuntimeError("V9 frozen contract changed after audit")
        write_json(self.output / "audit/freeze.json", {
            "status": "FROZEN_BEFORE_OUTER_EVAL",
            "sha256": expected,
            "pairlex_increment_is_empty_under_exact_h2": CONFIG_COLUMNS["CP"] == CONFIG_COLUMNS["C0"],
        })

    def _scope(self, cell: str, fold: int, seed: int):
        train_runtime, train_rows, train_targets = self.data.load(cell, fold, seed, "meta")
        eval_runtime, eval_rows, eval_targets = self.data.load(cell, fold, seed, "eval")
        train_runtime, train_rows, train_targets, embargo = _embargo(
            train_runtime, train_rows, train_targets, eval_runtime,
        )
        normalize_weights(train_rows, pooled=False)
        return train_runtime, train_rows, train_targets, eval_runtime, eval_rows, eval_targets, embargo

    def _save_policy(self, root: Path, policy, *, config_id: str, stage: str, extra: dict | None = None) -> None:
        write_joblib(root / "utility_model.joblib", policy.utility_model)
        write_joblib(root / "harm_model.joblib", policy.harm_model)
        write_json(root / "selection.json", {
            "config_id": config_id,
            "stage": stage,
            "feature_columns": list(policy.columns),
            "selection": policy.selection,
            **(extra or {}),
        })

    @staticmethod
    def _annotate(rows: list[dict], *, config_id: str, split_role: str, select_fold: int | None = None) -> list[dict]:
        for row in rows:
            runtime = row["runtime"]
            runtime["selected_component_config"] = config_id
            runtime["split_role"] = split_role
            runtime["select_fold"] = select_fold
            runtime["component_selector_used_outer_eval"] = False
        return rows

    def _select_oof_for_seed(
        self,
        cell: str,
        outer_fold: int,
        seed: int,
        runtime: list[dict],
        rows: list[dict],
        targets: dict,
        split_manifest: dict,
    ) -> dict[str, list[dict]]:
        output = {config: [] for config in CONFIGS}
        for select_fold in range(self.config.select_folds):
            select_train, select_valid_rows = split_rows(rows, split_manifest, select_fold)
            select_valid_runtime, select_valid_targets = split_runtime(runtime, targets, split_manifest, select_fold)
            train_docs = {x["canonical_document_id"] for x in select_train}
            valid_docs = {x["canonical_document_id"] for x in select_valid_runtime}
            if train_docs & valid_docs:
                raise AssertionError("SELECT_VALID document used in config tuning")
            policies = {}
            for config_id in ("C0", "CA"):
                self.progress.log(
                    "nested config tuning",
                    cell=cell, outer_fold=outer_fold, seed=seed,
                    select_fold=select_fold, config=config_id,
                    train_rows=len(select_train), valid_rows=len(select_valid_rows),
                )
                policy = tune_policy(
                    copy.deepcopy(select_train), config_id, CONFIG_COLUMNS[config_id],
                    seed, self.config.inner_folds, True,
                )
                policies[config_id] = policy
                model_root = (
                    self.output / "selections/tuning_manifests" / cell / f"outer_fold_{outer_fold}"
                    / f"seed_{seed}" / f"select_fold_{select_fold}" / config_id
                )
                self._save_policy(model_root, policy, config_id=config_id, stage="CONFIG_TUNING", extra={
                    "select_train_documents": sorted(train_docs),
                    "select_valid_documents": sorted(valid_docs),
                    "select_valid_used_for_tuning": False,
                })
            policies["CP"] = policies["C0"]
            policies["CAP"] = policies["CA"]
            for alias, source in (("CP", "C0"), ("CAP", "CA")):
                alias_root = (
                    self.output / "selections/tuning_manifests" / cell / f"outer_fold_{outer_fold}"
                    / f"seed_{seed}" / f"select_fold_{select_fold}" / alias
                )
                self._save_policy(
                    alias_root, policies[alias], config_id=alias, stage="CONFIG_TUNING",
                    extra={
                        "exact_schema_alias_of": source,
                        "reason": "exact H2 already contains the full PairLEX group",
                        "select_train_documents": sorted(train_docs),
                        "select_valid_documents": sorted(valid_docs),
                        "select_valid_used_for_tuning": False,
                    },
                )
            for config_id in CONFIGS:
                predicted = build_predictions(
                    "hrr", cell, select_valid_runtime, select_valid_rows, select_valid_targets,
                    config_id, "local", outer_fold, seed, policies[config_id],
                )
                predicted = self._annotate(
                    predicted, config_id=config_id, split_role="SELECT_OOF", select_fold=select_fold,
                )
                output[config_id].extend(predicted)
                path = (
                    self.output / "selections/select_fold_predictions" / cell / f"outer_fold_{outer_fold}"
                    / f"seed_{seed}" / config_id / f"select_fold_{select_fold}.jsonl"
                )
                write_jsonl(path, predicted)
        expected_questions = len({x["question_id"] for x in runtime})
        for config_id, values in output.items():
            if len(values) != expected_questions:
                raise RuntimeError(
                    f"SELECT_OOF coverage incomplete: {cell} fold={outer_fold} seed={seed} "
                    f"config={config_id}: {len(values)} != {expected_questions}"
                )
        if not _same_predictions(output["C0"], output["CP"]):
            raise AssertionError("CP must equal C0 because exact H2 already contains PairLEX")
        if not _same_predictions(output["CA"], output["CAP"]):
            raise AssertionError("CAP must equal CA because exact H2 already contains PairLEX")
        return output

    def _assert_v8_identity(self, cell: str, fold: int, seed: int, system: str, rows: list[dict]) -> None:
        historical = read_jsonl(
            self.config.v8_run / "branch_hrr/predictions/local" / cell / cell
            / f"outer_fold_{fold}" / f"seed_{seed}" / f"{system}.jsonl"
        )
        if not _same_predictions(historical, rows):
            raise RuntimeError(f"V9 exact {system} identity failed: {cell} fold={fold} seed={seed}")

    def _execute_outer_fold(self, cell: str, outer_fold: int) -> None:
        state = self.output / "state/local" / cell / f"outer_fold_{outer_fold}"
        done = state / "COMPLETED.json"
        if done.is_file():
            self.progress.log("outer fold already complete", cell=cell, outer_fold=outer_fold)
            return
        self.progress.stage(
            f"local:{cell}:f{outer_fold}",
            "build nested SELECT_OOF, select one config for all seeds, refit and evaluate once",
        )
        seed_data = {}
        split_manifest = None
        for seed in self.config.seeds:
            values = self._scope(cell, outer_fold, seed)
            seed_data[seed] = values
            current_manifest = make_select_split_manifest(values[0])
            if split_manifest is None:
                split_manifest = current_manifest
            elif current_manifest["sha256"] != split_manifest["sha256"]:
                raise RuntimeError("SELECT split assignment differs across model seeds")
            write_json(state / f"embargo_seed_{seed}.json", values[-1])
        assert split_manifest is not None
        split_path = self.output / "audit/select_split_manifests" / cell / f"outer_fold_{outer_fold}.json"
        write_json(split_path, split_manifest)

        select_predictions = {config: [] for config in CONFIGS}
        for seed in self.config.seeds:
            runtime, rows, targets, *_ = seed_data[seed]
            per_seed = self._select_oof_for_seed(
                cell, outer_fold, seed, runtime, rows, targets, split_manifest,
            )
            for config_id in CONFIGS:
                select_predictions[config_id].extend(per_seed[config_id])
        selection = select_configuration(select_predictions, self.config.seeds)
        selection.update(cell=cell, outer_fold=outer_fold, select_split_sha256=split_manifest["sha256"])
        selection_path = self.output / "selections/manifests" / cell / f"outer_fold_{outer_fold}.json"
        write_json(selection_path, selection)
        selected_config = selection["selected_config"]
        self.progress.log(
            "component configuration frozen for outer fold",
            cell=cell, outer_fold=outer_fold, selected_config=selected_config,
            reason=selection["selection_reason"],
        )

        for seed in self.config.seeds:
            train_runtime, train_rows, train_targets, eval_runtime, eval_rows, eval_targets, _ = seed_data[seed]
            policies = {}
            for config_id in ("C0", "CA"):
                self.progress.log(
                    "final outer-train refit", cell=cell, outer_fold=outer_fold,
                    seed=seed, config=config_id, rows=len(train_rows),
                )
                policy = tune_policy(
                    copy.deepcopy(train_rows), config_id, CONFIG_COLUMNS[config_id],
                    seed, self.config.inner_folds, True,
                )
                policies[config_id] = policy
                root = self.output / "models/diagnostics" / config_id / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}"
                self._save_policy(root, policy, config_id=config_id, stage="FINAL_OUTER_TRAIN")
            policies["CP"] = policies["C0"]
            policies["CAP"] = policies["CA"]
            for alias, source in (("CP", "C0"), ("CAP", "CA")):
                root = self.output / "models/diagnostics" / alias / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}"
                self._save_policy(
                    root, policies[alias], config_id=alias, stage="FINAL_OUTER_TRAIN",
                    extra={
                        "exact_schema_alias_of": source,
                        "reason": "exact H2 already contains the full PairLEX group",
                    },
                )

            s0 = self._annotate(
                build_predictions("hrr", cell, eval_runtime, eval_rows, eval_targets, "H0", "local", outer_fold, seed),
                config_id="H0", split_role="OUTER_EVAL",
            )
            config_outputs = {}
            for config_id in CONFIGS:
                config_outputs[config_id] = self._annotate(
                    build_predictions(
                        "hrr", cell, eval_runtime, eval_rows, eval_targets,
                        config_id, "local", outer_fold, seed, policies[config_id],
                    ),
                    config_id=config_id, split_role="OUTER_EVAL",
                )
            self._assert_v8_identity(cell, outer_fold, seed, "H2", config_outputs["C0"])
            self._assert_v8_identity(cell, outer_fold, seed, "H3", config_outputs["CA"])
            if not _same_predictions(config_outputs["C0"], config_outputs["CP"]):
                raise AssertionError("outer CP/C0 identity failed")
            if not _same_predictions(config_outputs["CA"], config_outputs["CAP"]):
                raise AssertionError("outer CAP/CA identity failed")

            systems = {
                "S0": s0,
                "S1": config_outputs["C0"],
                "S2": config_outputs[selected_config],
                "D-A": config_outputs["CA"],
                "D-P": config_outputs["CP"],
                "D-AP": config_outputs["CAP"],
            }
            for system, predicted in systems.items():
                for row in predicted:
                    row["runtime"]["system"] = system
                    row["runtime"]["selected_component_config"] = (
                        selected_config if system == "S2" else SYSTEM_TO_CONFIG.get(system, "H0")
                    )
                path = self.output / "predictions" / system / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}.jsonl"
                write_jsonl(path, predicted)
            selected_policy = policies[selected_config]
            s2_root = self.output / "models/S2" / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}"
            self._save_policy(s2_root, selected_policy, config_id=selected_config, stage="FINAL_SELECTED")
            s1_root = self.output / "models/S1" / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}"
            self._save_policy(s1_root, policies["C0"], config_id="C0", stage="FINAL_FIXED_H2")
        write_json(done, {
            "status": "complete", "cell": cell, "outer_fold": outer_fold,
            "seeds": list(self.config.seeds), "selected_config": selected_config,
            "outer_eval_used_for_selection": False,
        })

    def development(self) -> None:
        self._status("RUNNING")
        self.progress.stage("development", "V9 LOCAL primary with nested component selection")
        total = len(self.config.datasets) * len(self.config.outer_folds)
        index = 0
        for cell in self.config.datasets:
            for outer_fold in self.config.outer_folds:
                index += 1
                self.progress.count(index, total, item="outer cell/fold", extra={"cell": cell, "outer_fold": outer_fold})
                self._execute_outer_fold(cell, outer_fold)
        manifests = []
        hashes = {}
        for path in sorted((self.output / "selections/manifests").glob("*/outer_fold_*.json")):
            manifests.append(json.loads(path.read_text()))
        for path in sorted((self.output / "audit/select_split_manifests").glob("*/outer_fold_*.json")):
            hashes[str(path.relative_to(self.output))] = json.loads(path.read_text())["sha256"]
        write_jsonl(self.output / "selections/component_selection_rows.jsonl", manifests)
        write_json(self.output / "audit/select_split_hashes.json", hashes)
        self._status("RUN_COMPLETE", completed_outer_units=total)

    def finish(self) -> None:
        self.progress.stage("finalize", "assert V1-V8 hashes unchanged and write completion marker")
        after = historical_hashes(self.config)
        write_json(self.output / "audit/historical_hashes_after.json", after)
        git_snapshot(self.output / "audit/git_after")
        if self.before is None:
            self.before = json.loads((self.output / "audit/historical_hashes_before.json").read_text())
        if self.before != after:
            raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED")
        write_json(self.output / "audit/regression_summary.json", {
            "historical_hashes_unchanged": True,
            "runtime_gold_leakage": False,
            "selection_outer_eval_leakage": False,
            "seed_isolation": True,
            "select_prediction_oof": True,
            "same_select_splits_all_configs_and_seeds": True,
            "scier_cross_cell_embargo": True,
            "manual_dataset_or_language_rules": False,
            "same_neural_forwards_s1_s2": True,
        })
        failed = self.output / "FAILED.json"
        if failed.is_file():
            write_json(self.output / "audit/recovered_failure.json", {
                "recovered_by_resume": True,
                "failure": json.loads(failed.read_text()),
            })
            remove_file(failed)
        self._status("COMPLETED")
        write_json(self.output / "COMPLETED.json", {
            "status": "COMPLETED",
            "elapsed_s": time.time() - self.started,
            "historical_hashes_unchanged": True,
            "primary_systems": ["S1", "S2"],
            "estimate": "DEVELOPMENT",
        })

