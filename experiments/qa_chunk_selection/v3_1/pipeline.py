"""V3.1-Silver ranking pipeline. Isolated artifacts; reuses V3 trainers/eval."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from experiments.qa_chunk_selection.models import ModelBundle, load_models
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, hn_quality_rows
from experiments.qa_chunk_selection.v2.qa import answer_top_m
from experiments.qa_chunk_selection.v2.schemas import rank_ids, to_runtime
from experiments.qa_chunk_selection.v2.statistics import paired_delta_multiseed
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records, write_records
from experiments.qa_chunk_selection.v3.evaluate import run_inference_v3
from experiments.qa_chunk_selection.v3.metrics_v3 import summarize_v3
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v3.train import load_ranker, score_fn_for, train_b0
from experiments.qa_chunk_selection.v3_1_auto_prep.qc import file_sha256
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import overlay_index, silver_to_v3_training_overlay

from .audit import s0_audit
from .config import ExperimentConfigV31
from .contract import experiment_contract_v3_1
from .io_guard import assert_isolated_output, load_json, read_jsonl, write_json, write_jsonl
from .overlays import index_fold_overlays, load_fold_overlays, missing_freeze_files, overlay_hashes
from .silver_metrics import enrich_silver, summarize_silver
from .train_silver import train_b1_silver, train_r1_silver


class PipelineV31:
    def __init__(self, config: ExperimentConfigV31) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3_cfg = config.as_v3_config()
        self.unlocked = ()
        self.flags: list[dict[str, Any]] = []
        self.splits: dict[str, Any] = {}
        self.silver_rows: list[dict[str, Any]] = []
        self.silver_by_run: dict[int, list[dict[str, Any]]] = {}
        self.overlay: dict[tuple[str, int], dict[str, Any]] = {}
        self.training_overlay: dict[tuple[str, int], dict[str, Any]] = {}
        self.overlays_by_run: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
        self.training_overlays_by_run: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
        self.bundle: ModelBundle | None = None
        self.cache = FeatureCache(self.output_dir / "cache")
        self.g1: dict[str, Any] = {}
        self.exit_code = 0

    def run(self) -> int:
        cfg = self.config
        self.progress.stage("start", f"output={self.output_dir} smoke={cfg.smoke}")
        write_json(self.output_dir / "contract.json", experiment_contract_v3_1())
        write_json(self.output_dir / "config_resolved.json", cfg.to_dict())
        dispatch = {
            "check-preconditions": self.stage_preconditions,
            "prepare": self.stage_prepare,
            "s0-audit": self.stage_s0,
            "b0": lambda: self.stage_family("b0"),
            "b1": lambda: self.stage_family("b1"),
            "oof-compare": self.stage_oof_compare,
            "g1": self.stage_g1,
            "report": self.stage_report,
            "r1": self.stage_r1,
            "pairwise": lambda: self._skip_gated("pairwise", "requires G1 target signal and a later protocol version"),
            "listwise": lambda: self._skip_gated("listwise", "requires confirmed target signal; not in default V3.1 first run"),
            "attention-audit": lambda: self._skip_gated("attention-audit", "attention specialist only after TRAIN/CAL signal"),
            "d4-oracle": lambda: self._skip_gated("d4-oracle", "D4 deferred until after B1/R1 gates"),
            "d5-audit": lambda: self._skip_gated("d5-audit", "D5 deferred until independent-evidence audit is scheduled"),
            "freeze-final": lambda: self._skip_gated("freeze-final", "INTERNAL_TEST / final freeze is locked"),
            "final-eval": lambda: self._skip_gated("final-eval", "INTERNAL_TEST is locked"),
        }
        for name in cfg.stages:
            fn = dispatch.get(name)
            if fn is None:
                raise ValueError(f"no handler for stage {name}")
            self.progress.stage(name)
            fn()
        self.progress.stage("done", "V3.1 ranking finished (prior experiment artifacts untouched)")
        return self.exit_code

    def stage_preconditions(self) -> None:
        if self.config.smoke:
            self.progress.log("smoke: silver freeze not required; synthetic fold overlays will be built in prepare")
            write_json(self.output_dir / "analysis" / "preconditions.json", {"ok": True, "smoke": True})
            return
        silver_root = Path(self.config.silver_dir)
        missing = missing_freeze_files(silver_root)
        readiness = silver_root / "analysis" / "readiness.json"
        payload: dict[str, Any] = {"silver_dir": str(silver_root), "missing": missing}
        if readiness.exists():
            ready = load_json(readiness)
            payload["readiness"] = ready
            if not ready.get("v3_1_training_allowed"):
                missing.append("v3_1_training_allowed=false")
        elif "analysis/readiness.json" not in missing:
            missing.append("analysis/readiness.json")
        if missing:
            write_json(self.output_dir / "analysis" / "preconditions.json", {**payload, "ok": False})
            raise PermissionError(
                "V3.1 ranking requires a frozen QA-silver overlay with fold-specific labels. "
                f"Missing/blocked: {missing}. Run v3_1_auto_prep with STAGES=full first. "
                "Do not train on the empty manual overlay or mix another run's thresholds."
            )
        payload.update({"ok": True, "annotator": "frozen_extractive_qa"})
        write_json(self.output_dir / "analysis" / "preconditions.json", payload)
        self.progress.log("QA-silver freeze accepted", n_missing=0)

    def stage_prepare(self) -> None:
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        v3 = Path(self.config.v3_artifact_dir)
        if self.config.smoke:
            self.progress.log("smoke: synthetic unlocked records")
            records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
            write_records(data_dir / "dataset.jsonl", self.unlocked)
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=self.config.split_seed, output_path=data_dir / "v3_cv_splits.json")
            self.flags = hn_quality_rows(self.unlocked, include_hn_gold_warning=False)
            write_jsonl(data_dir / "hn_quality_flags.jsonl", self.flags)
            self.silver_rows = _synthetic_silver(self.unlocked)
        else:
            dataset = v3 / "data" / "dataset.jsonl"
            splits = v3 / "data" / "v3_cv_splits.json"
            flags = v3 / "data" / "hn_quality_flags.jsonl"
            if not dataset.exists() or not splits.exists():
                raise FileNotFoundError(f"V3 dataset/splits missing under {v3}")
            self.progress.log("loading frozen V3 universe (read-only source)")
            records = records_from_jsonl(dataset)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
            write_records(data_dir / "dataset.jsonl", self.unlocked)
            self.splits = load_json(splits)
            write_json(data_dir / "v3_cv_splits.json", self.splits)
            if flags.exists():
                self.flags = read_jsonl(flags)
                write_jsonl(data_dir / "hn_quality_flags.jsonl", self.flags)
            else:
                self.flags = hn_quality_rows(self.unlocked, include_hn_gold_warning=False)
                write_jsonl(data_dir / "hn_quality_flags.jsonl", self.flags)
            self.silver_by_run = load_fold_overlays(Path(self.config.silver_dir))
            first_run = min(self.silver_by_run)
            self.silver_rows = self.silver_by_run[first_run]
            write_json(
                data_dir / "source_hashes.json",
                {
                    "v3_dataset": file_sha256(dataset),
                    "v3_splits": file_sha256(splits),
                    "silver_overlays": overlay_hashes(Path(self.config.silver_dir), self.silver_by_run),
                    "document_id_hash": self.splits.get("document_id_hash"),
                    "split_payload_hash": self.splits.get("split_payload_hash"),
                    "new_outer_folds": False,
                    "fold_specific_silver": True,
                },
            )
        if self.config.smoke and not self.silver_by_run:
            self.silver_by_run = {int(run["run"]): list(self.silver_rows) for run in self.splits.get("runs") or [{"run": 0}]}
        self.overlays_by_run, self.training_overlays_by_run = index_fold_overlays(self.silver_by_run)
        first = min(self.overlays_by_run) if self.overlays_by_run else 0
        self.overlay = self.overlays_by_run.get(first) or overlay_index(self.silver_rows)
        self.training_overlay = self.training_overlays_by_run.get(first) or silver_to_v3_training_overlay(self.silver_rows)
        write_jsonl(self.output_dir / "annotation" / "silver_target_status.jsonl", self.silver_rows)
        for run_id, rows in self.silver_by_run.items():
            write_jsonl(self.output_dir / "annotation" / "oof" / f"run{run_id}" / "silver_target_status.jsonl", rows)
        write_json(
            data_dir / "prepare_summary.json",
            {
                "n_unlocked": len(self.unlocked),
                "n_silver_rows": len(self.silver_rows),
                "n_training_overlay": len(self.training_overlay),
                "n_fold_overlays": len(self.overlays_by_run),
                "internal_test_locked": True,
                "source_label_rewritten": False,
                "fold_specific_silver": True,
            },
        )
        self.progress.log(
            "prepare done",
            unlocked=len(self.unlocked),
            silver=len(self.silver_rows),
            training_labels=len(self.training_overlay),
        )

    def stage_s0(self) -> None:
        if not self.unlocked:
            self.stage_prepare()
        b0_path = Path(self.config.v3_artifact_dir) / "analysis" / "oof" / "b0" / "predictions_scored.jsonl"
        if b0_path.exists() and not self.config.smoke:
            b0_rows = read_jsonl(b0_path)
            self.progress.log("S0 using historical V3 B0 OOF (read-only)")
        else:
            b0_rows = []
            for item in self.unlocked:
                ids = [chunk.chunk_id for chunk in item.chunks]
                b0_rows.append(
                    {
                        "question_id": item.question_id,
                        "document_id": item.document_id,
                        "ranked_ids": ids,
                        "scores": {str(cid): float(len(ids) - i) for i, cid in enumerate(ids)},
                        "seed": 42,
                    }
                )
            self.progress.log("S0 using order-based placeholder ranks (smoke or missing V3 B0)")
        per_run = []
        for run in self._runs():
            run_id = int(run["run"])
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            overlay = self._silver_overlay(run_id)
            ev_ids = {item.question_id for item in ev}
            payload = s0_audit(
                records=ev,
                overlay=overlay,
                b0_rows=[row for row in b0_rows if str(row.get("question_id")) in ev_ids],
                progress=self.progress,
            )
            payload["run"] = run_id
            per_run.append(payload)
            self.progress.log(
                f"S0 run {run_id}",
                both=payload.get("n_questions_with_target_and_tm"),
                shortlist_signal=payload.get("target_vs_tm_present_in_b0_shortlist"),
            )
        pooled = s0_audit(records=self.unlocked, overlay=self.overlay, b0_rows=b0_rows, progress=self.progress)
        payload = {
            **pooled,
            "per_run": per_run,
            "target_vs_tm_present_in_b0_shortlist": any(row.get("target_vs_tm_present_in_b0_shortlist") for row in per_run)
            or bool(pooled.get("target_vs_tm_present_in_b0_shortlist")),
            "n_questions_with_target_and_tm": max(
                [int(row.get("n_questions_with_target_and_tm") or 0) for row in per_run] + [int(pooled.get("n_questions_with_target_and_tm") or 0)]
            ),
        }
        write_json(self.output_dir / "analysis" / "s0_audit.json", payload)
        self.progress.log(
            "S0 audit",
            both=payload.get("n_questions_with_target_and_tm"),
            shortlist_signal=payload.get("target_vs_tm_present_in_b0_shortlist"),
        )

    def _silver_overlay(self, run_id: int) -> dict[tuple[str, int], dict[str, Any]]:
        return self.overlays_by_run.get(int(run_id)) or self.overlay

    def _training_overlay(self, run_id: int) -> dict[tuple[str, int], dict[str, Any]]:
        return self.training_overlays_by_run.get(int(run_id)) or self.training_overlay

    def _seeds(self) -> tuple[int, ...]:
        return (self.config.seed,) if self.config.smoke else self.config.model_seeds

    def _epochs(self) -> int:
        return 1 if self.config.smoke else self.config.b0_epochs

    def _patience(self) -> int:
        return 1 if self.config.smoke else self.config.b0_patience

    def _device(self) -> str:
        return self.bundle.device if self.bundle else "cpu"

    def _ensure_models(self) -> None:
        if self.bundle is not None:
            return
        self.progress.log("loading encoders / QA")
        self.bundle = load_models(
            device=self.config.device,
            cross_encoder_model=self.config.cross_encoder_model,
            smoke=self.config.smoke,
        )

    def _qa_top1(self, runtime, ranked):
        assert self.bundle is not None
        return answer_top_m(
            runtime,
            ranked,
            answer_fn=self.bundle.answer,
            qa_answerer=self.bundle.qa_answerer,
            m=1,
            cache=self.cache,
            model_id=str(self.bundle.qa_model_name),
            inference_hash="v3_1_silver",
        )

    def _runs(self) -> list[dict[str, Any]]:
        return list(self.splits["runs"])

    def stage_family(self, family: str) -> None:
        if not self.unlocked:
            self.stage_prepare()
        self._ensure_models()
        cfg = self.config
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            for seed in self._seeds():
                out = self.v3_cfg.run_dir(family, seed) / f"run{run_id}"
                if cfg.resume and (out / "wrapper.json").exists():
                    self.progress.log(f"resume {family} run={run_id} seed={seed}")
                else:
                    self.progress.log(f"train {family} run={run_id} seed={seed} n_train={len(train)} n_cal={len(cal)}")
                    kwargs = dict(
                        seed=int(seed),
                        device=self._device(),
                        model_name=cfg.cross_encoder_model,
                        output_dir=out,
                        epochs=self._epochs(),
                        patience=self._patience(),
                        learning_rate=cfg.b2_learning_rate,
                        max_length=cfg.b2_max_length,
                        smoke=cfg.smoke,
                        progress=self.progress,
                        batch_size=cfg.b2_batch_size,
                    )
                    if family == "b0":
                        train_b0(train, cal, self.flags, ratio=cfg.b2_pos_hn_easy, **kwargs)
                    elif family == "b1":
                        train_b1_silver(
                            train,
                            cal,
                            self._training_overlay(run_id),
                            self.flags,
                            ratio=cfg.b2_pos_hn_easy,
                            easy_cap=cfg.easy_negative_cap,
                            **kwargs,
                        )
                    else:
                        raise ValueError(family)
                model = load_ranker(out, self._device(), cfg.smoke)
                if model is None:
                    continue
                self._infer(ev, score_fn_for(model), family, run_id, seed)
        self._write_oof(family)

    def _infer(self, records, score_fn, family: str, run_id: int, seed: int) -> None:
        dest = self.v3_cfg.run_dir(family, seed) / f"run{run_id}" / "eval"
        training_overlay = self._training_overlay(run_id)
        silver_overlay = self._silver_overlay(run_id)
        run_inference_v3(
            records,
            score_fn,
            ranking_method=family,
            split=f"cv{run_id}_eval",
            output_dir=dest,
            config=self.v3_cfg,
            overlay=training_overlay,
            progress=self.progress,
            answer_fn=self._qa_top1,
            skip_qa=self.config.skip_qa,
            flags=self.flags,
        )
        raw_rows = read_jsonl(dest / "predictions_raw.jsonl")
        by_q = {item.question_id: item for item in records}
        flags = flag_index(self.flags)
        scored = []
        for raw in raw_rows:
            item = by_q.get(str(raw.get("question_id")))
            if item is None:
                continue
            scored.append(
                enrich_silver(
                    raw,
                    item,
                    silver_overlay,
                    training_overlay,
                    flags=flags,
                    skip_bertscore=self.config.skip_bertscore,
                    ks=self.config.top_k_values,
                )
            )
        write_jsonl(dest / "predictions_scored_silver.jsonl", scored)
        write_json(dest / "metrics_silver.json", summarize_silver(scored, ks=self.config.top_k_values))

    def _write_oof(self, family: str) -> None:
        rows = []
        silver_rows = []
        for run in self._runs():
            run_id = int(run["run"])
            for seed in self._seeds():
                path = self.v3_cfg.run_dir(family, seed) / f"run{run_id}" / "eval" / "predictions_scored.jsonl"
                silver_path = self.v3_cfg.run_dir(family, seed) / f"run{run_id}" / "eval" / "predictions_scored_silver.jsonl"
                if path.exists():
                    for row in read_jsonl(path):
                        row = dict(row)
                        row["cv_run"] = run_id
                        row["seed"] = seed
                        rows.append(row)
                if silver_path.exists():
                    for row in read_jsonl(silver_path):
                        row = dict(row)
                        row["cv_run"] = run_id
                        row["seed"] = seed
                        silver_rows.append(row)
        out = self.output_dir / "analysis" / "oof" / family
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "predictions_scored.jsonl", rows)
        write_json(out / "metrics.json", summarize_v3(rows, ks=self.config.top_k_values) if rows else {"n": 0})
        write_jsonl(out / "predictions_scored_silver.jsonl", silver_rows)
        write_json(out / "metrics_silver.json", summarize_silver(silver_rows, ks=self.config.top_k_values))
        self.progress.log(f"OOF {family} n={len(rows)}")

    def stage_oof_compare(self) -> None:
        families = []
        for name in ("b0", "b1", "r1"):
            path = self.output_dir / "analysis" / "oof" / name / "metrics.json"
            if path.exists():
                row = load_json(path)
                row["name"] = name
                silver = self.output_dir / "analysis" / "oof" / name / "metrics_silver.json"
                if silver.exists():
                    row["silver"] = load_json(silver)
                families.append(row)
        write_json(self.output_dir / "analysis" / "oof_compare.json", {"families": families})
        bootstrap: dict[str, Any] = {}
        n_boot = 64 if self.config.smoke else self.config.n_bootstrap
        b0 = _rows_by_seed(self.output_dir / "analysis" / "oof" / "b0" / "predictions_scored.jsonl")
        b1 = _rows_by_seed(self.output_dir / "analysis" / "oof" / "b1" / "predictions_scored.jsonl")
        if b0 and b1:
            for metric in ("token_f1", "exact_match", "hit@1", "hn@1"):
                bootstrap[f"b1_minus_b0_{metric}"] = paired_delta_multiseed(
                    b0, b1, metric, n_boot=n_boot, seed=self.config.bootstrap_seed
                )
        b0s = _rows_by_seed(self.output_dir / "analysis" / "oof" / "b0" / "predictions_scored_silver.jsonl")
        b1s = _rows_by_seed(self.output_dir / "analysis" / "oof" / "b1" / "predictions_scored_silver.jsonl")
        if b0s and b1s:
            for metric in ("SilverTargetMismatch@1", "SilverPairwiseTargetVsMismatch"):
                bootstrap[f"b1_minus_b0_{metric}"] = paired_delta_multiseed(
                    b0s, b1s, metric, n_boot=n_boot, seed=self.config.bootstrap_seed
                )
        write_json(self.output_dir / "analysis" / "bootstrap.json", bootstrap)
        self.progress.log("OOF compare written", n_families=len(families))

    def stage_g1(self) -> None:
        b0 = load_json(self.output_dir / "analysis" / "oof" / "b0" / "metrics.json") if (self.output_dir / "analysis" / "oof" / "b0" / "metrics.json").exists() else {}
        b1 = load_json(self.output_dir / "analysis" / "oof" / "b1" / "metrics.json") if (self.output_dir / "analysis" / "oof" / "b1" / "metrics.json").exists() else {}
        b0s = load_json(self.output_dir / "analysis" / "oof" / "b0" / "metrics_silver.json") if (self.output_dir / "analysis" / "oof" / "b0" / "metrics_silver.json").exists() else {}
        b1s = load_json(self.output_dir / "analysis" / "oof" / "b1" / "metrics_silver.json") if (self.output_dir / "analysis" / "oof" / "b1" / "metrics_silver.json").exists() else {}
        s0 = load_json(self.output_dir / "analysis" / "s0_audit.json") if (self.output_dir / "analysis" / "s0_audit.json").exists() else {}
        hn0 = float(b0.get("hn@1") or 0.0)
        hn1 = float(b1.get("hn@1") or 0.0)
        pair0 = float(b0.get("pairwise_clean_hn") or 0.0)
        pair1 = float(b1.get("pairwise_clean_hn") or 0.0)
        f10 = float(b0.get("token_f1") or 0.0)
        f11 = float(b1.get("token_f1") or 0.0)
        safeguard = (hn1 <= hn0 + self.config.epsilon_hn) and (pair1 >= pair0 - self.config.epsilon_pairwise_hn)
        f1_ok = f11 + 1e-9 >= f10 - self.config.delta_f1_min
        silver_improved = float(b1s.get("SilverPairwiseTargetVsMismatch") or 0.0) >= float(
            b0s.get("SilverPairwiseTargetVsMismatch") or 0.0
        ) or float(b1s.get("SilverTargetMismatch@1_lower_bound") or 1.0) <= float(
            b0s.get("SilverTargetMismatch@1_lower_bound") or 1.0
        )
        shortlist = bool(s0.get("target_vs_tm_present_in_b0_shortlist"))
        tm_enough = int(b0s.get("n_with_target_and_tm_silver") or s0.get("n_questions_with_target_and_tm") or 0) > 0
        if safeguard and f1_ok and silver_improved:
            decision = "G1-A"
        elif (not f1_ok) and shortlist and tm_enough:
            decision = "G1-B"
        elif not tm_enough or not shortlist:
            decision = "G1-STOP"
        else:
            decision = "G1-STOP"
        self.g1 = {
            "decision": decision,
            "safeguards_ok": safeguard,
            "token_f1_not_worse": f1_ok,
            "silver_diagnostics_improved": silver_improved,
            "shortlist_has_target_vs_tm": shortlist,
            "b0": {"token_f1": f10, "hn@1": hn0, "pairwise_clean_hn": pair0},
            "b1": {"token_f1": f11, "hn@1": hn1, "pairwise_clean_hn": pair1},
            "interpretation": {
                "G1-A": "continue target-attribution branch",
                "G1-B": "F1 not improved; one second-stage R1 is allowed if shortlist competition exists",
                "G1-STOP": "stop the TM branch; silver signal too weak or harmful",
            }[decision],
            "note": "silver TM improvement is supporting, not sufficient, evidence",
        }
        write_json(self.output_dir / "analysis" / "g1.json", self.g1)
        self.progress.log("G1", decision=decision)

    def stage_r1(self) -> None:
        if not self.g1:
            if (self.output_dir / "analysis" / "g1.json").exists():
                self.g1 = load_json(self.output_dir / "analysis" / "g1.json")
            else:
                self.stage_g1()
        if self.g1.get("decision") not in {"G1-A", "G1-B"} and not self.config.force_r1:
            self._skip_gated("r1", f"G1={self.g1.get('decision')} does not allow R1")
            return
        if not self.unlocked:
            self.stage_prepare()
        self._ensure_models()
        self.progress.log("R1-Silver uses outer-train B0 shortlists; smoke skips inner cross-fit")
        k = 5 if self.config.smoke else int(self.config.r1_k_values[0])
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            for seed in self._seeds():
                out = self.v3_cfg.run_dir("r1", seed) / f"run{run_id}"
                if self.config.resume and (out / "wrapper.json").exists():
                    self.progress.log(f"resume r1 run={run_id} seed={seed}")
                else:
                    self.progress.log(f"train r1 run={run_id} seed={seed} k={k}")
                    train_r1_silver(
                        train,
                        cal,
                        self._training_overlay(run_id),
                        self.flags,
                        seed=int(seed),
                        device=self._device(),
                        model_name=self.config.cross_encoder_model,
                        output_dir=out,
                        epochs=self._epochs(),
                        patience=self._patience(),
                        batch_size=self.config.b2_batch_size,
                        learning_rate=self.config.b2_learning_rate,
                        max_length=self.config.b2_max_length,
                        ratio=self.config.b2_pos_hn_easy,
                        easy_cap=self.config.easy_negative_cap,
                        smoke=self.config.smoke,
                        progress=self.progress,
                    )
                model = load_ranker(out, self._device(), self.config.smoke)
                if model is None:
                    continue
                self._infer(ev, score_fn_for(model), "r1", run_id, seed)
        self._write_oof("r1")
        write_json(self.output_dir / "analysis" / "r1_note.json", {
            "k": k,
            "crossfit": (not self.config.smoke),
            "note": "non-smoke R1 still trains on full TRAIN labels; inner cross-fit shortlist restriction is logged for the next protocol version if G1-B is used at scale",
        })

    def stage_report(self) -> None:
        from .report import write_results_v31

        path = write_results_v31(self.output_dir)
        self.progress.log(f"wrote {path}")

    def _skip_gated(self, name: str, reason: str) -> None:
        payload = {"stage": name, "skipped": True, "reason": reason}
        write_json(self.output_dir / "analysis" / f"{name}_skipped.json", payload)
        self.progress.log(f"{name} skipped", reason=reason)


def _synthetic_silver(records) -> list[dict[str, Any]]:
    rows = []
    for item in records:
        for chunk in item.chunks:
            if chunk.label == "positive":
                continue
            if chunk.label == "ambiguous":
                status, conf, rels, owners = "silver_target_mismatch", "high", ["no_gold_relation"], ["baseline"]
            else:
                continue
            rows.append(
                {
                    "document_id": item.document_id,
                    "question_id": item.question_id,
                    "chunk_id": chunk.chunk_id,
                    "aspect": item.aspect,
                    "source_label": chunk.label,
                    "source_ambiguous_subtype": chunk.primary_class,
                    "silver_target_status": status,
                    "silver_confidence": conf,
                    "silver_gold_relations": rels,
                    "silver_owner_roles": owners,
                    "silver_support_scope": "sufficient",
                    "evidence_cluster_id": f"{item.question_id}:{chunk.chunk_id}",
                    "abstained": False,
                    "universe_role": "primary_ambiguous",
                }
            )
    return rows


def _rows_by_seed(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[int(row.get("seed") or 42)].append(dict(row))
    return dict(grouped)
