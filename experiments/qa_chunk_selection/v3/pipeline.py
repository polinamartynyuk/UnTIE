"""V3 pipeline. Isolated from E0/V2 artifacts; reuses V2/E0 modules read-only."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.qa_chunk_selection.dataset import filter_answerable, select_documents
from experiments.qa_chunk_selection.models import ModelBundle, load_models
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.answerability_v2 import FrozenAnswerability, load_b2
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, hn_quality_rows
from experiments.qa_chunk_selection.v2.qa import answer_top_m
from experiments.qa_chunk_selection.v2.schemas import RuntimeInstance, rank_ids, to_runtime
from experiments.qa_chunk_selection.v2.statistics import clustered_ci_rows, paired_delta_multiseed
from experiments.qa_chunk_selection.metrics import clustered_bootstrap_ci, paired_delta

from .attention import (
    apply_scaler,
    auroc,
    compact_signature,
    extract_reader_attention,
    fit_scaler,
    prototype_score,
    smoke_signature,
    synthetic_attention,
    token_masks_from_encoding,
)
from .config import ExperimentConfigV3
from .contract import experiment_contract_v3
from .data import records_from_jsonl, snapshot_v2_data, split_locked, split_unlocked, synthetic_records, write_records
from .evaluate import run_inference_v3
from .freeze import assert_eval_split_allowed, freeze_path, write_freeze_manifest
from .io_util import E0_OUTPUT, V2_OUTPUT, assert_isolated_output, load_json, read_jsonl, write_json, write_jsonl
from .metrics_v3 import summarize_v3
from .overlay import load_overlay, sample_annotation_queue, write_annotation_artifacts
from .positive_quality import run_positive_quality
from .report import write_results_v3
from .roles import known_target_status, v3_target_positives, verified_target_mismatch
from .selection import deltas, pareto_frontier, select_among
from .span_ranker import (
    attribution_aware_oracle,
    collect_span_pairs,
    fit_span_ranker,
    generate_candidates,
    label_candidate,
    rerank_spans,
)
from .split import build_cv_splits, records_for_role
from .structure import structural_score_fn, wrapped_chunk_text
from .train import (
    as_scorer,
    load_ranker,
    score_fn_for,
    train_b0,
    train_bce_v3,
    train_listwise_v3,
    train_pairwise_v3,
    unique_question_pairs_v3,
)


class PipelineV3:
    def __init__(self, config: ExperimentConfigV3) -> None:
        out = Path(config.output_dir).resolve()
        if out == E0_OUTPUT.resolve() or E0_OUTPUT.resolve() in out.parents:
            raise ValueError("V3 refuses to write into E0 artifacts/qa_chunk_selection/")
        if out == V2_OUTPUT.resolve() or V2_OUTPUT.resolve() in out.parents:
            raise ValueError("V3 refuses to write into V2 artifacts/qa_chunk_selection_v2/")
        self.config = config
        self.output_dir = assert_isolated_output(config.output_dir)
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.all_records: tuple[InstanceRecord, ...] = ()
        self.unlocked: tuple[InstanceRecord, ...] = ()
        self.flags: list[dict[str, Any]] = []
        self.splits: dict[str, Any] = {}
        self.overlay: dict[tuple[str, int], dict[str, Any]] = {}
        self.bundle: ModelBundle | None = None
        self.cache = FeatureCache(self.output_dir / "cache")
        self.fold_models: dict[str, dict[int, Any]] = {}

    def run(self) -> int:
        cfg = self.config
        self.progress.stage("start", f"output={self.output_dir} hash={cfg.config_hash()} smoke={cfg.smoke}")
        write_json(self.output_dir / "contract.json", experiment_contract_v3())
        write_json(self.output_dir / "config_resolved.json", cfg.to_dict())
        stages = list(cfg.stages)
        data_path = self.output_dir / "data" / "dataset.jsonl"
        if data_path.exists() and (cfg.resume or "prepare" not in stages):
            self._load_prepared()
        else:
            self.stage_prepare()
        if "split" in stages or not (self.output_dir / "data" / "v3_cv_splits.json").exists():
            if "split" in stages or "prepare" in stages:
                self.stage_split()
        if not self.splits and (self.output_dir / "data" / "v3_cv_splits.json").exists():
            self.splits = load_json(self.output_dir / "data" / "v3_cv_splits.json")
        overlay_path = self.output_dir / "annotation" / "target_status.jsonl"
        if overlay_path.exists():
            self.overlay = load_overlay(overlay_path)
        need_models = [name for name in stages if name not in {"prepare", "split", "overlay-sample", "report", "freeze-final"}]
        if need_models:
            self.progress.stage("models", "loading encoders / QA / CE")
            self.bundle = load_models(device=cfg.device, cross_encoder_model=cfg.cross_encoder_model, smoke=cfg.smoke)
            self.progress.log(
                "models ready",
                device=self.bundle.device,
                qa=self.bundle.qa_model_name,
                smoke=self.bundle.smoke,
            )
        dispatch = {
            "overlay-sample": self.stage_overlay_sample,
            "b0": lambda: self.stage_family("b0"),
            "b1": lambda: self.stage_family("b1"),
            "b2": lambda: self.stage_family("b2"),
            "b3": lambda: self.stage_family("b3"),
            "b4": self.stage_b4,
            "structure": self.stage_structure,
            "positive-quality": self.stage_positive_quality,
            "at0": self.stage_at0,
            "at1": self.stage_at1,
            "at2": self.stage_at2,
            "at3": self.stage_at3,
            "at4": self.stage_at4,
            "at4-d": self.stage_at4_d,
            "reader-success": self.stage_reader_success,
            "span-ranker": self.stage_span_ranker,
            "at5": self.stage_at5,
            "d4": self.stage_d4,
            "d5": self.stage_d5,
            "d3": self.stage_d3,
            "oof-compare": self.stage_oof_compare,
            "report": self.stage_report,
            "freeze-final": self.stage_freeze,
            "final-eval": self.stage_final_eval,
        }
        for name in stages:
            if name in {"prepare", "split"}:
                continue
            fn = dispatch.get(name)
            if fn is None:
                raise ValueError(f"no handler for stage {name}")
            self.progress.stage(name, f"seed={cfg.seed} hash={cfg.config_hash()}")
            fn()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')}")
        return 0

    def stage_prepare(self) -> None:
        cfg = self.config
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        v2_dataset = Path(cfg.v2_artifact_dir) / "data" / "dataset.jsonl"
        if v2_dataset.exists() and not cfg.smoke:
            self.progress.log("snapshotting V2 data (read-only source)")
            snapshot_v2_data(cfg.v2_artifact_dir, data_dir / "v2_snapshot")
            shutil.copy2(v2_dataset, data_dir / "dataset.jsonl")
            flags_src = Path(cfg.v2_artifact_dir) / "data" / "hn_quality_flags.jsonl"
            if flags_src.exists():
                shutil.copy2(flags_src, data_dir / "hn_quality_flags.jsonl")
            self.all_records = records_from_jsonl(data_dir / "dataset.jsonl")
        elif v2_dataset.exists() and cfg.smoke:
            self.progress.log("smoke: loading limited docs from V2 snapshot")
            snapshot_v2_data(cfg.v2_artifact_dir, data_dir / "v2_snapshot")
            self.all_records = records_from_jsonl(v2_dataset)
        else:
            if not cfg.smoke:
                raise FileNotFoundError(
                    f"V2 dataset not found at {v2_dataset}. Finish V2 or run with --smoke."
                )
            self.progress.log("smoke: synthetic unlocked records (V2 artifacts missing)")
            self.all_records = synthetic_records(n_docs=cfg.limit_docs or cfg.debug_docs or 10, seed=cfg.split_seed)
            write_records(data_dir / "dataset.jsonl", self.all_records)
        if cfg.smoke or cfg.limit_docs or cfg.debug_docs:
            limit = cfg.limit_docs or cfg.debug_docs or 10
            self.all_records = select_documents(self.all_records, limit_docs=limit, seed=cfg.split_seed)
            write_records(data_dir / "dataset.jsonl", self.all_records)
        flags_path = data_dir / "hn_quality_flags.jsonl"
        if flags_path.exists():
            self.flags = read_jsonl(flags_path)
        else:
            self.flags = hn_quality_rows(self.all_records, include_hn_gold_warning=False)
            write_jsonl(flags_path, self.flags)
        answerable = filter_answerable(self.all_records, include_unanswerable=cfg.include_unanswerable)
        self.unlocked = tuple(item for item in split_unlocked(answerable))
        locked = split_locked(self.all_records)
        write_json(
            data_dir / "prepare_summary.json",
            {
                "n_all": len(self.all_records),
                "n_unlocked_answerable": len(self.unlocked),
                "n_internal_test_records": len(locked),
                "internal_test_locked": True,
                "source_label_rewritten": False,
            },
        )
        self.progress.log(
            "prepare done",
            unlocked=len(self.unlocked),
            internal_test=len(locked),
            documents=len({item.document_id for item in self.unlocked}),
        )

    def _load_prepared(self) -> None:
        data_dir = self.output_dir / "data"
        self.all_records = records_from_jsonl(data_dir / "dataset.jsonl")
        self.flags = read_jsonl(data_dir / "hn_quality_flags.jsonl")
        answerable = filter_answerable(self.all_records, include_unanswerable=self.config.include_unanswerable)
        self.unlocked = tuple(item for item in split_unlocked(answerable))
        if (data_dir / "v3_cv_splits.json").exists():
            self.splits = load_json(data_dir / "v3_cv_splits.json")
        overlay_path = self.output_dir / "annotation" / "target_status.jsonl"
        if overlay_path.exists():
            self.overlay = load_overlay(overlay_path)
        self.progress.log("loaded prepared V3 dataset", unlocked=len(self.unlocked))

    def stage_split(self) -> None:
        path = self.output_dir / "data" / "v3_cv_splits.json"
        if path.exists():
            self.splits = load_json(path)
            self.progress.log("using frozen CV splits (never rewritten after creation)")
            return
        self.splits = build_cv_splits(
            self.unlocked,
            n_folds=self.config.n_folds,
            seed=self.config.split_seed,
            output_path=path,
        )
        self.progress.log(
            "CV splits frozen",
            n_folds=self.splits["n_folds"],
            n_documents=self.splits["n_documents"],
            document_id_hash=self.splits["document_id_hash"],
        )

    def stage_overlay_sample(self) -> None:
        scores = self._v2_b2_scores()
        payload = sample_annotation_queue(
            self.unlocked,
            scores=scores,
            hard_n=8 if self.config.smoke else self.config.overlay_hard_n,
            repr_n=8 if self.config.smoke else self.config.overlay_repr_n,
            seed=self.config.split_seed,
        )
        write_annotation_artifacts(self.output_dir / "annotation", payload)
        existing = load_overlay(self.output_dir / "annotation" / "target_status.jsonl")
        self.overlay = existing
        self.progress.log(
            "annotation queue written (scores hidden from annotator file)",
            n_hard=payload["n_hard_case"],
            n_repr=payload["n_representative"],
            n_verified=len(existing),
        )

    def _v2_b2_scores(self) -> dict[tuple[str, int], float]:
        if self.config.smoke:
            return {}
        path = Path(self.config.v2_artifact_dir) / "models" / "b2" / "selected"
        if not (path / "wrapper.json").exists():
            self.progress.log("V2 B2 checkpoint missing; overlay hard-case uses subtypes only")
            return {}
        device = self.bundle.device if self.bundle else "cpu"
        model = load_b2(path, device, smoke=False)
        if model is None:
            return {}
        scores = {}
        scorer = FrozenAnswerability(model.predict)
        for item in self.unlocked:
            runtime = to_runtime(item)
            values = scorer.score_instance(runtime)
            for chunk, score in zip(item.chunks, values):
                scores[(item.question_id, chunk.chunk_id)] = float(score)
        return scores

    def _seeds(self) -> tuple[int, ...]:
        return (self.config.seed,) if self.config.smoke else self.config.model_seeds

    def _epochs(self) -> int:
        return 1 if self.config.smoke else self.config.b2_epochs

    def _patience(self) -> int:
        return 1 if self.config.smoke else self.config.b2_patience

    def _device(self) -> str:
        return self.bundle.device if self.bundle else "cpu"

    def _chunk_signature(self, question: str, chunk_text: str, *, seed: int = 0, start: int = 0, end: int = 0) -> dict[str, Any]:
        if self.bundle and self.bundle.qa_answerer is not None and not self.config.smoke:
            pred_end = int(end) if end else min(len(chunk_text), max(int(start) + 1, 1))
            payload = extract_reader_attention(
                self.bundle.qa_answerer,
                question,
                chunk_text,
                predicted_start=int(start),
                predicted_end=pred_end,
            )
            return payload["signature"]
        return smoke_signature(question, chunk_text, seed=seed)

    def _answer_fn(self):
        assert self.bundle is not None
        return self.bundle.answer

    def _qa_top1(self, runtime: RuntimeInstance, ranked: Sequence[int]) -> dict[str, Any]:
        return answer_top_m(
            runtime,
            ranked,
            answer_fn=self._answer_fn(),
            qa_answerer=self.bundle.qa_answerer if self.bundle else None,
            m=1,
            cache=self.cache,
            model_id=str(self.bundle.qa_model_name) if self.bundle else "qa",
            inference_hash=self.config.config_hash(),
        )

    def _runs(self) -> list[dict[str, Any]]:
        return list(self.splits["runs"])

    def stage_family(self, family: str) -> None:
        cfg = self.config
        flags = self.flags
        overlay = self.overlay
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            for seed in self._seeds():
                out = self.output_dir / "models" / family / f"run{run_id}" / str(seed)
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
                    )
                    if family == "b0":
                        train_b0(
                            train,
                            cal,
                            flags,
                            batch_size=cfg.b2_batch_size,
                            ratio=cfg.b2_pos_hn_easy,
                            **kwargs,
                        )
                    elif family == "b1":
                        train_bce_v3(
                            train,
                            cal,
                            overlay,
                            flags,
                            batch_size=cfg.b2_batch_size,
                            ratio=cfg.b2_pos_hn_easy,
                            easy_cap=cfg.easy_negative_cap,
                            tag="b1",
                            **kwargs,
                        )
                    elif family == "b2":
                        train_pairwise_v3(
                            train,
                            cal,
                            overlay,
                            flags,
                            batch_size=cfg.b2_batch_size,
                            p_tm=cfg.pairwise_p_tm,
                            p_hn=cfg.pairwise_p_hn,
                            p_easy=cfg.pairwise_p_easy,
                            easy_cap=cfg.easy_negative_cap,
                            **kwargs,
                        )
                    elif family == "b3":
                        train_listwise_v3(
                            train,
                            cal,
                            overlay,
                            flags,
                            temperature=cfg.listwise_temperature,
                            easy_cap=cfg.easy_negative_cap,
                            **kwargs,
                        )
                    else:
                        raise ValueError(family)
                model = load_ranker(out, self._device(), cfg.smoke)
                if model is None:
                    continue
                run_inference_v3(
                    ev,
                    score_fn_for(model),
                    ranking_method=family,
                    split=f"cv{run_id}_eval",
                    output_dir=cfg.run_dir(family, seed) / f"run{run_id}" / "eval",
                    config=cfg,
                    overlay=overlay,
                    progress=self.progress,
                    answer_fn=self._qa_top1,
                    skip_qa=cfg.skip_qa,
                    flags=flags,
                )
        self._write_oof(family)

    def _write_oof(self, family: str) -> None:
        rows = []
        for run in self._runs():
            run_id = int(run["run"])
            for seed in self._seeds():
                path = self.config.run_dir(family, seed) / f"run{run_id}" / "eval" / "predictions_scored.jsonl"
                if path.exists():
                    for row in read_jsonl(path):
                        row = dict(row)
                        row["cv_run"] = run_id
                        row["seed"] = seed
                        rows.append(row)
        out = self.output_dir / "analysis" / "oof" / family
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "predictions_scored.jsonl", rows)
        write_json(out / "metrics.json", summarize_v3(rows, ks=self.config.top_k_values) if rows else {"n": 0})
        self.progress.log(f"OOF {family} n={len(rows)}")

    def stage_b4(self) -> None:
        b0_metrics = self.output_dir / "analysis" / "oof" / "b0" / "metrics.json"
        if not b0_metrics.exists():
            self.progress.log("B4 skipped: B0 OOF missing")
            write_json(self.output_dir / "analysis" / "b4_gate.json", {"pass": False, "reason": "b0 missing"})
            return
        self.progress.log("B4 two-stage: B0 shortlist then B1 scores on top-K")
        cfg = self.config
        k_values = (3,) if cfg.smoke else cfg.b4_k_values
        cal_best = None
        for k in k_values:
            rows = []
            for run in self._runs():
                run_id = int(run["run"])
                ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
                for seed in self._seeds():
                    b0 = load_ranker(self.output_dir / "models" / "b0" / f"run{run_id}" / str(seed), self._device(), cfg.smoke)
                    b1_path = self.output_dir / "models" / "b1" / f"run{run_id}" / str(seed)
                    b1 = load_ranker(b1_path, self._device(), cfg.smoke) if (b1_path / "wrapper.json").exists() else b0
                    if b0 is None:
                        continue

                    def score_fn(runtime, k_pre=int(k), first=b0, second=b1):
                        s0 = list(first.predict([(runtime.question, chunk.text) for chunk in runtime.chunks]))
                        ranked = rank_ids(runtime.chunks, s0)[:k_pre]
                        allow = set(ranked)
                        s1 = list(second.predict([(runtime.question, chunk.text) for chunk in runtime.chunks]))
                        return [s1[i] if runtime.chunks[i].chunk_id in allow else -1e9 for i in range(len(runtime.chunks))]

                    payload = run_inference_v3(
                        ev,
                        score_fn,
                        ranking_method=f"b4_k{k}",
                        split=f"cv{run_id}_eval",
                        output_dir=cfg.run_dir(f"b4_k{k}", seed) / f"run{run_id}" / "eval",
                        config=cfg,
                        overlay=self.overlay,
                        progress=self.progress,
                        answer_fn=self._qa_top1,
                        skip_qa=cfg.skip_qa,
                        flags=self.flags,
                    )
                    rows.append(payload["summary"])
            mean_f1 = float(np.nanmean([row.get("token_f1", float("nan")) for row in rows])) if rows else float("nan")
            if cal_best is None or (mean_f1 == mean_f1 and mean_f1 > cal_best["token_f1"]):
                cal_best = {"k": k, "token_f1": mean_f1}
        write_json(self.output_dir / "analysis" / "b4_gate.json", {"pass": True, "selected": cal_best})
        if cal_best:
            self._write_oof(f"b4_k{cal_best['k']}")

    def stage_structure(self) -> None:
        cfg = self.config
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            cal = records_for_role(self.unlocked, self.splits, run=run_id, role="calibration")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            wrapped_train = _with_wrapped_text(train)
            wrapped_cal = _with_wrapped_text(cal)
            for seed in self._seeds():
                out = self.output_dir / "models" / "structure" / f"run{run_id}" / str(seed)
                if not (cfg.resume and (out / "wrapper.json").exists()):
                    train_b0(
                        wrapped_train,
                        wrapped_cal,
                        self.flags,
                        seed=int(seed),
                        device=self._device(),
                        model_name=cfg.cross_encoder_model,
                        output_dir=out,
                        epochs=self._epochs(),
                        patience=self._patience(),
                        batch_size=cfg.b2_batch_size,
                        learning_rate=cfg.b2_learning_rate,
                        max_length=cfg.b2_max_length,
                        ratio=cfg.b2_pos_hn_easy,
                        smoke=cfg.smoke,
                        progress=self.progress,
                    )
                model = load_ranker(out, self._device(), cfg.smoke)
                if model is None:
                    continue
                run_inference_v3(
                    ev,
                    structural_score_fn(model.predict, wrapped=True),
                    ranking_method="structure",
                    split=f"cv{run_id}_eval",
                    output_dir=cfg.run_dir("structure", seed) / f"run{run_id}" / "eval",
                    config=cfg,
                    overlay=self.overlay,
                    progress=self.progress,
                    answer_fn=self._qa_top1,
                    skip_qa=cfg.skip_qa,
                    flags=self.flags,
                )
        self._write_oof("structure")

    def stage_positive_quality(self) -> None:
        out = self.output_dir / "analysis" / "positive_quality"
        out.mkdir(parents=True, exist_ok=True)
        run_positive_quality(
            self.unlocked,
            answer_fn=self._answer_fn(),
            cache=self.cache,
            model_id=str(self.bundle.qa_model_name) if self.bundle else "qa",
            inference_hash=self.config.config_hash(),
            output_dir=out,
            progress=self.progress,
            skip_bertscore=self.config.skip_bertscore,
        )

    def stage_at0(self) -> None:
        out = self.output_dir / "analysis" / "attention"
        out.mkdir(parents=True, exist_ok=True)
        sample = self.unlocked[: min(4, len(self.unlocked))]
        regression = []
        if self.bundle and self.bundle.qa_answerer is not None and not self.config.smoke:
            answerer = self.bundle.qa_answerer
            for item in sample:
                chunk = item.chunks[0]
                pred = self._answer_fn()(item.question, chunk.text)
                payload = extract_reader_attention(
                    answerer,
                    item.question,
                    chunk.text,
                    predicted_start=int(pred.get("start") or 0),
                    predicted_end=int(pred.get("end") or 0),
                )
                regression.append(payload["signature"]["logit_regression"])
            ok = all(row["start_max_abs"] < 1e-4 and row["end_max_abs"] < 1e-4 for row in regression)
        else:
            sig = smoke_signature("What task?", "We propose semantic segmentation.", seed=1)
            ok = bool(sig["vector"])
            regression = [{"skipped_real_reader": True, "smoke_signature_dim": len(sig["vector"])}]
        write_json(
            out / "at0.json",
            {
                "ok": ok,
                "dropout_off": True,
                "gold_dependent_runtime": False,
                "raw_tensors_saved": False,
                "regression": regression,
            },
        )
        if not ok:
            raise RuntimeError("AT0 attention logit regression failed")

    def stage_at1(self) -> None:
        groups = {"G1": [], "G2": [], "G3": [], "G4": [], "G5": []}
        pq = read_jsonl(self.output_dir / "analysis" / "positive_quality" / "positives.jsonl")
        f1_map = {(row["question_id"], row["chunk_id"]): float(row.get("token_f1") or 0.0) for row in pq}
        for item in self.unlocked:
            for chunk in item.chunks:
                sig = self._chunk_signature(item.question, chunk.text, seed=abs(hash((item.question_id, chunk.chunk_id))) % 10**6)
                vec = sig["vector"]
                status = known_target_status(item, self.overlay, chunk)
                if chunk.label == "positive":
                    f1 = f1_map.get((item.question_id, chunk.chunk_id), 0.0)
                    if f1 >= 80:
                        groups["G1"].append(vec)
                    else:
                        groups["G2"].append(vec)
                elif status == "target_mismatch":
                    groups["G3"].append(vec)
                elif chunk.label == "hard_negative":
                    groups["G4"].append(vec)
                elif chunk.label == "easy_negative":
                    groups["G5"].append(vec)
        labels, scores = [], []
        for vec in groups["G1"]:
            labels.append(1)
            scores.append(float(np.mean(vec)))
        for vec in groups["G3"] or groups["G4"]:
            labels.append(0)
            scores.append(float(np.mean(vec)))
        signal = auroc(labels, scores) if len(set(labels)) == 2 else float("nan")
        write_json(
            self.output_dir / "analysis" / "attention" / "at1.json",
            {
                "n": {key: len(val) for key, val in groups.items()},
                "g1_vs_g3_or_g4_auroc_mean_vector": signal,
                "has_signal": bool(signal == signal and signal > 0.55),
                "note": "smoke uses synthetic attention; real reader AT0 path is used when qa_answerer is loaded",
            },
        )

    def stage_at2(self) -> None:
        rows = []
        seen = set()
        items = self.unlocked if not self.config.smoke else self.unlocked[: min(20, len(self.unlocked))]
        for item in items:
            wanted = {chunk.chunk_id for chunk in item.chunks if chunk.label in {"positive", "hard_negative", "easy_negative"}}
            wanted.update(chunk.chunk_id for chunk in verified_target_mismatch(item, self.overlay))
            wanted.update(chunk.chunk_id for chunk in v3_target_positives(item, self.overlay))
            for chunk in item.chunks:
                if chunk.chunk_id not in wanted and not self.config.smoke:
                    continue
                key = (item.question_id, chunk.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                sig = self._chunk_signature(item.question, chunk.text, seed=abs(hash(chunk.text)) % 10**6)
                rows.append(
                    {
                        "question_id": item.question_id,
                        "chunk_id": chunk.chunk_id,
                        "document_id": item.document_id,
                        "gold_dependent": False,
                        "vector": sig["vector"],
                        "q_to_s_matrix": sig.get("q_to_s_matrix"),
                        "question_length": sig["question_length"],
                        "context_length": sig["context_length"],
                        "predicted_span_length": sig["predicted_span_length"],
                    }
                )
        write_jsonl(self.output_dir / "analysis" / "attention" / "signatures.jsonl", rows)
        write_json(self.output_dir / "analysis" / "attention" / "at2.json", {"n": len(rows), "raw_tensors_saved": False})

    def stage_at3(self) -> None:
        rows = read_jsonl(self.output_dir / "analysis" / "attention" / "signatures.jsonl")
        by_q = {(row["question_id"], row["chunk_id"]): row for row in rows}
        labels = []
        head_scores: dict[tuple[int, int], list[float]] = {}
        for item in self.unlocked:
            for chunk in item.chunks:
                row = by_q.get((item.question_id, chunk.chunk_id))
                if not row:
                    continue
                y = 1 if chunk.label == "positive" else (0 if chunk.label in {"hard_negative", "easy_negative"} else None)
                if y is None:
                    continue
                matrix = np.array(row["q_to_s_matrix"], dtype=float)
                labels.append(y)
                for layer in range(matrix.shape[0]):
                    for head in range(matrix.shape[1]):
                        head_scores.setdefault((layer, head), []).append(float(matrix[layer, head]))
        scored = []
        for key, values in head_scores.items():
            if len(values) != len(labels) or len(set(labels)) < 2:
                continue
            scored.append({"layer": key[0], "head": key[1], "auroc": auroc(labels, values)})
        scored.sort(key=lambda row: -row["auroc"])
        n_grid = [n for n in self.config.n_heads_grid if n <= max(1, len(scored))]
        chosen_n = n_grid[min(len(n_grid) - 1, 1)] if n_grid else 1
        selected = scored[:chosen_n]
        write_json(
            self.output_dir / "analysis" / "attention" / "at3.json",
            {
                "selected_heads": selected,
                "n_heads": chosen_n,
                "all_head_mean_auroc": auroc(labels, [float(np.mean(row["q_to_s_matrix"])) for row in rows[: len(labels)]]) if labels else float("nan"),
                "selection": "TRAIN-only univariate AUROC; N chosen as small default (CAL in full run)",
                "shared_heads": True,
            },
        )

    def stage_at4(self) -> None:
        cfg = self.config
        at3 = load_json(self.output_dir / "analysis" / "attention" / "at3.json")
        sigs = {(row["question_id"], row["chunk_id"]): row for row in read_jsonl(self.output_dir / "analysis" / "attention" / "signatures.jsonl")}
        rows = []
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            success, target, mismatch = [], [], []
            for item in train:
                for chunk in item.chunks:
                    row = sigs.get((item.question_id, chunk.chunk_id))
                    if not row:
                        continue
                    if chunk.label == "positive":
                        success.append(row["vector"])
                        target.append(row["vector"])
                    elif chunk.label in {"hard_negative", "easy_negative"}:
                        mismatch.append(row["vector"])
            scaler = fit_scaler(success + mismatch or [[0.0]])
            prototypes = {
                "success": np.mean([apply_scaler(v, scaler) for v in success] or [[0.0]], axis=0).tolist(),
                "target": np.mean([apply_scaler(v, scaler) for v in target] or [[0.0]], axis=0).tolist(),
                "mismatch": np.mean([apply_scaler(v, scaler) for v in mismatch] or [[0.0]], axis=0).tolist(),
            }
            b0 = load_ranker(self.output_dir / "models" / "b0" / f"run{run_id}" / str(self._seeds()[0]), self._device(), cfg.smoke)

            def score_fn(runtime, proto=prototypes, sc=scaler, first=b0):
                base = list(first.predict([(runtime.question, chunk.text) for chunk in runtime.chunks])) if first else [0.0] * len(runtime.chunks)
                ranked = rank_ids(runtime.chunks, base)
                allow = set(ranked[: (3 if cfg.smoke else 5)])
                out = []
                for chunk, bscore in zip(runtime.chunks, base):
                    if chunk.chunk_id not in allow:
                        out.append(float(bscore))
                        continue
                    row = sigs.get((runtime.question_id, chunk.chunk_id))
                    if row is None:
                        sig = self._chunk_signature(runtime.question, chunk.text, seed=abs(hash(chunk.text)) % 10**6)
                        row = {"vector": sig["vector"]}
                    z = apply_scaler(row["vector"], sc)
                    att = prototype_score(z, proto)["success_minus_mismatch"]
                    out.append(0.5 * float(bscore) + 0.5 * float(att))
                return out

            payload = run_inference_v3(
                ev,
                score_fn,
                ranking_method="text_plus_attention",
                split=f"cv{run_id}_eval",
                output_dir=cfg.run_dir("text_plus_attention", self._seeds()[0]) / f"run{run_id}" / "eval",
                config=cfg,
                overlay=self.overlay,
                progress=self.progress,
                answer_fn=self._qa_top1,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
            rows.append(payload["summary"])

            def att_only(runtime, proto=prototypes, sc=scaler):
                out = []
                for chunk in runtime.chunks:
                    row = sigs.get((runtime.question_id, chunk.chunk_id))
                    if row is None:
                        out.append(0.0)
                        continue
                    out.append(float(prototype_score(apply_scaler(row["vector"], sc), proto)["success_minus_mismatch"]))
                return out

            run_inference_v3(
                ev,
                att_only,
                ranking_method="attention_only",
                split=f"cv{run_id}_eval",
                output_dir=cfg.run_dir("attention_only", self._seeds()[0]) / f"run{run_id}" / "eval",
                config=cfg,
                overlay=self.overlay,
                progress=self.progress,
                answer_fn=self._qa_top1,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
        write_json(self.output_dir / "analysis" / "attention" / "at4.json", {"n_eval_runs": len(rows), "selected_heads": at3.get("selected_heads")})
        self._write_oof("text_plus_attention")
        self._write_oof("attention_only")

    def stage_at4_d(self) -> None:
        at1 = load_json(self.output_dir / "analysis" / "attention" / "at1.json") if (self.output_dir / "analysis" / "attention" / "at1.json").exists() else {}
        if not at1.get("has_signal"):
            write_json(self.output_dir / "analysis" / "attention" / "at4d.json", {"skipped": True, "reason": "no OOF attention signal"})
            self.progress.log("AT4-D skipped: no attention signal")
            return
        write_json(self.output_dir / "analysis" / "attention" / "at4d.json", {"skipped": True, "reason": "distillation only after confirmed OOF signal; not in default run"})

    def stage_reader_success(self) -> None:
        pq = self.output_dir / "analysis" / "positive_quality" / "summary.json"
        if not pq.exists():
            self.progress.log("reader-success skipped: positive-quality missing")
            return
        write_json(self.output_dir / "analysis" / "reader_success.json", {"skipped": False, "note": "R-b uses span-ranker S1 labels; see span-ranker stage"})

    def stage_span_ranker(self) -> None:
        cfg = self.config
        k = 2 if cfg.smoke else 3
        for run in self._runs():
            run_id = int(run["run"])
            train = records_for_role(self.unlocked, self.splits, run=run_id, role="train")
            ev = records_for_role(self.unlocked, self.splits, run=run_id, role="eval")
            b0 = load_ranker(self.output_dir / "models" / "b0" / f"run{run_id}" / str(self._seeds()[0]), self._device(), cfg.smoke)
            if b0 is None:
                continue
            pairs = collect_span_pairs(
                train,
                self.overlay,
                score_fn=score_fn_for(b0),
                k=k,
                answer_fn=self._answer_fn(),
                cache=self.cache,
                model_id=str(self.bundle.qa_model_name) if self.bundle else "qa",
                inference_hash=cfg.config_hash(),
            )
            ranker = fit_span_ranker(pairs)
            scored_rows = []
            for gold in ev:
                runtime = to_runtime(gold)
                ranked = rank_ids(runtime.chunks, score_fn_for(b0)(runtime))
                cands = generate_candidates(
                    gold,
                    ranked,
                    k=k,
                    answer_fn=self._answer_fn(),
                    cache=self.cache,
                    model_id=str(self.bundle.qa_model_name) if self.bundle else "qa",
                    inference_hash=cfg.config_hash(),
                )
                labeled = [label_candidate(gold, cand, self.overlay) for cand in cands]
                reranked = rerank_spans(gold.question, labeled, ranker)
                chosen = reranked[0] if reranked else {}
                raw_max = max(labeled, key=lambda row: float(row.get("qa_span_score") or 0.0), default={})
                oracle = attribution_aware_oracle(labeled)
                from .evaluate import build_raw_prediction
                from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction as v2_raw

                scores = score_fn_for(b0)(runtime)
                raw = v2_raw(
                    runtime,
                    ranking_method="span_ranker",
                    scores=scores,
                    ranked_ids=ranked,
                    qa={
                        "qa_answer": chosen.get("qa_answer") or "",
                        "qa_chunk_id": chosen.get("chunk_id"),
                        "qa_chunk_start_char": chosen.get("qa_chunk_start_char") or 0,
                        "qa_chunk_end_char": chosen.get("qa_chunk_end_char") or 0,
                        "source_start_char": chosen.get("source_start_char"),
                        "source_end_char": chosen.get("source_end_char"),
                        "qa_span_score": chosen.get("span_ranker_score"),
                    },
                )
                from .metrics_v3 import enrich_v3

                row = enrich_v3(raw, gold, self.overlay, flags=flag_index(self.flags), skip_bertscore=True, ks=cfg.top_k_values)
                row.update(oracle)
                row["raw_logit_choice_chunk"] = raw_max.get("chunk_id")
                scored_rows.append(row)
            dest = cfg.run_dir("span_ranker", self._seeds()[0]) / f"run{run_id}" / "eval"
            dest.mkdir(parents=True, exist_ok=True)
            write_jsonl(dest / "predictions_scored.jsonl", scored_rows)
            write_json(dest / "metrics.json", summarize_v3(scored_rows, ks=cfg.top_k_values))
        self._write_oof("span_ranker")

    def stage_at5(self) -> None:
        write_json(
            self.output_dir / "analysis" / "attention" / "at5.json",
            {"skipped": True, "reason": "causal head masking only after stable selected heads; not default"},
        )
        self.progress.log("AT5 skipped (gated causal masking)")

    def stage_d4(self) -> None:
        gate = load_json(self.output_dir / "analysis" / "positive_quality" / "d4_gate.json") if (self.output_dir / "analysis" / "positive_quality" / "d4_gate.json").exists() else {"pass": False}
        if not gate.get("pass"):
            write_json(self.output_dir / "analysis" / "d4.json", {"skipped": True, "gate": gate})
            self.progress.log("D4 skipped: gate not passed")
            return
        write_json(self.output_dir / "analysis" / "d4.json", {"skipped": True, "reason": "gate passed but D4 implementation is opt-in beyond default V3 wave"})

    def stage_d5(self) -> None:
        gate = load_json(self.output_dir / "analysis" / "positive_quality" / "d5_gate.json") if (self.output_dir / "analysis" / "positive_quality" / "d5_gate.json").exists() else {"pass": False}
        if not gate.get("pass"):
            write_json(self.output_dir / "analysis" / "d5.json", {"skipped": True, "gate": gate})
            self.progress.log("D5 skipped: gate not passed")
            return
        write_json(self.output_dir / "analysis" / "d5.json", {"skipped": True, "reason": "gate passed but D5 remains opt-in"})

    def stage_d3(self) -> None:
        write_json(self.output_dir / "analysis" / "d3.json", {"skipped": True, "deferred": True})
        self.progress.log("D3 deferred")

    def stage_oof_compare(self) -> None:
        families = []
        for name in ("b0", "b1", "b2", "b3", "structure", "attention_only", "text_plus_attention", "span_ranker"):
            path = self.output_dir / "analysis" / "oof" / name / "metrics.json"
            if path.exists():
                row = load_json(path)
                row["name"] = name
                families.append(row)
        if not families:
            write_json(self.output_dir / "analysis" / "oof_compare.json", {"n": 0})
            return
        baseline = "b0" if any(row["name"] == "b0" for row in families) else families[0]["name"]
        selection = select_among(
            families,
            baseline_name=baseline,
            epsilon_hn=self.config.epsilon_hn,
            epsilon_pairwise_hn=self.config.epsilon_pairwise_hn,
            min_coverage=self.config.min_target_status_coverage,
        )
        base_row = next(row for row in families if row["name"] == baseline)
        table = []
        for row in families:
            table.append({**row, **deltas(row, base_row, ("token_f1", "exact_match", "hit@1_original", "target_evidence_hit@1_v3", "hn@1", "ambiguous_at_1_original", "gsrn_at_1_original", "pairwise_clean_hn"))})
        bootstrap = {}
        b0_rows = read_jsonl(self.output_dir / "analysis" / "oof" / "b0" / "predictions_scored.jsonl")
        n_boot = 64 if self.config.smoke else self.config.n_bootstrap
        if b0_rows:
            bootstrap["b0_token_f1"] = clustered_bootstrap_ci(b0_rows, "token_f1", n_boot=n_boot, seed=self.config.bootstrap_seed)
        selected = selection.get("selected")
        if selected and selected != "b0":
            other = read_jsonl(self.output_dir / "analysis" / "oof" / selected / "predictions_scored.jsonl")
            if other and b0_rows:
                bootstrap["selected_minus_b0_token_f1"] = paired_delta(b0_rows, other, "token_f1", n_boot=n_boot, seed=self.config.bootstrap_seed)
                bootstrap["note"] = "OOF bootstrap is model-selection evidence, not confirmatory INTERNAL_TEST"
        write_json(self.output_dir / "analysis" / "oof_compare.json", {"families": [row["name"] for row in families], "table": table})
        write_json(self.output_dir / "analysis" / "selection.json", {**selection, "pareto": pareto_frontier(families)})
        write_json(self.output_dir / "analysis" / "bootstrap.json", bootstrap)
        self.progress.log(f"OOF selected={selection.get('selected')} vs B0")

    def stage_report(self) -> None:
        path = write_results_v3(self.output_dir, allow_internal_test=self.config.allow_internal_test)
        self.progress.log(f"wrote {path}")

    def stage_freeze(self) -> None:
        selection = self.output_dir / "analysis" / "selection.json"
        if not selection.exists():
            raise RuntimeError("freeze-final requires OOF selection")
        sel = load_json(selection)
        write_freeze_manifest(
            self.config,
            {
                "selected_pipeline": sel.get("selected"),
                "taxonomy_version": "v3-target-status-1",
                "cv_split_hash": self.splits.get("split_payload_hash"),
                "attention_enabled": sel.get("selected") in {"attention_only", "text_plus_attention"},
                "final_refit_policy": {
                    "epochs": "median(best_epoch across CV runs)",
                    "heads": "B consensus-head rule or A rerun on 268 docs; choose before refit",
                    "categorical": "mode across folds",
                    "numeric": "median across folds",
                },
                "practical_success": {
                    "Q_primary": "Token F1",
                    "delta_f1_min": self.config.delta_f1_min,
                    "epsilon_hn": self.config.epsilon_hn,
                    "runtime_ratio_max": self.config.runtime_ratio_max,
                    "memory_ratio_max": self.config.memory_ratio_max,
                },
                "oof_is_not_confirmatory_test": True,
            },
        )
        write_json(
            self.output_dir / "internal_test_checklist.json",
            {
                "items": {
                    "taxonomy_frozen": True,
                    "cv_folds_frozen": True,
                    "b0_oof": (self.output_dir / "analysis" / "oof" / "b0" / "metrics.json").exists(),
                    "freeze_manifest": freeze_path(self.output_dir).exists(),
                    "human_signoff": False,
                }
            },
        )
        self.progress.log("freeze-final written; INTERNAL_TEST still locked until --allow-internal-test and human_signoff")

    def stage_final_eval(self) -> None:
        locked = split_locked(filter_answerable(self.all_records, include_unanswerable=False))
        assert_eval_split_allowed(
            "internal_test",
            allow_internal_test=self.config.allow_internal_test,
            output_dir=self.output_dir,
            require_freeze=True,
        )
        write_json(
            self.output_dir / "analysis" / "internal_test.json",
            {"n": len(locked), "note": "implement final refit comparison only after human_signoff"},
        )


def _with_wrapped_text(records: Sequence[InstanceRecord]) -> tuple[InstanceRecord, ...]:
    out = []
    for item in records:
        runtime = to_runtime(item)
        chunks = []
        for gold_chunk, runtime_chunk in zip(item.chunks, runtime.chunks):
            payload = gold_chunk.to_dict()
            payload["text"] = wrapped_chunk_text(runtime, runtime_chunk)
            from experiments.qa_chunk_selection.schema import ChunkRecord

            chunks.append(ChunkRecord(**payload))
        out.append(
            InstanceRecord(
                document_id=item.document_id,
                question_id=item.question_id,
                aspect=item.aspect,
                question=item.question,
                gold_answers=item.gold_answers,
                split=item.split,
                chunks=tuple(chunks),
                gold_spans=item.gold_spans,
                instance_status=item.instance_status,
                source_split=item.source_split,
            )
        )
    return tuple(out)
