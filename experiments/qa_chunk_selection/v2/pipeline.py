"""V2 pipeline. Isolated from E0 artifacts; reuses E0 dataset/ranking/QA/metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.answerability import AnswerabilityModel, TfidfAnswerabilityScorer
from experiments.qa_chunk_selection.candidates import freeze_candidate_sets
from experiments.qa_chunk_selection.dataset import filter_answerable, load_split_records, select_documents
from experiments.qa_chunk_selection.evaluate import evaluate_ranking, write_run
from experiments.qa_chunk_selection.models import ModelBundle, load_models
from experiments.qa_chunk_selection.pipeline import _score_ce, _score_dense, _score_rrf
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.ranking import cross_encoder_scores, section_scores
from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.structure import structural_mix

from .adaptive_m import calibration_grid, choose_m
from .ambiguous_audit import run_ambiguous_audit
from .answer_aggregation_v2 import aggregate_d5
from .answerability_v2 import FrozenAnswerability, evaluate_answerability, load_b2, train_b2_variant
from .cache import FeatureCache
from .config import ExperimentConfigV2
from .contract import experiment_contract_v2
from .d2_ablation import D2_VARIANTS, d2_scores
from .evaluate_v2 import run_inference
from .freeze import assert_eval_split_allowed, load_freeze_manifest, verify_freeze_hash, write_freeze_manifest
from .hn_flags import flag_index, hn_quality_rows
from .io_util import DEFAULT_V2_OUTPUT, E0_OUTPUT, assert_isolated_output, load_json, read_jsonl, write_json, write_jsonl
from .qa import answer_top_m
from .reader_aware import r3_scores, r3_weight_grid
from .reader_oracle import run_reader_oracle
from .regression import regression_r1, regression_r2_compare, write_regression_report
from .report import write_results_v2
from .schemas import RuntimeInstance, to_runtime
from .sentence_evidence import SENTENCE_SPLITTER_VERSION, assemble_d4b, d4_training_examples, project_assembled_span, sentence_units
from .signals import dense_scores, relevance_scores
from .split import apply_document_roles, split_document_ids, split_sanity
from .statistics import mean_std, paired_delta_multiseed
from .unanswerable_audit import run_unanswerable_audit


class PipelineV2:
    def __init__(self, config: ExperimentConfigV2) -> None:
        if Path(config.output_dir).resolve() == E0_OUTPUT.resolve() or E0_OUTPUT.resolve() in Path(config.output_dir).resolve().parents:
            raise ValueError("V2 refuses to write into E0 artifacts/qa_chunk_selection/")
        self.config = config
        self.output_dir = assert_isolated_output(config.output_dir)
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.records: dict[str, tuple[InstanceRecord, ...]] = {}
        self.all_records: tuple[InstanceRecord, ...] = ()
        self.flags: list[dict[str, Any]] = []
        self.bundle: ModelBundle | None = None
        self.cache = FeatureCache(self.output_dir / "cache")
        self.b2 = None
        self.b2_meta: dict[str, Any] = {}
        self.r3_meta: dict[str, Any] = {}
        self.c_meta: dict[str, Any] = {}
        self.adaptive_meta: dict[str, Any] = {}
        self.d2_meta: dict[str, Any] = {}

    def run(self) -> int:
        cfg = self.config
        self.progress.stage("start", f"output={self.output_dir} hash={cfg.config_hash()} smoke={cfg.smoke}")
        write_json(self.output_dir / "contract.json", experiment_contract_v2())
        write_json(self.output_dir / "config_resolved.json", cfg.to_dict())
        stages = list(cfg.stages)
        dataset_path = self.output_dir / "data" / "dataset.jsonl"
        if dataset_path.exists() and (cfg.resume or ("prepare-v2" not in stages and "split-v2" not in stages)):
            self._load_prepared()
        else:
            self.prepare()
        need_models = set(stages) - {"prepare-v2", "split-v2", "e0-regression", "compare-v2", "report-v2", "freeze-final"}
        if need_models:
            self.progress.stage("models", "loading encoders / QA / CE")
            self.bundle = load_models(device=cfg.device, cross_encoder_model=cfg.cross_encoder_model, smoke=cfg.smoke)
            self.progress.log(
                "models ready",
                device=self.bundle.device,
                qa=self.bundle.qa_model_name,
                ce=self.bundle.ce_model_name,
                encoder=self.bundle.encoder_name,
                smoke=self.bundle.smoke,
            )
        dispatch = {
            "e0-regression": self.stage_e0_regression,
            "reader-oracle": self.stage_reader_oracle,
            "ambiguous-audit": lambda: self.stage_ambiguous_audit("e0"),
            "b2-train": self.stage_b2_train,
            "b2-eval": self.stage_b2_eval,
            "ambiguous-audit-b2": lambda: self.stage_ambiguous_audit("b2"),
            "reader-aware": self.stage_reader_aware,
            "c-sweep": self.stage_c_sweep,
            "c-adaptive": self.stage_c_adaptive,
            "d2-ablate": self.stage_d2,
            "compare-v2": self.stage_compare,
            "report-v2": self.stage_report,
            "freeze-final": self.stage_freeze,
            "final-eval": self.stage_final_eval,
            "d4-train": self.stage_d4_train,
            "d4": self.stage_d4,
            "d5": self.stage_d5,
            "unanswerable-audit": self.stage_unanswerable,
        }
        for name in stages:
            if name in {"prepare-v2", "split-v2"}:
                continue
            fn = dispatch.get(name)
            if fn is None:
                raise ValueError(f"no handler for stage {name}")
            self.progress.stage(name, f"seed={cfg.seed} hash={cfg.config_hash()}")
            fn()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')}")
        return 0

    def prepare(self) -> None:
        cfg = self.config
        self.progress.stage("prepare-v2", "loading official Q5 labels (unfiltered)")
        train_all = load_split_records(cfg.train_label_dir, cfg.train_preprocess_dir, split_name="train", progress=self.progress)
        dev_all = load_split_records(cfg.dev_label_dir, cfg.dev_preprocess_dir, split_name="dev", progress=self.progress)
        if cfg.smoke or cfg.limit_docs or cfg.debug_docs:
            limit = cfg.limit_docs or cfg.debug_docs or 8
            train_all = select_documents(train_all, limit_docs=limit, seed=cfg.split_seed)
            dev_all = select_documents(dev_all, limit_docs=max(1, limit // 2), seed=cfg.split_seed)
            self.progress.log(f"smoke/limit docs train={len({r.document_id for r in train_all})} dev={len({r.document_id for r in dev_all})}")
        self.progress.stage("split-v2", "document-level TRAIN/CALIBRATION/DEVELOPMENT/INTERNAL_TEST")
        roles = split_document_ids(
            train_all,
            seed=cfg.split_seed,
            fractions=(cfg.train_fraction, cfg.calibration_fraction, cfg.development_fraction, cfg.internal_test_fraction),
        )
        assigned = apply_document_roles(train_all, roles, official_dev=dev_all)
        summary = split_sanity(assigned)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        write_json(data_dir / "splits_v2.json", roles)
        write_json(data_dir / "data_summary.json", summary)
        write_jsonl(data_dir / "dataset.jsonl", [item.to_dict() for item in assigned])
        self.flags = hn_quality_rows(assigned, include_hn_gold_warning=cfg.include_hn_gold_warning)
        write_jsonl(data_dir / "hn_quality_flags.jsonl", self.flags)
        candidates = freeze_candidate_sets(assigned, mode=cfg.candidate_mode, n_easy=5, seed=cfg.seed)
        write_jsonl(data_dir / "candidate_sets.jsonl", candidates)
        self.all_records = assigned
        answerable = filter_answerable(assigned, include_unanswerable=cfg.include_unanswerable)
        self.records = _by_split(answerable)
        self.progress.log(
            "prepare-v2 done",
            train=len(self.records.get("train", ())),
            calibration=len(self.records.get("calibration", ())),
            development=len(self.records.get("development", ())),
            internal_test=len(self.records.get("internal_test", ())),
            seen_exploratory_dev=len(self.records.get("seen_exploratory_dev", ())),
            hn_warnings=sum(1 for row in self.flags if row.get("hn_gold_string_warning")),
        )

    def _load_prepared(self) -> None:
        rows = []
        for raw in (self.output_dir / "data" / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            payload = json.loads(raw)
            chunks = tuple(ChunkRecord(**item) for item in payload["chunks"])
            spans = tuple(GoldSpan(**item) for item in payload.get("gold_spans") or ())
            rows.append(
                InstanceRecord(
                    document_id=payload["document_id"],
                    question_id=payload["question_id"],
                    aspect=payload["aspect"],
                    question=payload["question"],
                    gold_answers=tuple(payload["gold_answers"]),
                    split=payload["split"],
                    chunks=chunks,
                    gold_spans=spans,
                    instance_status=payload.get("instance_status", "ok"),
                    source_split=payload.get("source_split", payload["split"]),
                )
            )
        self.all_records = tuple(rows)
        self.flags = read_jsonl(self.output_dir / "data" / "hn_quality_flags.jsonl")
        answerable = filter_answerable(rows, include_unanswerable=self.config.include_unanswerable)
        self.records = _by_split(answerable)
        self.progress.log("loaded prepared V2 dataset", **{key: len(val) for key, val in self.records.items()})

    def _eval_splits(self) -> tuple[str, ...]:
        return ("train", "calibration", "development", "seen_exploratory_dev")

    def _model_id(self) -> str:
        assert self.bundle is not None
        return str(self.bundle.qa_model_name)

    def _inf_hash(self) -> str:
        return self.config.config_hash()

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
            model_id=self._model_id(),
            inference_hash=self._inf_hash(),
        )

    def _r3_score_fn(
        self,
        *,
        weights: tuple[float, ...],
        k_pre: int,
        fusion: str,
        mode: str,
        a_model=None,
        b=None,
    ):
        b = b or self._b_scorer()
        a_model = a_model if a_model is not None else (self.bundle.relevance_reranker if self.bundle else None)
        cfg = self.config

        def score_fn(runtime: RuntimeInstance, w=tuple(weights), k=int(k_pre), fus=fusion, md=mode):
            b_scores = b.score_instance(runtime)
            a_scores = relevance_scores(runtime, a_model) if a_model is not None else None
            scores, _meta = r3_scores(
                runtime,
                mode=md,
                k_pre=k if md != "full" else max(len(runtime.chunks), 1),
                weights=w,  # type: ignore[arg-type]
                fusion=fus,
                b_scores=b_scores,
                a_scores=a_scores,
                answer_fn=self._answer_fn(),
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash=self._inf_hash(),
                rrf_k=cfg.rrf_k,
            )
            return scores

        return score_fn

    def _load_selection_artifacts(self) -> None:
        if not self.b2_meta:
            payload = _maybe_json(self.output_dir / "models" / "b2" / "selection.json")
            selected = payload.get("selected") if isinstance(payload, dict) else {}
            self.b2_meta = selected or {}
        if self.b2 is None:
            selected_dir = self.output_dir / "models" / "b2" / "selected"
            if selected_dir.exists():
                self.b2 = load_b2(selected_dir, self.bundle.device if self.bundle else "cpu", self.config.smoke)
        if not self.r3_meta:
            payload = _maybe_json(self.output_dir / "analysis" / "reader_aware" / "summary.json")
            self.r3_meta = (payload.get("selected") if isinstance(payload, dict) else {}) or {}
        if not self.c_meta:
            self.c_meta = _maybe_json(self.output_dir / "analysis" / "c_sweep" / "summary.json") or {}
        if not self.adaptive_meta:
            payload = _maybe_json(self.output_dir / "analysis" / "c_adaptive" / "summary.json")
            self.adaptive_meta = (payload.get("selected") if isinstance(payload, dict) else {}) or {}
        if not self.d2_meta:
            self.d2_meta = _maybe_json(self.output_dir / "analysis" / "d2_ablation" / "summary.json") or {}

    def _b_scorer(self):
        if self.b2 is None:
            selected = self.output_dir / "models" / "b2" / "selected"
            if selected.exists():
                self.b2 = load_b2(selected, self.bundle.device if self.bundle else "cpu", self.config.smoke)
        if self.b2 is None:
            raise RuntimeError("B2 model is not trained/loaded")
        return FrozenAnswerability(self.b2.predict)

    def _e0_b(self):
        path = self.config.e0_artifact_dir / "models" / "answerability_ce"
        if not (path / "wrapper.json").exists():
            return None
        if self.config.smoke:
            return None
        assert self.bundle is not None
        try:
            return AnswerabilityModel.load(path, device=self.bundle.device)
        except Exception:
            return None

    def stage_e0_regression(self) -> None:
        e0 = Path(self.config.e0_artifact_dir)
        out = self.output_dir / "regression"
        out.mkdir(parents=True, exist_ok=True)
        r1 = regression_r1(e0, atol=self.config.regression_atol, rtol=self.config.regression_rtol)
        self.progress.log(f"E0-R1 ok={r1['ok']} mismatches={len(r1['mismatches'])}")
        r2: dict[str, Any] = {"level": "E0-R2", "ok": True, "skipped": True}
        if not r1["ok"] and not self.config.smoke:
            write_regression_report(out / "e0_regression.json", {"r1": r1, "r2": r2})
            raise RuntimeError(f"E0-R1 regression failed: {r1['mismatches'][:3]}")
        if self.config.replay_e0_inference and not self.config.smoke and self.bundle is not None:
            replay = self._replay_e0()
            r2 = regression_r2_compare(replay, e0, atol=self.config.regression_atol, rtol=self.config.regression_rtol)
            self.progress.log(f"E0-R2 ok={r2['ok']}")
            if not r2["ok"]:
                write_regression_report(out / "e0_regression.json", {"r1": r1, "r2": r2})
                raise RuntimeError(f"E0-R2 replay failed: {r2['mismatches'][:3]}")
        write_regression_report(out / "e0_regression.json", {"r1": r1, "r2": r2})

    def _replay_e0(self) -> dict[str, dict[str, Any]]:
        assert self.bundle is not None
        records = self.records.get("seen_exploratory_dev") or ()
        if not records:
            return {}
        b = self._e0_b()
        if b is None:
            return {}
        replay_root = self.output_dir / "regression" / "e0_replay"
        out: dict[str, dict[str, Any]] = {}
        rrf = _score_rrf(self.bundle, self.config.rrf_k)

        def score_d2(question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
            retrieval = rrf(question, chunks)
            ans = cross_encoder_scores(question, chunks, b)
            section = section_scores(question, chunks, None)
            return structural_mix(retrieval, ans, section)

        scorers = {
            "a0": _score_dense(self.bundle),
            "a": _score_ce(self.bundle.relevance_reranker),
            "b": _score_ce(b),
            "d1": rrf,
            "d2": score_d2,
        }
        for name, fn in scorers.items():
            payload = evaluate_ranking(
                records,
                fn,
                experiment=f"e0_replay_{name}",
                split="seen_exploratory_dev",
                ks=self.config.top_k_values,
                progress=self.progress,
                answer_fn=self.bundle.answer,
                skip_qa=self.config.skip_qa,
                skip_bertscore=True,
                select_k=1,
                qa_mode="top1",
                qa_answerer=self.bundle.qa_answerer,
            )
            write_run(replay_root / name, experiment=name, split="dev", payload=payload, config=self.config.to_dict())
            out[name] = payload["summary"]
        payload = evaluate_ranking(
            records,
            scorers["b"],
            experiment="e0_replay_c",
            split="seen_exploratory_dev",
            ks=self.config.top_k_values,
            progress=self.progress,
            answer_fn=self.bundle.answer,
            skip_qa=self.config.skip_qa,
            skip_bertscore=True,
            select_k=5,
            qa_mode="global",
            qa_answerer=self.bundle.qa_answerer,
        )
        write_run(replay_root / "c", experiment="c", split="dev", payload=payload, config=self.config.to_dict())
        out["c"] = payload["summary"]
        return out

    def stage_reader_oracle(self) -> None:
        out = self.output_dir / "analysis" / "reader_oracle"
        out.mkdir(parents=True, exist_ok=True)
        merged = []
        for split in self._eval_splits():
            rows = self.records.get(split) or ()
            if not rows:
                continue
            self.progress.log(f"reader-oracle split={split} n={len(rows)}")
            run_reader_oracle(
                rows,
                answer_fn=self._answer_fn(),
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash=self._inf_hash(),
                output_dir=out / split,
                progress=self.progress,
                skip_bertscore=self.config.skip_bertscore,
            )
            merged.extend(read_jsonl(out / split / "per_question.jsonl"))
        write_json(
            out / "summary.json",
            {
                "n_questions": len(merged),
                "oracle": True,
                "diagnostic_only": True,
                "reader_solvable_EM": _avg(merged, "reader_solvable_EM"),
                "reader_solvable_F1>0": _avg(merged, "reader_solvable_F1>0"),
                "reader_solvable_F1>=50": _avg(merged, "reader_solvable_F1>=50"),
                "cache": self.cache.stats(),
            },
        )

    def stage_ambiguous_audit(self, tag: str) -> None:
        if tag == "b2":
            answerability = self._b_scorer()
        else:
            e0 = self._e0_b()
            if e0 is not None:
                answerability = FrozenAnswerability(e0.predict)
            else:
                answerability = FrozenAnswerability(self.bundle.relevance_reranker.predict) if self.bundle else None
        out = self.output_dir / "analysis" / "ambiguous_audit" / tag
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for split in self._eval_splits():
            part = self.records.get(split) or ()
            if not part:
                continue
            run_ambiguous_audit(
                part,
                encode=self.bundle.encode if self.bundle else None,
                relevance=self.bundle.relevance_reranker if self.bundle else None,
                answerability=answerability,
                answer_fn=self._answer_fn(),
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash=self._inf_hash(),
                output_dir=out / split,
                tag=f"{tag}-{split}",
                progress=self.progress,
                skip_bertscore=self.config.skip_bertscore,
            )
            rows.extend(read_jsonl(out / split / "rows.jsonl"))
        from .ambiguous_audit import export_top_errors

        export_top_errors(rows, out, top_n=40)
        write_json(out / "summary.json", {"tag": tag, "n": len(rows)})

    def stage_b2_train(self) -> None:
        cfg = self.config
        seeds = (cfg.seed,) if cfg.smoke else cfg.model_seeds
        variants = ("b2a",) if cfg.smoke else cfg.b2_variants
        train = self.records.get("train") or ()
        cal = self.records.get("calibration") or ()
        for variant in variants:
            for seed in seeds:
                out = cfg.output_dir / "models" / "b2" / variant / str(seed)
                if cfg.resume and (out / "wrapper.json").exists():
                    self.progress.log(f"resume B2 {variant} seed={seed}")
                    continue
                self.progress.log(f"training B2 {variant} seed={seed}")
                train_b2_variant(
                    train,
                    cal,
                    self.flags,
                    variant=variant,
                    seed=int(seed),
                    device=self.bundle.device if self.bundle else "cpu",
                    model_name=cfg.cross_encoder_model,
                    output_dir=out,
                    epochs=1 if cfg.smoke else cfg.b2_epochs,
                    patience=1 if cfg.smoke else cfg.b2_patience,
                    batch_size=cfg.b2_batch_size,
                    learning_rate=cfg.b2_learning_rate,
                    max_length=cfg.b2_max_length,
                    ratio=cfg.b2_pos_hn_easy,
                    smoke=cfg.smoke,
                    progress=self.progress,
                )

    def stage_b2_eval(self) -> None:
        cfg = self.config
        seeds = (cfg.seed,) if cfg.smoke else cfg.model_seeds
        variants = ("b2a",) if cfg.smoke else cfg.b2_variants
        flags = flag_index(self.flags)
        table = []
        best = None
        for variant in variants:
            seed_metrics = []
            for seed in seeds:
                path = cfg.output_dir / "models" / "b2" / variant / str(seed)
                model = load_b2(path, self.bundle.device if self.bundle else "cpu", cfg.smoke)
                if model is None:
                    continue
                for split in ("train", "calibration", "development"):
                    rows = self.records.get(split) or ()
                    metrics = evaluate_answerability(rows, model, flags)
                    self.progress.log(
                        f"B2 {variant} seed={seed} {split} pairwise_clean={metrics['pairwise_clean_hn']:.4f} pr={metrics['pr_auc']:.4f}"
                    )
                    table.append({"variant": variant, "seed": seed, "split": split, **metrics})
                    if split == "development":
                        seed_metrics.append(metrics["pairwise_clean_hn"])
            summary = mean_std([item for item in seed_metrics if item == item])
            score = (summary["mean"],)
            if best is None or score > best["score"]:
                best = {"variant": variant, "score": score, "summary": summary}
        if best is None:
            raise RuntimeError("no B2 variant produced a checkpoint")
        cal_rows = [
            row
            for row in table
            if row.get("variant") == best["variant"] and row.get("split") == "calibration"
        ]
        if cal_rows:
            chosen_seed = int(max(cal_rows, key=lambda row: float(row.get("pairwise_clean_hn") or float("-inf")))["seed"])
        else:
            chosen_seed = int(seeds[0])
        src = cfg.output_dir / "models" / "b2" / best["variant"] / str(chosen_seed)
        dest = cfg.output_dir / "models" / "b2" / "selected"
        _copy_model(src, dest)
        self.b2 = load_b2(dest, self.bundle.device if self.bundle else "cpu", cfg.smoke)
        self.b2_meta = {"variant": best["variant"], "seed_used_for_downstream": chosen_seed, "dev_pairwise_clean": best["summary"]}
        write_json(cfg.output_dir / "models" / "b2" / "selection.json", {"table": table, "selected": self.b2_meta})
        scorer = self._b_scorer()
        for split in self._eval_splits():
            rows = self.records.get(split) or ()
            if not rows:
                continue
            run_inference(
                rows,
                scorer.score_instance,
                ranking_method="b2",
                split=split,
                output_dir=cfg.run_dir("b2", chosen_seed) / split,
                config=cfg,
                progress=self.progress,
                answer_fn=lambda runtime, ranked: answer_top_m(
                    runtime, ranked, answer_fn=self._answer_fn(), qa_answerer=self.bundle.qa_answerer, m=1, cache=self.cache, model_id=self._model_id(), inference_hash=self._inf_hash()
                ),
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )

    def stage_reader_aware(self) -> None:
        cfg = self.config
        b = self._b_scorer()
        a_model = self.bundle.relevance_reranker if self.bundle else None
        cal = self.records.get("calibration") or ()
        k_values = (3,) if cfg.smoke else tuple(sorted(cfg.r3_k_pre_values, reverse=True))
        fusions = ("minmax",) if cfg.smoke else ("minmax", "rank")
        names = list(r3_weight_grid())
        if cfg.smoke:
            names = [names[0]]
        cal_rows = []
        best_by_name: dict[str, dict[str, Any]] = {}
        for name, weights in names:
            for fusion in fusions:
                for k_pre in k_values:
                    payload = run_inference(
                        cal,
                        self._r3_score_fn(weights=weights, k_pre=k_pre, fusion=fusion, mode="topk", a_model=a_model, b=b),
                        ranking_method=f"{name}_{fusion}_k{k_pre}",
                        split="calibration",
                        output_dir=cfg.output_dir / "runs" / "reader_aware_topk" / name / fusion / f"k{k_pre}" / "calibration",
                        config=cfg,
                        progress=self.progress,
                        answer_fn=self._qa_top1,
                        skip_qa=cfg.skip_qa,
                        flags=self.flags,
                    )
                    summary = payload["summary"]
                    row = {
                        "name": name,
                        "mode": "topk",
                        "k_pre": k_pre,
                        "fusion": fusion,
                        "weights": list(weights),
                        **{key: summary.get(key) for key in ("hit@1", "hn@1", "token_f1", "pairwise_clean_hn", "f1_given_selected_positive")},
                    }
                    cal_rows.append(row)
                    self.progress.log(
                        f"R3 {name} fusion={fusion} k_pre={k_pre} cal hit@1={summary.get('hit@1')} f1={summary.get('token_f1')}"
                    )
                    current = best_by_name.get(name)
                    if current is None or _selection_key(row) > _selection_key(current):
                        best_by_name[name] = row
        # Architecture R3_0..R3_3 is chosen on DEVELOPMENT with calibration-tuned K_pre/fusion.
        dev = self.records.get("development") or ()
        dev_rows = []
        for name, tuned in best_by_name.items():
            payload = run_inference(
                dev,
                self._r3_score_fn(
                    weights=tuple(tuned["weights"]),
                    k_pre=int(tuned["k_pre"]),
                    fusion=str(tuned["fusion"]),
                    mode="topk",
                    a_model=a_model,
                    b=b,
                ),
                ranking_method=f"{name}_dev",
                split="development",
                output_dir=cfg.output_dir / "runs" / "reader_aware_topk" / name / "development",
                config=cfg,
                progress=self.progress,
                answer_fn=self._qa_top1,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
            summary = payload["summary"]
            row = {**tuned, "split": "development", **{key: summary.get(key) for key in ("hit@1", "hn@1", "token_f1", "pairwise_clean_hn", "f1_given_selected_positive")}}
            dev_rows.append(row)
            self.progress.log(f"R3 {name} development hit@1={summary.get('hit@1')} f1={summary.get('token_f1')}")
        winner = max(dev_rows, key=_selection_key) if dev_rows else best_by_name.get("R3_0") or {
            "name": "R3_0",
            "mode": "topk",
            "k_pre": cfg.r3_default_k_pre,
            "fusion": "minmax",
            "weights": [0.0, 1.0, 0.0],
        }
        self.r3_meta = {k: winner[k] for k in ("name", "mode", "k_pre", "fusion", "weights") if k in winner}
        write_json(
            cfg.output_dir / "analysis" / "reader_aware" / "summary.json",
            {"calibration": cal_rows, "development": dev_rows, "selected": self.r3_meta},
        )
        for split in ("seen_exploratory_dev",):
            rows = self.records.get(split) or ()
            if not rows:
                continue
            run_inference(
                rows,
                self._r3_score_fn(
                    weights=tuple(self.r3_meta["weights"]),
                    k_pre=int(self.r3_meta["k_pre"]),
                    fusion=str(self.r3_meta["fusion"]),
                    mode="topk",
                    a_model=a_model,
                    b=b,
                ),
                ranking_method="r3_selected",
                split=split,
                output_dir=cfg.output_dir / "runs" / "reader_aware_topk" / "selected" / split,
                config=cfg,
                progress=self.progress,
                answer_fn=self._qa_top1,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
        selected_dev = cfg.output_dir / "runs" / "reader_aware_topk" / winner["name"] / "development"
        dest = cfg.output_dir / "runs" / "reader_aware_topk" / "selected" / "development"
        if selected_dev.exists():
            import shutil

            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(selected_dev, dest)
        if not cfg.smoke and dev:
            run_inference(
                dev,
                self._r3_score_fn(
                    weights=tuple(self.r3_meta["weights"]),
                    k_pre=len(dev[0].chunks) if dev else 8,
                    fusion=str(self.r3_meta["fusion"]),
                    mode="full",
                    a_model=a_model,
                    b=b,
                ),
                ranking_method="r3_full_diagnostic",
                split="development",
                output_dir=cfg.output_dir / "runs" / "reader_aware_full" / "development",
                config=cfg,
                progress=self.progress,
                answer_fn=self._qa_top1,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )

    def stage_c_sweep(self) -> None:
        cfg = self.config
        self._load_selection_artifacts()
        b = self._b_scorer()
        ms = (1, 2, 5) if cfg.smoke else cfg.passage_m_values
        rankers: list[tuple[str, Any]] = [("b2", b.score_instance)]
        if self.r3_meta.get("weights"):
            rankers.append(
                (
                    "r3_selected",
                    self._r3_score_fn(
                        weights=tuple(self.r3_meta["weights"]),
                        k_pre=int(self.r3_meta.get("k_pre") or cfg.r3_default_k_pre),
                        fusion=str(self.r3_meta.get("fusion") or "minmax"),
                        mode="topk",
                    ),
                )
            )
        table = []
        by_ranking: dict[str, dict[str, Any]] = {}
        for ranking_name, score_fn in rankers:
            for m in ms:
                rows = self.records.get("calibration") or ()
                payload = run_inference(
                    rows,
                    score_fn,
                    ranking_method=f"c_rawlogit_{ranking_name}_m{m}",
                    split="calibration",
                    output_dir=cfg.output_dir / "runs" / f"c_m{m}" / ranking_name / "calibration",
                    config=cfg,
                    progress=self.progress,
                    answer_fn=lambda runtime, ranked, mm=m: answer_top_m(
                        runtime,
                        ranked,
                        answer_fn=self._answer_fn(),
                        qa_answerer=self.bundle.qa_answerer,
                        m=mm,
                        cache=self.cache,
                        model_id=self._model_id(),
                        inference_hash=self._inf_hash(),
                        mode="global",
                    ),
                    skip_qa=cfg.skip_qa,
                    flags=self.flags,
                )
                table.append({"ranking": ranking_name, "M": m, "split": "calibration", **payload["summary"]})
                self.progress.log(f"C M={m} {ranking_name} calibration F1={payload['summary'].get('token_f1')}")
            cal = [row for row in table if row.get("ranking") == ranking_name and row.get("split") == "calibration"]
            winner = max(cal, key=lambda row: _finite(row, "token_f1")) if cal else {"M": 5}
            m_star = int(winner.get("M") or 5)
            dev_rows = self.records.get("development") or ()
            payload = run_inference(
                dev_rows,
                score_fn,
                ranking_method=f"c_rawlogit_{ranking_name}_m{m_star}",
                split="development",
                output_dir=cfg.output_dir / "runs" / f"c_m{m_star}" / ranking_name / "development",
                config=cfg,
                progress=self.progress,
                answer_fn=lambda runtime, ranked, mm=m_star: answer_top_m(
                    runtime,
                    ranked,
                    answer_fn=self._answer_fn(),
                    qa_answerer=self.bundle.qa_answerer,
                    m=mm,
                    cache=self.cache,
                    model_id=self._model_id(),
                    inference_hash=self._inf_hash(),
                    mode="global",
                ),
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
            table.append({"ranking": ranking_name, "M": m_star, "split": "development", **payload["summary"]})
            by_ranking[ranking_name] = {"M_fixed": m_star, "development": payload["summary"]}
            self.progress.log(f"C selected M={m_star} for {ranking_name} development F1={payload['summary'].get('token_f1')}")
        primary = "r3_selected" if "r3_selected" in by_ranking else "b2"
        self.c_meta = {
            "M_fixed": int(by_ranking.get(primary, {}).get("M_fixed") or 5),
            "ranking_for_M": primary,
            "by_ranking": by_ranking,
            "table": table,
            "note": "c_rawlogit is not trained Multi-passage BERT; M chosen on CALIBRATION, confirmed on DEVELOPMENT",
        }
        write_json(cfg.output_dir / "analysis" / "c_sweep" / "summary.json", self.c_meta)

    def stage_c_adaptive(self) -> None:
        cfg = self.config
        b = self._b_scorer()
        max_m = max(cfg.passage_m_values)
        grid = [(0.7, 0.15, 3)] if cfg.smoke else calibration_grid()
        cal_rows = self.records.get("calibration") or ()
        scored = []
        for tau_c, tau_m, m_u in grid:
            counts: dict[int, int] = defaultdict(int)

            def answer_fn(runtime, ranked, tc=tau_c, tm=tau_m, mu=m_u):
                scores = b.score_instance(runtime)
                used = choose_m(scores, tau_conf=tc, tau_margin=tm, m_uncertain=mu, max_m=max_m)
                counts[used] += 1
                return answer_top_m(
                    runtime,
                    ranked,
                    answer_fn=self._answer_fn(),
                    qa_answerer=self.bundle.qa_answerer,
                    m=used,
                    cache=self.cache,
                    model_id=self._model_id(),
                    inference_hash=self._inf_hash(),
                    mode="global",
                )

            payload = run_inference(
                cal_rows,
                b.score_instance,
                ranking_method=f"adaptive_{tau_c}_{tau_m}_{m_u}",
                split="calibration",
                output_dir=cfg.output_dir / "runs" / "c_adaptive" / f"{tau_c}_{tau_m}_{m_u}" / "calibration",
                config=cfg,
                progress=self.progress,
                answer_fn=answer_fn,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
            summary = payload["summary"]
            scored.append({"tau_conf": tau_c, "tau_margin": tau_m, "m_uncertain": m_u, "m_hist": dict(counts), **{k: summary.get(k) for k in ("token_f1", "exact_match", "hit@1")}})
            self.progress.log(f"adaptive tau={tau_c}/{tau_m} Mu={m_u} F1={summary.get('token_f1')} hist={dict(counts)}")
        winner = max(scored, key=lambda row: float(row.get("token_f1") or 0)) if scored else {"tau_conf": 0.7, "tau_margin": 0.15, "m_uncertain": 3}
        self.adaptive_meta = winner
        write_json(cfg.output_dir / "analysis" / "c_adaptive" / "summary.json", {"calibration": scored, "selected": winner})
        for split in ("development",):
            rows = self.records.get(split) or ()
            if not rows:
                continue
            counts = defaultdict(int)

            def answer_fn(runtime, ranked):
                scores = b.score_instance(runtime)
                used = choose_m(
                    scores,
                    tau_conf=float(winner["tau_conf"]),
                    tau_margin=float(winner["tau_margin"]),
                    m_uncertain=int(winner["m_uncertain"]),
                    max_m=max_m,
                )
                counts[used] += 1
                return answer_top_m(
                    runtime,
                    ranked,
                    answer_fn=self._answer_fn(),
                    qa_answerer=self.bundle.qa_answerer,
                    m=used,
                    cache=self.cache,
                    model_id=self._model_id(),
                    inference_hash=self._inf_hash(),
                    mode="global",
                )

            payload = run_inference(
                rows,
                b.score_instance,
                ranking_method="c_adaptive",
                split=split,
                output_dir=cfg.output_dir / "runs" / "c_adaptive" / "selected" / split,
                config=cfg,
                progress=self.progress,
                answer_fn=answer_fn,
                skip_qa=cfg.skip_qa,
                flags=self.flags,
            )
            self.progress.log(f"adaptive {split} F1={payload['summary'].get('token_f1')} M_hist={dict(counts)}")

    def stage_d2(self) -> None:
        cfg = self.config
        b = self._b_scorer()
        variants = ("D2a", "D2b", "D2e") if cfg.smoke else D2_VARIANTS
        table = []
        for variant in variants:
            for split in ("calibration", "development"):
                rows = self.records.get(split) or ()
                payload = run_inference(
                    rows,
                    lambda runtime, v=variant: d2_scores(
                        runtime,
                        variant=v,
                        b_scores=b.score_instance(runtime),
                        encode=self.bundle.encode,
                        relevance=self.bundle.relevance_reranker,
                        rrf_k=cfg.rrf_k,
                    ),
                    ranking_method=variant,
                    split=split,
                    output_dir=cfg.output_dir / "runs" / "d2_ablation" / variant / split,
                    config=cfg,
                    progress=self.progress,
                    answer_fn=lambda runtime, ranked: answer_top_m(
                        runtime, ranked, answer_fn=self._answer_fn(), qa_answerer=self.bundle.qa_answerer, m=1, cache=self.cache, model_id=self._model_id(), inference_hash=self._inf_hash()
                    ),
                    skip_qa=cfg.skip_qa,
                    flags=self.flags,
                )
                table.append({"variant": variant, "split": split, **payload["summary"]})
                self.progress.log(f"{variant} {split} hit@1={payload['summary'].get('hit@1')} F1={payload['summary'].get('token_f1')}")
        dev = [row for row in table if row.get("split") == "development"]
        winner = max(dev, key=lambda row: _finite(row, "token_f1")) if dev else {"variant": "D2a"}
        self.d2_meta = {
            "selected": winner.get("variant"),
            "table": table,
            "weights_source": "fixed_a_priori; numerical weights are not re-tuned per variant",
        }
        write_json(cfg.output_dir / "analysis" / "d2_ablation" / "summary.json", self.d2_meta)

    def stage_compare(self) -> None:
        cfg = self.config
        self._load_selection_artifacts()
        seed = int(self.b2_meta.get("seed_used_for_downstream") or (cfg.seed if cfg.smoke else cfg.model_seeds[0]))
        loaded = {}
        for name, rel in (
            ("b2", cfg.run_dir("b2", seed) / "development" / "predictions_scored.jsonl"),
            ("r3_selected", cfg.output_dir / "runs" / "reader_aware_topk" / "selected" / "development" / "predictions_scored.jsonl"),
        ):
            if rel.exists():
                loaded[name] = read_jsonl(rel)
        comparisons = {}
        if "b2" in loaded and "r3_selected" in loaded:
            comparisons["R3-B2_token_f1"] = paired_delta_multiseed(
                {0: loaded["b2"]}, {0: loaded["r3_selected"]}, "token_f1", n_boot=min(500, cfg.n_bootstrap) if cfg.smoke else cfg.n_bootstrap, seed=cfg.bootstrap_seed
            )
            comparisons["R3-B2_hit@1"] = paired_delta_multiseed(
                {0: loaded["b2"]}, {0: loaded["r3_selected"]}, "hit@1", n_boot=min(500, cfg.n_bootstrap) if cfg.smoke else cfg.n_bootstrap, seed=cfg.bootstrap_seed
            )
        write_json(cfg.output_dir / "analysis" / "paired_comparisons.json", comparisons)
        self.progress.log(f"compare-v2 pairs={list(comparisons)}")

    def stage_report(self) -> None:
        path = write_results_v2(self.output_dir, allow_internal_test=self.config.allow_internal_test)
        self.progress.log(f"wrote {path}")

    def stage_freeze(self) -> None:
        self._load_selection_artifacts()
        extra = {
            "selected_b2": self.b2_meta,
            "selected_r3": self.r3_meta,
            "selected_c": self.c_meta,
            "selected_adaptive": self.adaptive_meta,
            "selected_d2": self.d2_meta,
            "qa_model": self._model_id() if self.bundle else None,
            "ce_model": self.config.cross_encoder_model,
            "source_offsets": "chunk.char_start + chunk-local offsets",
            "final_ranking": "selected_r3 if present else b2",
            "final_qa_policy": "c_fixed_M from CALIBRATION; adaptive M is recorded as an ablation",
        }
        path = write_freeze_manifest(self.config, extra)
        self.progress.log(f"freeze-final wrote {path}")

    def stage_final_eval(self) -> None:
        cfg = self.config
        if not cfg.allow_internal_test:
            raise PermissionError("final-eval requires --allow-internal-test")
        verify_freeze_hash(cfg)
        self._load_selection_artifacts()
        rows = self.records.get("internal_test") or ()
        self.progress.log(f"final-eval INTERNAL_TEST n={len(rows)}")
        b = self._b_scorer()
        r3 = self.r3_meta
        if r3.get("weights"):
            score_fn = self._r3_score_fn(
                weights=tuple(r3["weights"]),
                k_pre=int(r3.get("k_pre") or cfg.r3_default_k_pre),
                fusion=str(r3.get("fusion") or "minmax"),
                mode="topk",
            )
            ranking_method = "frozen_r3"
        else:
            score_fn = b.score_instance
            ranking_method = "frozen_b2"
        m_fixed = int((self.c_meta or {}).get("M_fixed") or 1)

        def answer_fn(runtime, ranked):
            return answer_top_m(
                runtime,
                ranked,
                answer_fn=self._answer_fn(),
                qa_answerer=self.bundle.qa_answerer,
                m=m_fixed,
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash=self._inf_hash(),
                mode="global",
            )

        payload = run_inference(
            rows,
            score_fn,
            ranking_method=ranking_method,
            split="internal_test",
            output_dir=cfg.output_dir / "runs" / "final_internal_test" / ranking_method,
            config=cfg,
            progress=self.progress,
            answer_fn=answer_fn,
            skip_qa=cfg.skip_qa,
            flags=self.flags,
        )
        (cfg.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "reports" / "final_internal_test.md").write_text(
            "# INTERNAL_TEST\n\n" + json.dumps(payload["summary"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def stage_d4_train(self) -> None:
        flags = flag_index(self.flags)
        examples = []
        for item in self.records.get("train") or ():
            examples.extend(d4_training_examples(item, flags))
        out = self.output_dir / "models" / "sentence_evidence"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "train_stats.json", {"n": len(examples), "splitter": SENTENCE_SPLITTER_VERSION, "positives": sum(row["y"] == 1 for row in examples)})
        if self.config.smoke:
            write_json(out / "wrapper.json", {"kind": "d4_placeholder", "n": len(examples)})
        self.progress.log(f"d4-train examples={len(examples)}")

    def stage_d4(self) -> None:
        cfg = self.config
        b = self._b_scorer()
        rows_out = []
        for split in ("development",):
            for item in self.records.get(split) or ():
                units = sentence_units(item)
                runtime = to_runtime(item)
                scores = b.score_instance(runtime)
                top = sorted(range(len(item.chunks)), key=lambda i: -scores[i])[:3]
                selected = []
                for index in top:
                    cid = item.chunks[index].chunk_id
                    sents = [unit["sentence_id"] for unit in units if unit["chunk_id"] == cid]
                    if sents:
                        selected.append(sents[0])
                if not selected and units:
                    selected = [units[0]["sentence_id"]]
                compact, mapping = assemble_d4b(units, selected, window=1)
                result = self.bundle.answer(item.question, compact) if compact else {"answer": "", "start": 0, "end": 0}
                proj = project_assembled_span(compact, mapping, int(result.get("start") or 0), int(result.get("end") or 0))
                rows_out.append({"question_id": item.question_id, "qa_answer": result.get("answer"), "projection": proj, "splitter": SENTENCE_SPLITTER_VERSION})
        write_jsonl(cfg.output_dir / "runs" / "d4" / "development" / "predictions_raw.jsonl", rows_out)
        write_json(cfg.output_dir / "analysis" / "d4" / "summary.json", {"n": len(rows_out), "splitter": SENTENCE_SPLITTER_VERSION})

    def stage_d5(self) -> None:
        cfg = self.config
        b = self._b_scorer()
        m = int((self.c_meta or {}).get("M_fixed") or 5)
        table = []
        for mode in ("D5a", "D5b", "D5c"):
            scored = []
            for item in self.records.get("development") or ():
                runtime = to_runtime(item)
                scores = b.score_instance(runtime)
                ranked = [chunk.chunk_id for _, chunk in sorted(zip(scores, runtime.chunks), key=lambda pair: -pair[0])]
                spans = []
                for chunk_id in ranked[:m]:
                    chunk = runtime.chunk_by_id(chunk_id)
                    ans = self.bundle.answer(item.question, chunk.text)
                    from .schemas import source_offsets as _src

                    s0, s1 = _src(chunk, int(ans.get("start") or 0), int(ans.get("end") or 0))
                    idx = ranked.index(chunk_id)
                    spans.append(
                        {
                            "answer_text": ans.get("answer") or "",
                            "text": ans.get("answer") or "",
                            "chunk_id": chunk_id,
                            "source_chunk_id": chunk_id,
                            "source_start_char": s0,
                            "source_end_char": s1,
                            "candidate_score": 1.0 / (60 + idx + 1),
                            "qa_span_score": float(ans.get("start_logit") or 0) + float(ans.get("end_logit") or 0),
                        }
                    )
                agg = aggregate_d5(spans, mode=mode, score_field="candidate_score", lambda_support=0.2 if mode == "D5c" else 0.0)
                from experiments.qa_chunk_selection.metrics import qa_row

                scored.append({"question_id": item.question_id, "qa_answer": agg["text"], **qa_row(agg["text"], item.gold_answers)})
            summary = {"mode": mode, "token_f1": _avg(scored, "token_f1"), "exact_match": _avg(scored, "exact_match")}
            table.append(summary)
            self.progress.log(f"{mode} F1={summary['token_f1']}")
        write_json(cfg.output_dir / "analysis" / "d5" / "summary.json", {"table": table})

    def stage_unanswerable(self) -> None:
        out = self.output_dir / "analysis" / "unanswerable_audit"
        out.mkdir(parents=True, exist_ok=True)
        allowed = []
        for item in self.all_records:
            if item.split == "internal_test" and not self.config.allow_internal_test:
                continue
            allowed.append(item)
        run_unanswerable_audit(allowed, out)


def _by_split(records: Sequence[InstanceRecord]) -> dict[str, tuple[InstanceRecord, ...]]:
    grouped: dict[str, list[InstanceRecord]] = defaultdict(list)
    for item in records:
        grouped[item.split].append(item)
    return {key: tuple(val) for key, val in grouped.items()}


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _finite(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("-inf")
    if value != value:
        return float("-inf")
    return value


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _finite(row, "pairwise_clean_hn"),
        _finite(row, "hit@1"),
        _finite(row, "f1_given_selected_positive"),
        _finite(row, "token_f1"),
    )


def _copy_model(src: Path, dest: Path) -> None:
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _maybe_json(path: Path) -> Any:
    if path.exists():
        return load_json(path)
    return {}
