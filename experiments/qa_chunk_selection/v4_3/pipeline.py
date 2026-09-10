"""Isolated V4.3 pipeline. V4.2 and earlier artifact trees are read-only. V6 is not retuned."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.qa_chunk_selection.metrics import ranking_row
from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v4.answers import pair_token_f1
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, hn_quality_rows
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v4.heads import rank_by_scores
from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device
from experiments.qa_chunk_selection.v4.qa_shortlist import answer_one
from experiments.qa_chunk_selection.v4.shortlist import b0_rows_for_seed, index_b0_oof, load_b0_oof, scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4_1.contract import PAIRWISE_TOLERANCE, V4_V6_WEIGHTS
from experiments.qa_chunk_selection.v4_1.ranking import shortlist_set_ok
from experiments.qa_chunk_selection.v4_2.cache_util import ReadThroughCache
from experiments.qa_chunk_selection.v4_2.heads import union_topk

from .config import ExperimentConfigV43
from .contract import (
    ALWAYS_TAU,
    HISTORICAL_B0,
    HISTORICAL_CORE3,
    HISTORICAL_V6_K5,
    NEVER_TAU,
    PRIMARY_FINAL_SYSTEMS,
    READER_EASY_THRESHOLD,
    experiment_contract_v4_3,
)
from .encoder import SemanticEncoder
from .evaluate import chunk_prediction, metric_close, paired_delta, seed_design, summarize_rows, transitions
from .features import b0_margin, cluster_diagnostics, lexical_pair_rows, semantic_pack, sim_at, support_gain, support_rel
from .io_guard import assert_isolated_output, file_sha256, load_json, read_jsonl, write_json, write_jsonl
from .ranking import (
    adapt_use_semcore,
    finite_tau,
    isolate_with_b0_tiebreak,
    lexcore_scores,
    routing_available,
    sem1_scores,
    semcore_scores,
    semctx_scores,
    v6_shortlist_scores,
)
from .report import write_report
from .sanity import gold_token_f1, sanity_pairs
from .select import choose_admissible_cal, choose_max_f1, mean_token_f1_across_seeds
from .text import answer_record, cosine, local_evidence_sentence, qa_answer_text, query_answer_text, semantic_normalize


class PipelineV43:
    def __init__(self, config: ExperimentConfigV43) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = Path(config.v3_artifact_dir)
        self.v4 = Path(config.v4_artifact_dir)
        self.v41 = Path(config.v4_1_artifact_dir)
        self.v42 = Path(config.v4_2_artifact_dir)
        fallbacks = []
        if config.reuse_v4_1_qa_cache and not config.smoke:
            if (self.v41 / "cache").exists():
                fallbacks.append(FeatureCache(self.v41 / "cache"))
            if (self.v4 / "cache").exists():
                fallbacks.append(FeatureCache(self.v4 / "cache"))
            if (self.v42 / "cache").exists():
                fallbacks.append(FeatureCache(self.v42 / "cache"))
        self.cache = ReadThroughCache(FeatureCache(self.output_dir / "cache"), fallbacks)
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.flags = {}
        self.flags_rows: list[dict[str, Any]] = []
        self.b0_rows: list[dict[str, Any]] = []
        self.b0_index: dict[tuple[str, int], dict[str, Any]] = {}
        self.qa_store: dict[tuple[str, int], dict[str, Any]] = {}
        self.v6_weights: dict[int, dict[str, float]] = {}
        self.v6_rows: list[dict[str, Any]] = []
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.packs: dict[tuple[str, int], dict[str, Any]] = {}
        self.encoder: SemanticEncoder | None = None
        self.sem1_gate = False
        self.sem1_hn_worse = False
        self.sem1_delta: dict[int, float] = {}
        self.lex_weights: dict[int, dict[str, float]] = {}
        self.semcore_weights: dict[int, dict[str, float]] = {}
        self.adapt_inner_weights: dict[int, dict[str, float]] = {}
        self.adapt_tau: dict[int, dict[str, Any]] = {}
        self.bundle = None
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "r0": self.stage_r0,
            "cache": self.stage_cache,
            "embed": self.stage_embed,
            "sanity": self.stage_sanity,
            "features": self.stage_features,
            "sem1": self.stage_sem1,
            "lexcore": self.stage_lexcore,
            "semcore": self.stage_semcore,
            "complement": self.stage_complement,
            "adapt1": self.stage_adapt1,
            "semctx1": self.stage_semctx1,
            "compare": self.stage_compare,
            "freeze": self.stage_freeze,
            "report": self.stage_report,
        }

    def run(self, stages: list[str] | None = None) -> int:
        names = list(stages or self.config.stages)
        self.progress.stage(
            "start",
            f"output={self.output_dir} smoke={self.config.smoke} device={self.config.device}->{self._resolved_device()} stages={names}",
        )
        self.progress.log("E0/V2/V3/V3.1/D45/V4/V4.1/V4.2 are read-only; V6 is not retuned")
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown V4.3 stage: {name}")
            self.progress.stage(name)
            handler()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')} exit={self.exit_code}")
        return self.exit_code

    def _resolved_device(self) -> str:
        return "cpu" if self.config.smoke else resolve_encoder_device(self.config.device)

    def _answer_fn(self):
        if self.config.smoke or self.config.skip_qa:
            from experiments.qa_chunk_selection.models import lexical_answer

            return lexical_answer
        if self.bundle is None:
            self.progress.log("loading frozen extractive QA only")
            self.bundle = load_frozen_qa(device=self.config.device, smoke=False)
        return self.bundle.answer

    def _model_id(self) -> str:
        if self.config.smoke or self.bundle is None:
            return "lexical-smoke" if self.config.smoke or self.config.skip_qa else "qa-pending"
        return str(self.bundle.qa_model_name)

    def _mkdir(self, *parts: str) -> Path:
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _b0_seed(self, seed: int) -> dict[str, dict[str, Any]]:
        return b0_rows_for_seed(self.b0_index, seed) if self.b0_index else {}

    def _by_qid(self):
        return {item.question_id: item for item in self.unlocked}

    def _n_boot(self) -> int:
        return 50 if self.config.smoke else int(self.config.n_bootstrap)

    def _record(self, name: str, rows: list[dict[str, Any]], dest: Path) -> dict[str, Any]:
        dest.mkdir(parents=True, exist_ok=True)
        self.predictions[name] = rows
        summary = summarize_rows(rows)
        self.summaries[name] = summary
        write_jsonl(dest / "predictions_scored.jsonl", rows)
        write_json(dest / "metrics.json", summary)
        self.progress.log(f"{name} done", n=len(rows), token_f1=summary.get("token_f1"))
        return summary

    def _qa_map(self, item, b0_row, k: int) -> dict[int, dict[str, Any]]:
        return {
            cid: self.qa_store[(item.question_id, cid)]
            for cid in top_k_ids(b0_row, k)
            if (item.question_id, cid) in self.qa_store
        }

    def _predict_ok(
        self,
        item,
        b0_row,
        scores,
        qa_by_chunk,
        ranked,
        method: str,
        extra: dict[str, Any] | None = None,
        *,
        preserve_scores: bool = False,
    ) -> dict[str, Any]:
        top = int(ranked[0]) if ranked else int(item.chunks[0].chunk_id)
        qa = qa_by_chunk.get(top) or self.qa_store.get((item.question_id, top)) or {"qa_answer": "", "qa_chunk_id": top}
        rank_scores = {int(cid): float(len(ranked) - index) for index, cid in enumerate(ranked)}
        pred = chunk_prediction(item, ranking_method=method, scores=rank_scores, qa=qa, flags=self.flags, skip_bertscore=True)
        rank_metrics = ranking_row(item, [int(cid) for cid in ranked], rank_scores, ks=(1, 5, 10))
        for key in ("mrr", "selected_chunk_id", "selected_chunk_label", "selected_is_hn", "pairwise_pos_gt_hn"):
            if key in rank_metrics:
                pred[key] = rank_metrics[key]
        for key, value in rank_metrics.items():
            if str(key).startswith(("hit@", "coverage@", "hn@")):
                pred[key] = value
        pred["shortlist_ok"] = shortlist_set_ok(b0_row, ranked, k=self.config.primary_k)
        pred["document_id"] = item.document_id
        pred["selected_chunk_id"] = top
        pred["ranked_ids"] = [int(cid) for cid in ranked]
        if preserve_scores:
            pred["fused_scores"] = {str(int(k)): float(v) for k, v in scores.items()}
        if extra:
            pred.update(extra)
        return pred

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v4_3())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        if self.config.smoke:
            records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=self.config.seed)
            self.flags_rows = hn_quality_rows(self.unlocked)
            self.flags = flag_index(self.flags_rows)
            self.progress.log("smoke: synthetic unlocked records", n=len(self.unlocked))
            return
        dataset = self.v3 / "data" / "dataset.jsonl"
        splits = self.v3 / "data" / "v3_cv_splits.json"
        if not dataset.exists() or not splits.exists():
            raise FileNotFoundError(f"V3 dataset/splits missing under {self.v3}")
        records = records_from_jsonl(dataset)
        self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
        self.splits = load_json(splits)
        flags_path = self.v3 / "data" / "hn_quality_flags.jsonl"
        if flags_path.exists():
            self.flags_rows = read_jsonl(flags_path)
            self.flags = flag_index(self.flags_rows)
        locked = [item for item in records if item.split in LOCKED_SPLITS]
        write_json(
            self.output_dir / "source_hashes.json",
            {
                "dataset": file_sha256(dataset),
                "splits": file_sha256(splits),
                "n_unlocked": len(self.unlocked),
                "n_internal_test": len(locked),
                "internal_test_locked": True,
            },
        )
        self.progress.log("loaded V3 universe (read-only)", n_unlocked=len(self.unlocked), n_internal_test=len(locked))

    def _synthetic_b0(self) -> list[dict[str, Any]]:
        rows = []
        for seed in self.config.model_seeds:
            for item in self.unlocked:
                scores = {int(chunk.chunk_id): float(len(chunk.text.split())) - 0.01 * float(chunk.position) for chunk in item.chunks}
                ranked = rank_by_scores(list(scores), [scores[cid] for cid in scores])
                pos = {int(chunk.chunk_id) for chunk in item.positives}
                rows.append(
                    {
                        "question_id": item.question_id,
                        "document_id": item.document_id,
                        "aspect": item.aspect,
                        "seed": int(seed),
                        "scores": {str(k): v for k, v in scores.items()},
                        "ranked_ids": ranked,
                        "selected_chunk_id": ranked[0] if ranked else None,
                        "qa_answer": item.gold_answers[0] if ranked and ranked[0] in pos and item.gold_answers else "none",
                        "qa_chunk_id": ranked[0] if ranked else None,
                        "hit@1": 1.0 if ranked and ranked[0] in pos else 0.0,
                        "hit@5": 1.0 if pos & set(ranked[:5]) else 0.0,
                        "hit@10": 1.0 if pos & set(ranked[:10]) else 0.0,
                        "hn@1": 0.0,
                        "mrr": 1.0 if ranked and ranked[0] in pos else 0.0,
                        "token_f1": 100.0 if ranked and ranked[0] in pos else 0.0,
                        "exact_match": 1.0 if ranked and ranked[0] in pos else 0.0,
                        "pairwise_clean_hn": 1.0,
                        "n_pairs_clean": 1,
                        "ambiguous_at_1_original": 0.0,
                        "gsrn_at_1_original": 0.0,
                        "selected_chunk_label": "positive" if ranked and ranked[0] in pos else "ambiguous",
                    }
                )
        return rows

    def stage_r0(self) -> None:
        dest = self._mkdir("reference", "b0_live")
        if self.config.smoke:
            self.b0_rows = self._synthetic_b0()
        else:
            self.b0_rows = load_b0_oof(self.v3)
        self.b0_index = index_b0_oof(self.b0_rows)
        write_jsonl(dest / "predictions_scored.jsonl", self.b0_rows)
        b0_summary = summarize_rows(self.b0_rows)
        self.predictions["B0-live"] = self.b0_rows
        self.summaries["B0-live"] = b0_summary
        write_json(dest / "metrics.json", b0_summary)
        ok = True
        identity: dict[str, Any] = {}
        if not self.config.smoke:
            for key, expected in HISTORICAL_B0.items():
                got = b0_summary.get(key) if key != "n" else len(self.b0_rows)
                close = metric_close(float(got), float(expected), tol=PAIRWISE_TOLERANCE if "pairwise" in key else 1e-6)
                identity[key] = {"expected": expected, "got": got, "ok": close}
                ok = ok and close
        v6_path = self.v41 / "r1_v6_oof" / "predictions_scored.jsonl"
        weights_path = self.v41 / "r1_v6_oof" / "weights_by_run.json"
        v6_dest = self._mkdir("reference", "v6_k5_v4_1")
        if self.config.smoke:
            self.v6_weights = {run: dict(V4_V6_WEIGHTS) for run in range(len(self.splits.get("runs") or [0, 1, 2, 3, 4]))}
            self.v6_rows = []
        elif v6_path.exists() and weights_path.exists():
            self.v6_rows = read_jsonl(v6_path)
            raw_w = load_json(weights_path)
            self.v6_weights = {int(k): dict(v) for k, v in raw_w.items()}
            write_jsonl(v6_dest / "predictions_scored.jsonl", self.v6_rows)
            v6_summary = summarize_rows(self.v6_rows)
            self.predictions["V6-K5"] = self.v6_rows
            self.summaries["V6-K5"] = v6_summary
            write_json(v6_dest / "metrics.json", v6_summary)
            v6_ok = metric_close(float(v6_summary.get("token_f1") or 0.0), HISTORICAL_V6_K5["token_f1"], tol=1e-6)
            identity["v6_k5"] = {"expected": HISTORICAL_V6_K5["token_f1"], "got": v6_summary.get("token_f1"), "ok": v6_ok}
            ok = ok and v6_ok
        else:
            raise FileNotFoundError(f"V4.1 V6-K5 OOF missing under {self.v41}; V4.3 cannot retune V6")
        write_json(v6_dest / "weights_by_run.json", {str(k): v for k, v in self.v6_weights.items()})
        self._copy_historical_core3(identity)
        clone = paired_delta(self.b0_rows, self.b0_rows, n_boot=min(self._n_boot(), 200), seed=self.config.seed)
        write_json(self.output_dir / "regression" / "r0_summary.json", {"b0_identity": identity, "ok": ok or self.config.smoke, "paired_identity": clone, "v6_weights_runs": sorted(self.v6_weights)})
        write_json(self.output_dir / "regression" / "top5_identity.json", {"ok": True, "note": "B0 ranked_ids are never rewritten"})
        if not (ok or self.config.smoke):
            self.exit_code = 2
            self.progress.warn("R0 identity failed")
        self.progress.log("R0 identity", ok=ok or self.config.smoke, b0=b0_summary.get("token_f1"))

    def _copy_historical_core3(self, identity: dict[str, Any]) -> None:
        dest = self._mkdir("reference", "core3_v4_2_historical")
        src = self.v42 / "core" / "core3" / "predictions_scored.jsonl"
        if self.config.smoke or not src.exists():
            write_json(dest / "note.json", {"copied": False, "reason": "smoke or V4.2 CORE3 missing"})
            return
        rows = read_jsonl(src)
        write_jsonl(dest / "predictions_scored.jsonl", rows)
        summary = summarize_rows(rows)
        write_json(dest / "metrics.json", summary)
        self.predictions["CORE3-historical"] = rows
        self.summaries["CORE3-historical"] = summary
        if not self.config.smoke:
            close = metric_close(float(summary.get("token_f1") or 0.0), HISTORICAL_CORE3["token_f1"], tol=1e-6)
            identity["core3_historical"] = {"expected": HISTORICAL_CORE3["token_f1"], "got": summary.get("token_f1"), "ok": close}

    def stage_cache(self) -> None:
        by_qid = self._by_qid()
        needed: set[tuple[str, int]] = set()
        short_rows = []
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                if qid not in by_qid:
                    continue
                top = top_k_ids(row, self.config.primary_k)
                short_rows.append({"question_id": qid, "seed": int(seed), "top5": top})
                for cid in top:
                    needed.add((qid, int(cid)))
        write_jsonl(self.output_dir / "cache" / "shortlists.jsonl", short_rows)
        answer_fn = self._answer_fn()
        for index, (qid, cid) in enumerate(sorted(needed), start=1):
            item = by_qid[qid]
            self.qa_store[(qid, cid)] = answer_one(
                item,
                cid,
                answer_fn,
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash=self.config.qa_inference_hash,
            )
            self.progress.count(index, max(len(needed), 1), item="qa_topk")
        write_jsonl(self.output_dir / "cache" / "qa_topk.jsonl", [{**row, "question_id": qid, "chunk_id": cid} for (qid, cid), row in self.qa_store.items()])
        if self.config.smoke and not self.v6_rows:
            self.v6_rows = self._eval_rows(lambda item, row, seed, run: self._v6_pred(item, row, run=run))
            self._record("V6-K5", self.v6_rows, self._mkdir("reference", "v6_k5_v4_1"))
        self.progress.log("cache ready", n_qa=len(self.qa_store), fallback_hits=self.cache.stats().get("fallback_hits"))

    def _v6_pred(self, item, b0_row, *, run: int) -> dict[str, Any]:
        weights = self.v6_weights.get(int(run), dict(V4_V6_WEIGHTS))
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        ranked, scores, _feats = v6_shortlist_scores(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=self.config.primary_k, min_overlap=self.config.source_overlap
        )
        return self._predict_ok(
            item,
            b0_row,
            scores,
            qa_by_chunk,
            ranked,
            "V6-K5",
            {"v6_shortlist_scores": {str(cid): float(scores[cid]) for cid in top_k_ids(b0_row, self.config.primary_k) if cid in scores}},
            preserve_scores=True,
        )

    def _v6_short(self, item, b0_row, run: int) -> tuple[dict[int, dict[str, Any]], list[int], dict[int, float]]:
        weights = self.v6_weights.get(int(run), dict(V4_V6_WEIGHTS))
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        ranked, scores, _feats = v6_shortlist_scores(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=self.config.primary_k, min_overlap=self.config.source_overlap
        )
        top = top_k_ids(b0_row, self.config.primary_k)
        return qa_by_chunk, ranked, {int(cid): float(scores.get(int(cid), 0.0)) for cid in top}

    def _eval_rows(self, fn) -> list[dict[str, Any]]:
        rows = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred = fn(item, row, seed, run)
                    pred["seed"] = int(seed)
                    pred["cv_run"] = int(run)
                    pred["document_id"] = item.document_id
                    rows.append(pred)
        return rows

    def _rows_for(self, items, run: int, fn) -> list[dict[str, Any]]:
        rows = []
        for seed in self.config.model_seeds:
            b0_map = self._b0_seed(int(seed))
            for item in items:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                pred = fn(item, row, seed, run)
                pred["seed"] = int(seed)
                pred["cv_run"] = int(run)
                pred["document_id"] = item.document_id
                rows.append(pred)
        return rows

    def _encoder(self) -> SemanticEncoder:
        if self.encoder is None:
            self.progress.log("loading frozen semantic encoder", model=self.config.semantic_model, smoke=self.config.smoke)
            self.encoder = SemanticEncoder(
                cache_root=self.output_dir / "semantic_cache",
                device=self.config.device,
                smoke=self.config.smoke,
                model_name=self.config.semantic_model,
                normalization_version=self.config.normalization_version,
                progress=self.progress,
            )
            write_json(self.output_dir / "semantic_cache" / "encoder_spec.json", self.encoder.spec())
        return self.encoder

    def stage_embed(self) -> None:
        encoder = self._encoder()
        texts = []
        for qa in self.qa_store.values():
            rec = answer_record(qa_answer_text(qa))
            if rec["valid"]:
                texts.append(rec["semantic_norm"])
        encoder.embed_many(texts, store="answer")
        encoder.write_contract(self.output_dir / "semantic_cache" / "feature_contract.json")
        self.progress.log("answer embeddings ready", n_unique=len(set(texts)), stats=encoder.stats())

    def stage_sanity(self) -> None:
        encoder = self._encoder()
        n_runs = len(self.splits.get("runs") or [])
        leakage = []
        summaries = []
        for run in range(n_runs):
            train_items = list(records_for_role(self.unlocked, self.splits, run=run, role="train"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            train_ids = {item.question_id for item in train_items}
            eval_ids = {item.question_id for item in eval_items}
            overlap = sorted(train_ids & eval_ids)
            if overlap:
                leakage.append({"run": run, "n": len(overlap)})
            b0_by_seed = {int(seed): self._b0_seed(int(seed)) for seed in self.config.model_seeds}
            union = union_topk(train_items, b0_by_seed, self.config.model_seeds, k=self.config.primary_k)
            embeddings = {}
            for item in train_items:
                for cid in union.get(item.question_id) or []:
                    qa = self.qa_store.get((item.question_id, int(cid)))
                    if not qa:
                        continue
                    rec = answer_record(qa_answer_text(qa))
                    if not rec["valid"]:
                        continue
                    embeddings[(item.question_id, int(cid))] = encoder.embed_one(rec["semantic_norm"], store="answer")
            payload = sanity_pairs(
                train_items,
                union,
                self.qa_store,
                embeddings,
                reader_easy_threshold=READER_EASY_THRESHOLD if not self.config.smoke else 0.0,
                gold_f1=lambda item, cid: gold_token_f1(item, self.qa_store, cid),
            )
            payload["run"] = run
            payload["n_train"] = len(train_items)
            payload["n_union_pairs"] = int(sum(len(v) for v in union.values()))
            write_json(self._mkdir("semantic_sanity", f"run{run}") / "pair_distributions.json", payload)
            summaries.append(payload)
            self.progress.log(f"sanity run {run}", pairwise=payload.get("pairwise_pospos_gt_posHN"))
        write_json(self.output_dir / "semantic_sanity" / "pair_distributions.json", {"runs": summaries, "eval_leakage_runs": leakage})
        write_json(self.output_dir / "regression" / "inner_train_no_eval_leakage.json", {"ok": not leakage, "eval_leakage_runs": leakage})
        if leakage:
            self.progress.warn("EVAL ids overlapped TRAIN sanity", n=len(leakage))
        self.progress.log("TRAIN semantic sanity done")

    def stage_features(self) -> None:
        encoder = self._encoder()
        by_qid = self._by_qid()
        pair_rows = []
        pack_rows = []
        n = 0
        total = len(self.config.model_seeds) * len(self.unlocked)
        for seed in self.config.model_seeds:
            b0_map = self._b0_seed(int(seed))
            for item in self.unlocked:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                top = top_k_ids(row, self.config.primary_k)
                answers = {int(cid): qa_answer_text(self.qa_store.get((item.question_id, int(cid)))) for cid in top}
                embeddings = {}
                for cid, ans in answers.items():
                    rec = answer_record(ans)
                    if rec["valid"]:
                        embeddings[int(cid)] = encoder.embed_one(rec["semantic_norm"], store="answer")
                pack = semantic_pack(answers, embeddings, top_ids=top)
                b0_scores = scores_from_row(row)
                pack["b0_margin"] = b0_margin([float(b0_scores.get(int(cid), 0.0)) for cid in top])
                pack["question_id"] = item.question_id
                pack["seed"] = int(seed)
                pack["document_id"] = item.document_id
                pack["aspect"] = item.aspect
                pack["top5"] = top
                pack["answers"] = {str(cid): answer_record(answers[int(cid)]) for cid in top}
                pack["clusters"] = cluster_diagnostics(pack.get("sim") or {}, pack.get("valid_ids") or [])
                self.packs[(item.question_id, int(seed))] = pack
                pack_rows.append({k: pack[k] for k in pack if k != "sim"})
                for pair in lexical_pair_rows(answers, pack.get("sim") or {}, pack.get("valid_ids") or []):
                    pair_rows.append({"question_id": item.question_id, "seed": int(seed), "aspect": item.aspect, **pair})
                n += 1
                self.progress.count(n, max(total, 1), item="sem_features")
        write_jsonl(self.output_dir / "semantic_cache" / "shortlist_features.jsonl", pack_rows)
        write_jsonl(self.output_dir / "semantic_cache" / "pairwise_similarity.jsonl", pair_rows)
        n_avail = sum(1 for pack in self.packs.values() if pack.get("semantic_available"))
        write_json(
            self.output_dir / "analysis" / "semantic_availability.json",
            {"n_shortlists": len(self.packs), "semantic_available": n_avail, "rate": n_avail / max(len(self.packs), 1)},
        )
        self._write_lexical_vs_semantic(pair_rows)
        self.progress.log("shortlist semantic features ready", n=len(self.packs), available=n_avail)
        del by_qid

    def _write_lexical_vs_semantic(self, rows: list[dict[str, Any]]) -> None:
        buckets = {
            "high_semantic_low_f1": 0,
            "high_both": 0,
            "low_both": 0,
            "high_f1_low_semantic": 0,
        }
        for row in rows:
            for key in buckets:
                buckets[key] += int(bool(row.get(key)))
        write_json(self.output_dir / "analysis" / "lexical_vs_semantic.json", {"n_pairs": len(rows), "buckets": buckets})

    def _pack(self, item, seed: int) -> dict[str, Any]:
        return self.packs.get((item.question_id, int(seed))) or {"sem_rel": {}, "semantic_available": False, "valid_ids": [], "best_triple": [], "adapt_answers_ok": False, "consensus_conf": None}

    def _sem_rel(self, item, seed: int) -> dict[int, float]:
        pack = self._pack(item, seed)
        return {int(k): float(v) for k, v in (pack.get("sem_rel") or {}).items()}

    def _live_v6_hn(self, items, run: int) -> float:
        rows = self._rows_for(items, run, lambda item, row, seed, _run: self._v6_pred(item, row, run=run))
        return float(summarize_rows(rows).get("hn@1") or 0.0) if rows else 0.0

    def _seed_mean_deltas(self, baseline, candidate, *, metric: str = "token_f1") -> dict[str, float]:
        left = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in baseline}
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in candidate:
            prev = left.get((str(row["question_id"]), int(row.get("seed") or 0)))
            if not prev:
                continue
            by_seed[int(row.get("seed") or 0)].append(float(row.get(metric) or 0.0) - float(prev.get(metric) or 0.0))
        return {str(seed): float(sum(values) / len(values)) for seed, values in sorted(by_seed.items()) if values}

    def _answer_pair_diag(self, left, right, pack: dict[str, Any]) -> dict[str, Any]:
        a = str(left.get("qa_answer") or "")
        b = str(right.get("qa_answer") or "")
        cid_a = int(left.get("selected_chunk_id") or -1)
        cid_b = int(right.get("selected_chunk_id") or -1)
        matrix = pack.get("sim") or {}
        semantic = None
        if cid_a >= 0 and cid_b >= 0 and matrix:
            semantic = float(sim_at(matrix, cid_a, cid_b)) if cid_a != cid_b else 1.0
        return {
            "token_f1_similarity": float(pair_token_f1(a, b)),
            "semantic_similarity": semantic,
            "selected_source_left": left.get("selected_chunk_label"),
            "selected_source_right": right.get("selected_chunk_label"),
        }

    def _sem1_pred(self, item, b0_row, seed: int, run: int, *, delta: float) -> dict[str, Any]:
        qa_by_chunk, ranked_v6, v6_short = self._v6_short(item, b0_row, run)
        pack = self._pack(item, seed)
        available = bool(pack.get("semantic_available"))
        use_v6 = (float(delta) == 0.0) or (not available)
        extra = {
            "semantic_available": available,
            "delta_s": 0.0 if use_v6 else float(delta),
            "sem_rel": pack.get("sem_rel") or {},
            "fallback_v6": use_v6,
        }
        if use_v6:
            return self._predict_ok(
                item,
                b0_row,
                {int(cid): float(v6_short.get(int(cid), 0.0)) for cid in v6_short},
                qa_by_chunk,
                ranked_v6,
                "SEM1",
                extra,
                preserve_scores=True,
            )
        scores = sem1_scores(v6_short, self._sem_rel(item, seed), delta=float(delta))
        ranked, fused, _tied = isolate_with_b0_tiebreak(item, b0_row, scores, k=self.config.primary_k)
        return self._predict_ok(item, b0_row, fused, qa_by_chunk, ranked, "SEM1", extra)

    def stage_sem1(self) -> None:
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        selected = {}
        v6_index = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in self.v6_rows}
        for run in range(n_runs):
            self.progress.log(f"SEM1 outer run {run}/{max(n_runs - 1, 0)}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_hn = self._live_v6_hn(cal_items, run)

            def predict(params, items=cal_items, run_id=run):
                return self._rows_for(items, run_id, lambda item, row, seed, _run: self._sem1_pred(item, row, seed, run_id, delta=float(params["delta"])))

            choice = choose_admissible_cal(
                cal_items,
                grid=[{"delta": delta} for delta in self.config.delta_sem_grid],
                predict_rows=predict,
                hn_ref=v6_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("delta",),
                progress=self.progress,
                item_name=f"sem1_run{run}",
            )
            delta = float(choice["selected"]["delta"])
            self.sem1_delta[run] = delta
            selected[run] = choice["selected"]
            run_dir = self._mkdir("sem1", f"run{run}")
            write_json(run_dir / "selected_delta.json", choice)
            d0_rows = self._rows_for(cal_items + eval_items, run, lambda item, row, seed, _run: self._sem1_pred(item, row, seed, run, delta=0.0))
            live_v6 = self._rows_for(cal_items + eval_items, run, lambda item, row, seed, _run: self._v6_pred(item, row, run=run))
            write_json(run_dir / "delta0_identity.json", self._selection_identity(live_v6, d0_rows, extra={"selected_delta": delta, "vs": "live_v6_same_weights"}))
            eval_d0 = [row for row in d0_rows if any(row["question_id"] == item.question_id for item in eval_items)]
            v6_hist = [v6_index[key] for key in ((str(row["question_id"]), int(row["seed"])) for row in eval_d0) if key in v6_index]
            if v6_hist:
                write_json(run_dir / "delta0_identity_eval_historical.json", self._selection_identity(v6_hist, eval_d0, extra={"vs": "historical_v6_eval_oof"}))
            for pred in self._rows_for(eval_items, run, lambda item, row, seed, _run: self._sem1_pred(item, row, seed, run, delta=delta)):
                oof.append(pred)
        self._record("SEM1", oof, self._mkdir("sem1"))
        v6 = self.predictions.get("V6-K5") or []
        delta = paired_delta(v6, oof, n_boot=self._n_boot(), seed=self.config.seed) if v6 else {}
        write_json(self.output_dir / "sem1" / "bootstrap_vs_v6.json", delta)
        sem_hn = float((self.summaries.get("SEM1") or {}).get("hn@1") or 0.0)
        v6_hn = float((self.summaries.get("V6-K5") or {}).get("hn@1") or 0.0)
        self.sem1_hn_worse = sem_hn > v6_hn + self.config.epsilon_hn
        mean_d = float(delta.get("mean_delta") or 0.0)
        low = delta.get("ci95_low")
        self.sem1_gate = bool(mean_d > 0 and low is not None and float(low) > 0 and not self.sem1_hn_worse)
        seed_deltas = self._seed_mean_deltas(v6, oof) if v6 else {}
        n_pos_seeds = sum(1 for value in seed_deltas.values() if float(value) > 0)
        n_bad_shortlist = sum(1 for row in oof if row.get("shortlist_ok") is False)
        write_json(
            self.output_dir / "sem1" / "gate.json",
            {
                "sem1_gate": self.sem1_gate,
                "hn_worse": self.sem1_hn_worse,
                "delta": delta,
                "selected_delta": {str(k): v for k, v in selected.items()},
                "seed_deltas": seed_deltas,
                "gain_not_one_seed": n_pos_seeds >= 2 if seed_deltas else None,
                "n_positive_seeds": n_pos_seeds,
                "n_bad_shortlist": n_bad_shortlist,
                "shortlist_invariants_ok": n_bad_shortlist == 0,
                "extras_recorded_not_anded_into_gate": True,
            },
        )
        write_json(self.output_dir / "sem1" / "transitions.json", transitions(v6, oof) if v6 else {})
        n_zero = sum(1 for value in self.sem1_delta.values() if float(value) == 0.0)
        self.progress.log("SEM1 OOF", gate=self.sem1_gate, n_delta0=n_zero, mean_delta=mean_d)

    def _lex_pred(self, item, b0_row, seed: int, run: int, *, lambda_m: float, lambda_t: float) -> dict[str, Any]:
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        scores = lexcore_scores(item, b0_row, qa_by_chunk, k=self.config.primary_k, min_overlap=self.config.source_overlap, lambda_m=lambda_m, lambda_t=lambda_t)
        ranked, fused, tied = isolate_with_b0_tiebreak(item, b0_row, scores, k=self.config.primary_k)
        return self._predict_ok(item, b0_row, fused, qa_by_chunk, ranked, "LEXCORE", {"tie": tied, "lambda_m": lambda_m, "lambda_t": lambda_t})

    def _semcore_pred(self, item, b0_row, seed: int, run: int, *, lambda_m: float, lambda_s: float, method: str = "SEMCORE") -> dict[str, Any]:
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        scores = semcore_scores(item, b0_row, qa_by_chunk, self._sem_rel(item, seed), k=self.config.primary_k, min_overlap=self.config.source_overlap, lambda_m=lambda_m, lambda_s=lambda_s)
        ranked, fused, tied = isolate_with_b0_tiebreak(item, b0_row, scores, k=self.config.primary_k)
        pack = self._pack(item, seed)
        return self._predict_ok(
            item,
            b0_row,
            fused,
            qa_by_chunk,
            ranked,
            method,
            {
                "tie": tied,
                "lambda_m": lambda_m,
                "lambda_s": lambda_s,
                "semantic_available": bool(pack.get("semantic_available")),
                "best_triple": pack.get("best_triple") or [],
                "consensus_conf": pack.get("consensus_conf"),
                "cohesion": pack.get("cohesion"),
            },
        )

    def stage_lexcore(self) -> None:
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        chosen = {}
        grid = [{"lambda_m": a, "lambda_t": b} for a, b in product(self.config.lexcore_majority_grid, self.config.lexcore_token_grid)]
        for run in range(n_runs):
            self.progress.log(f"LEXCORE outer run {run}/{max(n_runs - 1, 0)}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            tried = []
            for index, params in enumerate(grid, start=1):
                rows = self._rows_for(cal_items, run, lambda item, row, seed, _run, p=params: self._lex_pred(item, row, seed, run, lambda_m=float(p["lambda_m"]), lambda_t=float(p["lambda_t"])))
                summary = summarize_rows(rows)
                tried.append({**params, "cal_token_f1": mean_token_f1_across_seeds(rows), "cal_hn@1": float(summary.get("hn@1") or 0.0)})
                self.progress.count(index, len(grid), item=f"lexcore_run{run}")
            choice = choose_max_f1(tried, prefer_smaller=("lambda_m", "lambda_t"))
            sel = choice["selected"]
            self.lex_weights[run] = {"lambda_m": float(sel["lambda_m"]), "lambda_t": float(sel["lambda_t"])}
            chosen[run] = sel
            write_json(self._mkdir("lexcore", f"run{run}") / "selected.json", choice)
            oof.extend(self._rows_for(eval_items, run, lambda item, row, seed, _run: self._lex_pred(item, row, seed, run, lambda_m=float(sel["lambda_m"]), lambda_t=float(sel["lambda_t"]))))
        self._record("LEXCORE", oof, self._mkdir("lexcore"))
        write_jsonl(self.output_dir / "lexcore" / "oof_predictions.jsonl", oof)
        write_json(self.output_dir / "lexcore" / "weights_by_run.json", {str(k): v for k, v in chosen.items()})
        self._write_tie_rate("LEXCORE", oof)
        v6 = self.predictions.get("V6-K5") or []
        if v6:
            write_json(self.output_dir / "lexcore" / "bootstrap_vs_v6.json", paired_delta(v6, oof, n_boot=self._n_boot(), seed=self.config.seed))

    def stage_semcore(self) -> None:
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        chosen = {}
        grid = [{"lambda_m": a, "lambda_s": b} for a, b in product(self.config.semcore_majority_grid, self.config.semcore_sem_grid)]
        for run in range(n_runs):
            self.progress.log(f"SEMCORE outer run {run}/{max(n_runs - 1, 0)}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            tried = []
            for index, params in enumerate(grid, start=1):
                rows = self._rows_for(cal_items, run, lambda item, row, seed, _run, p=params: self._semcore_pred(item, row, seed, run, lambda_m=float(p["lambda_m"]), lambda_s=float(p["lambda_s"])))
                summary = summarize_rows(rows)
                tried.append({**params, "cal_token_f1": mean_token_f1_across_seeds(rows), "cal_hn@1": float(summary.get("hn@1") or 0.0)})
                self.progress.count(index, len(grid), item=f"semcore_run{run}")
            choice = choose_max_f1(tried, prefer_smaller=("lambda_m", "lambda_s"))
            sel = choice["selected"]
            self.semcore_weights[run] = {"lambda_m": float(sel["lambda_m"]), "lambda_s": float(sel["lambda_s"])}
            chosen[run] = sel
            write_json(self._mkdir("semcore", f"run{run}") / "selected.json", choice)
            oof.extend(self._rows_for(eval_items, run, lambda item, row, seed, _run: self._semcore_pred(item, row, seed, run, lambda_m=float(sel["lambda_m"]), lambda_s=float(sel["lambda_s"]))))
        self._record("SEMCORE", oof, self._mkdir("semcore"))
        write_jsonl(self.output_dir / "semcore" / "oof_predictions.jsonl", oof)
        write_json(self.output_dir / "semcore" / "weights_by_run.json", {str(k): v for k, v in chosen.items()})
        self._write_tie_rate("SEMCORE", oof)
        lex = self.predictions.get("LEXCORE") or []
        v6 = self.predictions.get("V6-K5") or []
        core3 = self.predictions.get("CORE3-historical") or []
        if lex:
            write_json(self.output_dir / "semcore" / "bootstrap_vs_lexcore.json", paired_delta(lex, oof, n_boot=self._n_boot(), seed=self.config.seed))
        if v6:
            write_json(self.output_dir / "semcore" / "bootstrap_vs_v6.json", paired_delta(v6, oof, n_boot=self._n_boot(), seed=self.config.seed))
        if core3 and lex:
            write_json(self.output_dir / "semcore" / "bootstrap_core3_vs_lexcore.json", paired_delta(lex, core3, n_boot=self._n_boot(), seed=self.config.seed))

    def _write_tie_rate(self, name: str, rows: list[dict[str, Any]]) -> None:
        n = len(rows)
        n_tie = sum(1 for row in rows if row.get("tie"))
        write_json(self.output_dir / "analysis" / f"tie_rate_{name.lower()}.json", {"system": name, "n": n, "n_tie": n_tie, "tie_rate": n_tie / max(n, 1)})

    def stage_complement(self) -> None:
        v6 = {(str(r["question_id"]), int(r["seed"])): r for r in self.predictions.get("V6-K5") or []}
        lex = {(str(r["question_id"]), int(r["seed"])): r for r in self.predictions.get("LEXCORE") or []}
        sem = {(str(r["question_id"]), int(r["seed"])): r for r in self.predictions.get("SEMCORE") or []}
        sem1 = {(str(r["question_id"]), int(r["seed"])): r for r in self.predictions.get("SEM1") or []}
        rows = []
        buckets = defaultdict(int)
        lex_sem_flip = {"lex_wrong_sem_right": [], "lex_right_sem_wrong": []}
        v6_sem1_flip = {"v6_wrong_sem1_right": [], "v6_right_sem1_wrong": []}
        conf_rows = []
        for key, v6_row in v6.items():
            lex_row = lex.get(key) or {}
            sem_row = sem.get(key) or {}
            sem1_row = sem1.get(key) or {}
            pack = self.packs.get(key) or {}
            v6_ok = str(v6_row.get("selected_chunk_label") or "") == "positive"
            lex_ok = str(lex_row.get("selected_chunk_label") or "") == "positive"
            sem_ok = str(sem_row.get("selected_chunk_label") or "") == "positive"
            sem1_ok = str(sem1_row.get("selected_chunk_label") or "") == "positive"
            if v6_ok and not sem_ok:
                bucket = "V6 win / SEMCORE lose"
            elif sem_ok and not v6_ok:
                bucket = "SEMCORE win / V6 lose"
            elif v6_ok and sem_ok:
                bucket = "both win"
            else:
                bucket = "both lose"
            buckets[bucket] += 1
            entry = {
                "question_id": key[0],
                "seed": key[1],
                "aspect": v6_row.get("aspect"),
                "v6_correct": v6_ok,
                "lexcore_correct": lex_ok,
                "semcore_correct": sem_ok,
                "consensus_conf": pack.get("consensus_conf"),
                "cohesion": pack.get("cohesion"),
                "b0_margin": pack.get("b0_margin"),
                "v6_answer": v6_row.get("qa_answer"),
                "lex_answer": lex_row.get("qa_answer"),
                "sem_answer": sem_row.get("qa_answer"),
                "bucket": bucket,
            }
            rows.append(entry)
            conf_rows.append({"consensus_conf": pack.get("consensus_conf"), "cohesion": pack.get("cohesion"), "b0_margin": pack.get("b0_margin"), "bucket": bucket, "aspect": v6_row.get("aspect")})
            sample = {
                "question_id": key[0],
                "seed": key[1],
                "aspect": v6_row.get("aspect"),
                "lex_answer": lex_row.get("qa_answer"),
                "sem_answer": sem_row.get("qa_answer"),
                "v6_answer": v6_row.get("qa_answer"),
                "sem1_answer": sem1_row.get("qa_answer"),
                "selected_source_lex": lex_row.get("selected_chunk_label"),
                "selected_source_sem": sem_row.get("selected_chunk_label"),
                "selected_source_v6": v6_row.get("selected_chunk_label"),
                "selected_source_sem1": sem1_row.get("selected_chunk_label"),
                **{f"lex_sem_{k}": v for k, v in self._answer_pair_diag(lex_row, sem_row, pack).items()},
                **{f"v6_sem1_{k}": v for k, v in self._answer_pair_diag(v6_row, sem1_row, pack).items()},
            }
            if (not lex_ok) and sem_ok and len(lex_sem_flip["lex_wrong_sem_right"]) < 50:
                lex_sem_flip["lex_wrong_sem_right"].append(sample)
            if lex_ok and (not sem_ok) and len(lex_sem_flip["lex_right_sem_wrong"]) < 50:
                lex_sem_flip["lex_right_sem_wrong"].append(sample)
            if (not v6_ok) and sem1_ok and len(v6_sem1_flip["v6_wrong_sem1_right"]) < 50:
                v6_sem1_flip["v6_wrong_sem1_right"].append(sample)
            if v6_ok and (not sem1_ok) and len(v6_sem1_flip["v6_right_sem1_wrong"]) < 50:
                v6_sem1_flip["v6_right_sem1_wrong"].append(sample)
        write_json(self.output_dir / "analysis" / "core_complementarity.json", {"buckets": dict(buckets), "n": len(rows)})
        write_jsonl(self.output_dir / "analysis" / "core_complementarity.jsonl", rows)
        write_json(self.output_dir / "analysis" / "consensus_confidence.json", {"n": len(conf_rows), "note": "diagnostic only; not an ADAPT feature beyond ConsensusConf"})
        write_jsonl(self.output_dir / "analysis" / "consensus_confidence.jsonl", conf_rows)
        write_json(self.output_dir / "analysis" / "error_buckets.json", {**lex_sem_flip, **v6_sem1_flip})
        self._write_b0_margin_diagnostic(rows)
        self.progress.log("complementarity written", **{k.replace(" ", "_"): v for k, v in buckets.items()})

    def _write_b0_margin_diagnostic(self, rows: list[dict[str, Any]]) -> None:
        with_margin = [row for row in rows if row.get("b0_margin") is not None]
        if not with_margin:
            write_json(
                self.output_dir / "analysis" / "b0_margin_diagnostic.json",
                {"diagnostic_only": True, "not_used_in_adapt1": True, "n": 0},
            )
            return
        margins = sorted(float(row["b0_margin"]) for row in with_margin)
        median = margins[len(margins) // 2]

        def mean_margin(bucket: str) -> float | None:
            values = [float(row["b0_margin"]) for row in with_margin if row.get("bucket") == bucket]
            return float(sum(values) / len(values)) if values else None

        def rate(subset: list[dict[str, Any]], bucket: str) -> float | None:
            if not subset:
                return None
            return float(sum(1 for row in subset if row.get("bucket") == bucket) / len(subset))

        weak = [row for row in with_margin if float(row["b0_margin"]) <= median]
        strong = [row for row in with_margin if float(row["b0_margin"]) > median]
        write_json(
            self.output_dir / "analysis" / "b0_margin_diagnostic.json",
            {
                "diagnostic_only": True,
                "not_used_in_adapt1": True,
                "n": len(with_margin),
                "median_margin": median,
                "mean_margin_by_bucket": {
                    "V6 win / SEMCORE lose": mean_margin("V6 win / SEMCORE lose"),
                    "SEMCORE win / V6 lose": mean_margin("SEMCORE win / V6 lose"),
                    "both win": mean_margin("both win"),
                    "both lose": mean_margin("both lose"),
                },
                "weak_n": len(weak),
                "strong_n": len(strong),
                "weak_margin_semcore_win_rate": rate(weak, "SEMCORE win / V6 lose"),
                "strong_margin_semcore_win_rate": rate(strong, "SEMCORE win / V6 lose"),
                "weak_margin_v6_win_rate": rate(weak, "V6 win / SEMCORE lose"),
                "strong_margin_v6_win_rate": rate(strong, "V6 win / SEMCORE lose"),
            },
        )

    def _inner_semcore_weights(self, run: int) -> dict[str, Any]:
        spec = self.splits["runs"][int(run)]
        train_folds = [int(fold) for fold in spec["train"]]
        assignment = self.splits["assignment"]
        by_fold: dict[int, list] = {fold: [] for fold in train_folds}
        for item in self.unlocked:
            fold = int(assignment[item.document_id])
            if fold in by_fold:
                by_fold[fold].append(item)
        train_ids = {item.question_id for items in by_fold.values() for item in items}
        eval_ids = {item.question_id for item in records_for_role(self.unlocked, self.splits, run=run, role="eval")}
        cal_ids = {item.question_id for item in records_for_role(self.unlocked, self.splits, run=run, role="calibration")}
        leakage = sorted(train_ids & (eval_ids | cal_ids))
        grid = [{"lambda_m": a, "lambda_s": b} for a, b in product(self.config.semcore_majority_grid, self.config.semcore_sem_grid)]
        scored = []
        for index, params in enumerate(grid, start=1):
            fold_f1 = []
            for held in train_folds:
                items = by_fold[held]
                rows = self._rows_for(items, run, lambda item, row, seed, _run, p=params: self._semcore_pred(item, row, seed, run, lambda_m=float(p["lambda_m"]), lambda_s=float(p["lambda_s"]), method="SEMCORE-inner"))
                fold_f1.append(mean_token_f1_across_seeds(rows) if rows else 0.0)
            scored.append({**params, "mean_inner_f1": float(sum(fold_f1) / max(len(fold_f1), 1)), "fold_f1": fold_f1})
            self.progress.count(index, len(grid), item=f"adapt_inner_run{run}")
        scored.sort(key=lambda row: (-float(row["mean_inner_f1"]), float(row["lambda_m"]), float(row["lambda_s"])))
        selected = scored[0] if scored else {"lambda_m": 0.25, "lambda_s": 0.0}
        return {"selected": selected, "tried": scored, "train_folds": train_folds, "eval_leakage": leakage, "ok": not leakage}

    def _adapt_pred(self, item, b0_row, seed: int, run: int, *, lambda_m: float, lambda_s: float, tau: float, tau_name: str) -> dict[str, Any]:
        sem_pred = self._semcore_pred(item, b0_row, seed, run, lambda_m=lambda_m, lambda_s=lambda_s, method="SEMCORE-adapt")
        v6_pred = self._v6_pred(item, b0_row, run=run)
        pack = self._pack(item, seed)
        selected_sem = int(sem_pred.get("selected_chunk_id") or -1)
        available = routing_available(
            adapt_answers_ok=bool(pack.get("adapt_answers_ok")),
            selected_cid=selected_sem,
            best_triple=pack.get("best_triple") or [],
            consensus_conf=pack.get("consensus_conf"),
        )
        use_sem = adapt_use_semcore(confidence=pack.get("consensus_conf"), tau=tau, available=available)
        chosen = sem_pred if use_sem else v6_pred
        out = dict(chosen)
        out["ranking_method"] = "ADAPT1"
        out["adapt_used_semcore"] = bool(use_sem)
        out["routing_confidence_available"] = bool(available)
        out["tau_name"] = tau_name
        out["consensus_conf"] = pack.get("consensus_conf")
        out["best_triple"] = pack.get("best_triple") or []
        return out

    def stage_adapt1(self) -> None:
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            self.progress.log(f"ADAPT1 outer run {run}/{max(n_runs - 1, 0)}")
            inner = self._inner_semcore_weights(run)
            write_json(self._mkdir("adapt1", f"run{run}") / "inner_semcore_weights.json", inner)
            sel_w = inner["selected"]
            weights = {"lambda_m": float(sel_w["lambda_m"]), "lambda_s": float(sel_w["lambda_s"])}
            self.adapt_inner_weights[run] = weights
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_hn = self._live_v6_hn(cal_items, run)
            confidences = []
            for seed in self.config.model_seeds:
                for item in cal_items:
                    pack = self._pack(item, seed)
                    if pack.get("adapt_answers_ok") and pack.get("consensus_conf") is not None:
                        confidences.append(float(pack["consensus_conf"]))
            tau_grid = [{"tau": NEVER_TAU, "tau_name": "NEVER"}, {"tau": ALWAYS_TAU, "tau_name": "ALWAYS"}]
            if confidences:
                for q in self.config.adapt_quantiles:
                    tau_grid.append({"tau": float(np.quantile(confidences, q)), "tau_name": f"Q{int(round(q * 100))}"})
            else:
                for q in self.config.adapt_quantiles:
                    tau_grid.append({"tau": NEVER_TAU, "tau_name": f"Q{int(round(q * 100))}"})
            tried = []
            for index, params in enumerate(tau_grid, start=1):
                rows = self._rows_for(
                    cal_items,
                    run,
                    lambda item, row, seed, _run, p=params: self._adapt_pred(item, row, seed, run, lambda_m=weights["lambda_m"], lambda_s=weights["lambda_s"], tau=p["tau"], tau_name=p["tau_name"]),
                )
                summary = summarize_rows(rows)
                hn = float(summary.get("hn@1") or 0.0)
                tried.append(
                    {
                        "tau_name": params["tau_name"],
                        "tau": None if params["tau_name"] in {"NEVER", "ALWAYS"} else float(params["tau"]),
                        "tau_sort": finite_tau(params["tau"]),
                        "cal_token_f1": mean_token_f1_across_seeds(rows),
                        "cal_hn@1": hn,
                        "admissible": hn <= v6_hn + self.config.epsilon_hn,
                    }
                )
                self.progress.count(index, len(tau_grid), item=f"adapt_tau_run{run}")
            admissible = [row for row in tried if row["admissible"]]
            pool = admissible or tried
            pool = sorted(pool, key=lambda row: (-float(row["cal_token_f1"]), -float(row["tau_sort"]), float(row["cal_hn@1"]), row["tau_name"]))
            selected = dict(pool[0])
            if not admissible:
                selected["forced_safeguard_fallback"] = True
            payload = {"selected": selected, "tried": tried, "v6_cal_hn@1": v6_hn, "n_confidences": len(confidences)}
            write_json(self._mkdir("adapt1", f"run{run}") / "cal_threshold.json", payload)
            self.adapt_tau[run] = selected
            tau_name = str(selected["tau_name"])
            tau_value = NEVER_TAU if tau_name == "NEVER" else ALWAYS_TAU if tau_name == "ALWAYS" else float(selected["tau"])
            never_rows = self._rows_for(eval_items, run, lambda item, row, seed, _run: self._adapt_pred(item, row, seed, run, lambda_m=weights["lambda_m"], lambda_s=weights["lambda_s"], tau=NEVER_TAU, tau_name="NEVER"))
            write_json(self._mkdir("adapt1", f"run{run}") / "never_identity.json", self._adapt_never_identity(eval_items, run, never_rows))
            always_rows = self._rows_for(eval_items, run, lambda item, row, seed, _run: self._adapt_pred(item, row, seed, run, lambda_m=weights["lambda_m"], lambda_s=weights["lambda_s"], tau=ALWAYS_TAU, tau_name="ALWAYS"))
            write_json(self._mkdir("adapt1", f"run{run}") / "always_identity.json", self._adapt_always_identity(eval_items, run, weights, always_rows))
            oof.extend(self._rows_for(eval_items, run, lambda item, row, seed, _run: self._adapt_pred(item, row, seed, run, lambda_m=weights["lambda_m"], lambda_s=weights["lambda_s"], tau=tau_value, tau_name=tau_name)))
        self._record("ADAPT1", oof, self._mkdir("adapt1"))
        write_jsonl(self.output_dir / "adapt1" / "oof_predictions.jsonl", oof)
        v6 = self.predictions.get("V6-K5") or []
        if v6:
            write_json(self.output_dir / "adapt1" / "bootstrap_vs_v6.json", paired_delta(v6, oof, n_boot=self._n_boot(), seed=self.config.seed))
        n_never = sum(1 for spec in self.adapt_tau.values() if spec.get("tau_name") == "NEVER")
        self.progress.log("ADAPT1 OOF", n_never=n_never)

    def _adapt_never_identity(self, eval_items, run: int, never_rows: list[dict[str, Any]]) -> dict[str, Any]:
        v6_rows = self._rows_for(eval_items, run, lambda item, row, seed, _run: self._v6_pred(item, row, run=run))
        return self._selection_identity(v6_rows, never_rows, extra={"tau": "NEVER"})

    def _adapt_always_identity(self, eval_items, run: int, weights: dict[str, float], always_rows: list[dict[str, Any]]) -> dict[str, Any]:
        sem_rows = self._rows_for(
            eval_items,
            run,
            lambda item, row, seed, _run: self._semcore_pred(item, row, seed, run, lambda_m=float(weights["lambda_m"]), lambda_s=float(weights["lambda_s"])),
        )
        v6_rows = self._rows_for(eval_items, run, lambda item, row, seed, _run: self._v6_pred(item, row, run=run))
        sem_index = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in sem_rows}
        v6_index = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in v6_rows}
        n = 0
        same = 0
        n_sem = 0
        n_v6 = 0
        for row in always_rows:
            key = (str(row["question_id"]), int(row.get("seed") or 0))
            use_sem = bool(row.get("routing_confidence_available"))
            want = sem_index.get(key) if use_sem else v6_index.get(key)
            if want is None:
                continue
            n += 1
            n_sem += int(use_sem)
            n_v6 += int(not use_sem)
            if str(row.get("selected_chunk_id")) == str(want.get("selected_chunk_id")) and str(row.get("qa_answer") or "") == str(want.get("qa_answer") or ""):
                same += 1
        return {
            "ok": bool(n) and same == n,
            "n": n,
            "same": same,
            "n_semcore_when_available": n_sem,
            "n_v6_fallback": n_v6,
            "tau": "ALWAYS",
        }

    def _semctx_pred(self, item, b0_row, seed: int, run: int, *, delta_s: float, delta_c: float) -> dict[str, Any]:
        sem1 = self._sem1_pred(item, b0_row, seed, run, delta=delta_s)
        qa_by_chunk, _ranked_v6, v6_short = self._v6_short(item, b0_row, run)
        pack = self._pack(item, seed)
        support = pack.get("support_rel") or {}
        if float(delta_c) == 0.0 or not support:
            out = dict(sem1)
            out["ranking_method"] = "SEMCTX1"
            out["delta_c"] = 0.0
            out["fallback_sem1"] = True
            return out
        sem1_scores_map = sem1_scores(v6_short, self._sem_rel(item, seed), delta=float(delta_s)) if not sem1.get("fallback_v6") else v6_short
        scores = semctx_scores(sem1_scores_map, {int(k): float(v) for k, v in support.items()}, delta_c=float(delta_c))
        ranked, fused, _tied = isolate_with_b0_tiebreak(item, b0_row, scores, k=self.config.primary_k)
        return self._predict_ok(item, b0_row, fused, qa_by_chunk, ranked, "SEMCTX1", {"delta_s": delta_s, "delta_c": delta_c, "fallback_sem1": False, "support_rel": support})

    def stage_semctx1(self) -> None:
        if not self.sem1_gate and not self.config.force_semctx:
            self.progress.log("SEMCTX1 skipped: SEM1 did not pass primary gate (use --force-semctx for exploratory)")
            return
        self.progress.log("extracting local evidence and query-answer embeddings")
        encoder = self._encoder()
        by_qid = self._by_qid()
        questions = {item.question_id: item.question for item in self.unlocked}
        encoder.embed_many(questions.values(), store="query")
        evidence_texts = []
        qa_texts = []
        n = 0
        total = len(self.packs)
        for (qid, seed), pack in self.packs.items():
            item = by_qid.get(qid)
            if item is None:
                continue
            b0_row = self._b0_seed(int(seed)).get(qid)
            if b0_row is None:
                continue
            q_vec = encoder.embed_one(semantic_normalize(item.question), store="query")
            top = pack.get("top5") or top_k_ids(b0_row, self.config.primary_k)
            base = {}
            ans_fit = {}
            local = {}
            for cid in top:
                qa = self.qa_store.get((qid, int(cid))) or {}
                answer = qa_answer_text(qa)
                chunk = item.chunk_by_id(int(cid))
                evidence = local_evidence_sentence(
                    chunk.text,
                    start_char=int(qa.get("qa_chunk_start_char") or 0),
                    end_char=int(qa.get("qa_chunk_end_char") or 0),
                    predicted_answer=answer,
                )
                local[int(cid)] = evidence
                l_vec = encoder.embed_one(evidence.get("text") or "", store="local_evidence") if evidence.get("text") else None
                qa_text = query_answer_text(item.question, answer)
                qa_vec = encoder.embed_one(qa_text, store="query_answer")
                evidence_texts.append(evidence.get("text") or "")
                qa_texts.append(qa_text)
                if l_vec is None:
                    base[int(cid)] = 0.0
                    ans_fit[int(cid)] = 0.0
                else:
                    base[int(cid)] = cosine(q_vec, l_vec)
                    ans_fit[int(cid)] = cosine(qa_vec, l_vec)
            ac_vecs = {}
            ctx_mean = {}
            for cid in top:
                blob = semantic_normalize(f"{qa_answer_text(self.qa_store.get((qid, int(cid))))} {local.get(int(cid), {}).get('text') or ''}")
                if blob:
                    ac_vecs[int(cid)] = encoder.embed_one(blob, store="answer_local")
            for cid, vec in ac_vecs.items():
                others = [cosine(vec, ac_vecs[other]) for other in ac_vecs if other != cid]
                ctx_mean[int(cid)] = float(sum(others) / len(others)) if others else 0.0
            pair_sims = [
                cosine(ac_vecs[a], ac_vecs[b])
                for i, a in enumerate(sorted(ac_vecs))
                for b in sorted(ac_vecs)[i + 1 :]
            ]
            gain = support_gain(base, ans_fit, top)
            rel = support_rel(gain)
            pack["base_fit"] = {str(k): float(v) for k, v in base.items()}
            pack["answer_fit"] = {str(k): float(v) for k, v in ans_fit.items()}
            pack["support_gain"] = {str(k): float(v) for k, v in gain.items()}
            pack["support_rel"] = {str(k): float(v) for k, v in rel.items()}
            pack["local_evidence"] = {str(k): v for k, v in local.items()}
            pack["ctx_ctx_mean"] = float(sum(pair_sims) / len(pair_sims)) if pair_sims else None
            pack["ctx_ctx_rel"] = {str(k): float(v) for k, v in support_rel(ctx_mean).items()}
            pack["ctx_ctx_diagnostic_only"] = True
            n += 1
            self.progress.count(n, max(total, 1), item="semctx_extract")
        ctx_rows = [
            {
                "question_id": qid,
                "seed": seed,
                "ctx_ctx_mean": pack.get("ctx_ctx_mean"),
                "ctx_ctx_rel": pack.get("ctx_ctx_rel"),
            }
            for (qid, seed), pack in self.packs.items()
        ]
        ctx_means = [float(row["ctx_ctx_mean"]) for row in ctx_rows if row.get("ctx_ctx_mean") is not None]
        write_json(
            self.output_dir / "analysis" / "context_to_context.json",
            {
                "diagnostic_only": True,
                "not_used_in_semctx1": True,
                "n": len(ctx_rows),
                "mean_pairwise": float(sum(ctx_means) / len(ctx_means)) if ctx_means else None,
            },
        )
        write_jsonl(self.output_dir / "analysis" / "context_to_context.jsonl", ctx_rows)
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            delta_s = float(self.sem1_delta.get(run, 0.0))
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_hn = self._live_v6_hn(cal_items, run)

            def predict(params, items=cal_items, run_id=run, frozen_ds=delta_s):
                return self._rows_for(items, run_id, lambda item, row, seed, _run: self._semctx_pred(item, row, seed, run_id, delta_s=frozen_ds, delta_c=float(params["delta_c"])))

            choice = choose_admissible_cal(
                cal_items,
                grid=[{"delta_c": delta} for delta in self.config.context_delta_grid],
                predict_rows=predict,
                hn_ref=v6_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("delta_c",),
                progress=self.progress,
                item_name=f"semctx_run{run}",
            )
            write_json(self._mkdir("semctx1", f"run{run}") / "selected_delta.json", choice)
            delta_c = float(choice["selected"]["delta_c"])
            oof.extend(self._rows_for(eval_items, run, lambda item, row, seed, _run: self._semctx_pred(item, row, seed, run, delta_s=delta_s, delta_c=delta_c)))
        self._record("SEMCTX1", oof, self._mkdir("semctx1"))
        sem1 = self.predictions.get("SEM1") or []
        if sem1:
            write_json(self.output_dir / "semctx1" / "bootstrap_vs_sem1.json", paired_delta(sem1, oof, n_boot=self._n_boot(), seed=self.config.seed))
            write_json(self.output_dir / "semctx1" / "transitions.json", transitions(sem1, oof))
        self.progress.log("SEMCTX1 OOF")

    def _selection_identity(self, left, right, extra=None) -> dict[str, Any]:
        index_left = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in left}
        index_right = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in right}
        keys = sorted(set(index_left) & set(index_right))
        n_chunk = sum(str(index_left[key].get("selected_chunk_id")) == str(index_right[key].get("selected_chunk_id")) for key in keys)
        n_ans = sum(str(index_left[key].get("qa_answer") or "") == str(index_right[key].get("qa_answer") or "") for key in keys)
        payload = {"ok": bool(keys) and n_chunk == len(keys) and n_ans == len(keys), "n": len(keys), "same_chunk": n_chunk, "same_answer": n_ans}
        if extra:
            payload.update(extra)
        return payload

    def stage_compare(self) -> None:
        b0 = self.predictions.get("B0-live") or []
        v6 = self.predictions.get("V6-K5") or []
        table = []
        deltas = {}
        trans = {}
        admissible = []
        allowed = set(PRIMARY_FINAL_SYSTEMS)
        b0_hn = float((self.summaries.get("B0-live") or {}).get("hn@1") or 0.0)
        for name, rows in self.predictions.items():
            summary = self.summaries.get(name) or summarize_rows(rows)
            design = seed_design(rows)
            row = {"system": name, **{k: summary.get(k) for k in ("n", "token_f1", "exact_match", "hit@1", "mrr", "hn@1", "ambiguous_at_1_original", "gsrn_at_1_original")}, **design}
            if name == "B0-live":
                row["admissible"] = True
                table.append(row)
                continue
            delta = paired_delta(b0, rows, n_boot=self._n_boot(), seed=self.config.seed)
            deltas[name] = delta
            trans[name] = transitions(b0, rows)
            hn_ok = float(summary.get("hn@1") or 0.0) <= b0_hn + self.config.epsilon_hn
            auto = name in allowed and bool(design.get("three_seed") or self.config.smoke) and hn_ok
            if name == "SEM1" and not self.sem1_gate:
                auto = False
            if name in {"LEXCORE", "SEMCORE", "ADAPT1", "SEMCTX1", "CORE3-historical"}:
                auto = False
            row["admissible"] = auto
            row["hn_safeguard"] = hn_ok
            if auto:
                admissible.append(name)
            table.append(row)
        selected = "V6-K5" if "V6-K5" in self.predictions else "B0-live"
        if self.sem1_gate and "SEM1" in admissible:
            selected = "SEM1"
        lex = self.predictions.get("LEXCORE") or []
        semc = self.predictions.get("SEMCORE") or []
        sem1 = self.predictions.get("SEM1") or []
        payload = {
            "table": table,
            "deltas": deltas,
            "transitions": trans,
            "selected": selected,
            "admissible": admissible,
            "b0_token_f1": (self.summaries.get("B0-live") or {}).get("token_f1"),
            "v6_token_f1": (self.summaries.get("V6-K5") or {}).get("token_f1"),
            "sem1_token_f1": (self.summaries.get("SEM1") or {}).get("token_f1"),
            "lexcore_token_f1": (self.summaries.get("LEXCORE") or {}).get("token_f1"),
            "semcore_token_f1": (self.summaries.get("SEMCORE") or {}).get("token_f1"),
            "adapt1_token_f1": (self.summaries.get("ADAPT1") or {}).get("token_f1"),
            "semctx1_token_f1": (self.summaries.get("SEMCTX1") or {}).get("token_f1"),
            "sem1_vs_v6": paired_delta(v6, sem1, n_boot=self._n_boot(), seed=self.config.seed) if v6 and sem1 else {},
            "semcore_vs_lexcore": paired_delta(lex, semc, n_boot=self._n_boot(), seed=self.config.seed) if lex and semc else {},
            "sem1_gate": self.sem1_gate,
            "internal_test_locked": True,
            "systems": list(self.predictions),
        }
        write_json(self.output_dir / "analysis" / "compare.json", payload)
        write_json(self.output_dir / "analysis" / "selection.json", {"selected": selected, "admissible": admissible, "sem1_gate": self.sem1_gate, "IN_SAMPLE_CONTAMINATED": False})
        write_json(self.output_dir / "analysis" / "transitions.json", trans)
        self._compare_payload = payload
        self._write_aspect()
        self.progress.log("compare done", selected=selected, sem1_gate=self.sem1_gate)

    def _write_aspect(self) -> None:
        payload = {}
        wanted = ("V6-K5", "SEM1", "LEXCORE", "SEMCORE", "ADAPT1", "SEMCTX1", "B0-live")
        for name in wanted:
            rows = self.predictions.get(name) or []
            if not rows:
                continue
            by = defaultdict(list)
            for row in rows:
                by[str(row.get("aspect") or "none")].append(row)
            payload[name] = {asp: summarize_rows(group) for asp, group in sorted(by.items())}
        write_json(self.output_dir / "analysis" / "aspect_metrics.json", payload)

    def stage_freeze(self) -> None:
        payload = getattr(self, "_compare_payload", None) or {}
        selected = payload.get("selected") or "V6-K5"
        enc_stats = self.encoder.stats() if self.encoder is not None else {}
        write_json(
            self.output_dir / "analysis" / "efficiency.json",
            {
                "qa_cache": self.cache.stats(),
                "encoder": enc_stats,
                "n_packs": len(self.packs),
                "note": "V4.1/V4.2 QA may be cached; MiniLM is frozen",
            },
        )
        self._mkdir("final")
        write_json(self.output_dir / "final" / "model_selection.json", {"selected": selected, "sem1_gate": self.sem1_gate, "admissible": payload.get("admissible")})
        write_json(
            self.output_dir / "final" / "freeze_manifest.json",
            {
                "final_architecture": selected,
                "v6_retuned": False,
                "v6_weights": "V4.1 per-run",
                "primary_k": self.config.primary_k,
                "internal_test": "locked",
                "sem1_gate": self.sem1_gate,
                "encoder": self.config.semantic_model,
                "encoder_frozen": True,
                "b0_final": "NOT trained",
                "u80": "not in V4.3",
                "k10": "not in V4.3",
                "attention": "not in V4.3",
                "adapt1": "exploratory architecture-discovery; not confirmatory",
            },
        )
        self.progress.log("freeze written; INTERNAL_TEST still locked", selected=selected)

    def stage_report(self) -> None:
        payload = getattr(self, "_compare_payload", None)
        if payload is None:
            path = self.output_dir / "analysis" / "compare.json"
            payload = load_json(path) if path.exists() else {"internal_test_locked": True}
        write_report(self.output_dir, payload)
        self.progress.log("report written", path=str(self.output_dir / "reports" / "results_v4_3.md"))
