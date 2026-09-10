"""Isolated V4.2 pipeline. V4.1 and earlier artifact trees are read-only. V6 is not retuned."""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Callable

from experiments.qa_chunk_selection.metrics import token_f1
from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, hn_quality_rows
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v4.heads import rank_by_scores
from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device
from experiments.qa_chunk_selection.v4.qa_shortlist import answer_one
from experiments.qa_chunk_selection.v4.shortlist import b0_rows_for_seed, index_b0_oof, load_b0_oof, top_k_ids
from experiments.qa_chunk_selection.v4_1.contract import PAIRWISE_TOLERANCE, V4_V6_WEIGHTS
from experiments.qa_chunk_selection.v4_1.ranking import shortlist_set_ok

from .attention import extract_compact_attention, logits_close, span_length_bias_probe, synthetic_compact
from .cache_util import ReadThroughCache
from .config import ExperimentConfigV42
from .contract import (
    HISTORICAL_B0,
    HISTORICAL_V6_K5,
    NONINFERIORITY_F1,
    PRIMARY_FINAL_SYSTEMS,
    experiment_contract_v4_2,
)
from .evaluate import chunk_prediction, metric_close, paired_delta, seed_design, summarize_rows, transitions
from .features import apply_shortlist_idf, shortlist_idf
from .heads import build_pools, choose_n, contrastive_r, stable_heads, union_topk
from .io_guard import assert_isolated_output, file_sha256, load_json, read_jsonl, write_json, write_jsonl
from .ranking import (
    att1_scores,
    att3_scores,
    bidirectional_shortlist,
    core_shortlist_scores,
    ranking_from_shortlist,
    relative_attention,
    v6_shortlist_scores,
)
from .report import write_report
from .select import choose_by_cal


class PipelineV42:
    def __init__(self, config: ExperimentConfigV42) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = Path(config.v3_artifact_dir)
        self.v4 = Path(config.v4_artifact_dir)
        self.v41 = Path(config.v4_1_artifact_dir)
        fallbacks = []
        if config.reuse_v4_1_qa_cache and not config.smoke:
            if (self.v41 / "cache").exists():
                fallbacks.append(FeatureCache(self.v41 / "cache"))
            if (self.v4 / "cache").exists():
                fallbacks.append(FeatureCache(self.v4 / "cache"))
        self.cache = ReadThroughCache(FeatureCache(self.output_dir / "cache"), fallbacks)
        self.attn_cache = FeatureCache(self.output_dir / "attention_cache")
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.flags = {}
        self.flags_rows: list[dict[str, Any]] = []
        self.b0_rows: list[dict[str, Any]] = []
        self.b0_index: dict[tuple[str, int], dict[str, Any]] = {}
        self.qa_store: dict[tuple[str, int], dict[str, Any]] = {}
        self.attn_store: dict[tuple[str, int], dict[str, Any]] = {}
        self.v6_weights: dict[int, dict[str, float]] = {}
        self.v6_rows: list[dict[str, Any]] = []
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.head_sets: dict[int, dict[str, Any]] = {}
        self.att1_gate = False
        self.att1_hn_worse = False
        self.bundle = None
        self.eager = False
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "r0": self.stage_r0,
            "cache": self.stage_cache,
            "core": self.stage_core,
            "att-reg": self.stage_att_reg,
            "att-extract": self.stage_att_extract,
            "heads": self.stage_heads,
            "att1": self.stage_att1,
            "att2": self.stage_att2,
            "att3": self.stage_att3,
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
        self.progress.log("E0/V2/V3/V3.1/D45/V4/V4.1 are read-only; V6 is not retuned")
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown V4.2 stage: {name}")
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

    def _tokenizer_hash(self) -> str:
        if self.config.smoke or self.config.skip_qa or self.bundle is None:
            return "lexical-smoke"
        answerer = getattr(self.bundle, "qa_answerer", None)
        tokenizer = getattr(answerer, "tokenizer", None)
        name = getattr(tokenizer, "name_or_path", None) or getattr(self.bundle, "qa_model_name", None) or "qa"
        return str(name)

    def _mkdir(self, *parts: str) -> Path:
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _b0_seed(self, seed: int) -> dict[str, dict[str, Any]]:
        return b0_rows_for_seed(self.b0_index, seed) if self.b0_index else {}

    def _b0_by_seeds(self):
        return {int(seed): self._b0_seed(int(seed)) for seed in self.config.model_seeds}

    def _by_qid(self):
        return {item.question_id: item for item in self.unlocked}

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

    def _predict(self, item, scores, qa_by_chunk, ranked, method: str) -> dict[str, Any]:
        top = int(ranked[0]) if ranked else int(item.chunks[0].chunk_id)
        qa = qa_by_chunk.get(top) or self.qa_store.get((item.question_id, top)) or {"qa_answer": "", "qa_chunk_id": top}
        return chunk_prediction(item, ranking_method=method, scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)

    def _gold_f1(self, item, cid: int) -> float:
        qa = self.qa_store.get((item.question_id, int(cid))) or {}
        return float(token_f1(str(qa.get("qa_answer") or ""), list(item.gold_answers)))

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v4_2())
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
        identity = {}
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
            raise FileNotFoundError(f"V4.1 V6-K5 OOF missing under {self.v41}; V4.2 cannot retune V6")
        write_json(v6_dest / "weights_by_run.json", {str(k): v for k, v in self.v6_weights.items()})
        clone = paired_delta(self.b0_rows, self.b0_rows, n_boot=min(self.config.n_bootstrap, 200 if self.config.smoke else self.config.n_bootstrap), seed=self.config.seed)
        write_json(self.output_dir / "regression" / "r0_summary.json", {"b0_identity": identity, "ok": ok or self.config.smoke, "paired_identity": clone, "v6_weights_runs": sorted(self.v6_weights)})
        if not (ok or self.config.smoke):
            self.exit_code = 2
            self.progress.warn("R0 identity failed")
        self.progress.log("R0 identity", ok=ok or self.config.smoke, b0=b0_summary.get("token_f1"))

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
            self.v6_rows = self._score_system("V6-K5", lambda item, row, seed, run: self._v6_item(item, row, run=run), eval_only=True)
            self._record("V6-K5", self.v6_rows, self._mkdir("reference", "v6_k5_v4_1"))
        self.progress.log("cache ready", n_qa=len(self.qa_store), fallback_hits=self.cache.stats().get("fallback_hits"))

    def _v6_item(self, item, b0_row, *, run: int):
        weights = self.v6_weights.get(int(run), dict(V4_V6_WEIGHTS))
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        ranked, scores, _feats = v6_shortlist_scores(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=self.config.primary_k, min_overlap=self.config.source_overlap
        )
        pred = self._predict(item, scores, qa_by_chunk, ranked, "V6-K5")
        pred["shortlist_ok"] = shortlist_set_ok(b0_row, ranked, k=self.config.primary_k)
        pred["v6_shortlist_scores"] = {str(cid): float(scores[cid]) for cid in top_k_ids(b0_row, self.config.primary_k) if cid in scores}
        return pred, ranked, scores

    def _score_system(self, method: str, fn, *, eval_only: bool = True) -> list[dict[str, Any]]:
        by_qid = self._by_qid()
        rows = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval" if eval_only else "train"))
            if not eval_only:
                items = list(self.unlocked)
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred, _ranked, _scores = fn(item, row, seed, run)
                    pred["seed"] = int(seed)
                    pred["cv_run"] = int(run)
                    pred["document_id"] = item.document_id
                    rows.append(pred)
        del by_qid
        return rows

    def stage_core(self) -> None:
        oof1, oof2, oof3 = [], [], []
        n_runs = len(self.splits.get("runs") or [])
        chosen = {}
        for run in range(n_runs):
            self.progress.log(f"CORE outer run {run}/{n_runs - 1}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            b0_cal = [row for row in self.b0_rows if any(row["question_id"] == item.question_id for item in cal_items)]
            b0_hn = float(summarize_rows(b0_cal).get("hn@1") or 0.0)

            def predict_core1(params, items=cal_items):
                return self._core_rows(items, run, "CORE1", float(params["lambda_m"]), 0.0)

            choice1 = choose_by_cal(
                cal_items,
                grid=[{"lambda_m": lam} for lam in self.config.core_majority_grid],
                predict_rows=predict_core1,
                b0_hn=b0_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("lambda_m",),
                progress=self.progress,
                item_name=f"core1_run{run}",
            )
            grid2 = [{"lambda_m": a, "lambda_t": b} for a, b in product(self.config.core_majority_grid, self.config.core_token_grid)]

            def predict_core2(params, items=cal_items):
                return self._core_rows(items, run, "CORE2", float(params["lambda_m"]), float(params["lambda_t"]))

            choice2 = choose_by_cal(
                cal_items,
                grid=grid2,
                predict_rows=predict_core2,
                b0_hn=b0_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("lambda_m", "lambda_t"),
                progress=self.progress,
                item_name=f"core2_run{run}",
            )
            run_dir = self._mkdir("core", f"run{run}")
            write_json(run_dir / "selected_core1.json", choice1)
            write_json(run_dir / "selected_core2.json", choice2)
            chosen[run] = {"core1": choice1["selected"], "core2": choice2["selected"]}
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in eval_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    p1 = self._core_one(item, row, run, "CORE1", float(choice1["selected"]["lambda_m"]), 0.0)
                    p1["seed"] = int(seed)
                    p1["cv_run"] = int(run)
                    oof1.append(p1)
                    p2 = self._core_one(item, row, run, "CORE2", float(choice2["selected"]["lambda_m"]), float(choice2["selected"]["lambda_t"]))
                    p2["seed"] = int(seed)
                    p2["cv_run"] = int(run)
                    oof2.append(p2)
                    p3 = self._core_one(item, row, run, "CORE3", float(choice2["selected"]["lambda_m"]), float(choice2["selected"]["lambda_t"]))
                    p3["seed"] = int(seed)
                    p3["cv_run"] = int(run)
                    oof3.append(p3)
        self._record("CORE1", oof1, self._mkdir("core", "core1"))
        self._record("CORE2", oof2, self._mkdir("core", "core2"))
        self._record("CORE3", oof3, self._mkdir("core", "core3"))
        write_jsonl(self.output_dir / "core" / "oof_core1.jsonl", oof1)
        write_jsonl(self.output_dir / "core" / "oof_core2.jsonl", oof2)
        write_json(self.output_dir / "core" / "weights_by_run.json", {str(k): v for k, v in chosen.items()})
        v6 = self.predictions.get("V6-K5") or []
        if v6:
            write_json(self.output_dir / "core" / "bootstrap_core1_vs_v6.json", paired_delta(v6, oof1, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed))
            write_json(self.output_dir / "core" / "bootstrap_core2_vs_v6.json", paired_delta(v6, oof2, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed))

    def _core_rows(self, items, run: int, variant: str, lambda_m: float, lambda_t: float) -> list[dict[str, Any]]:
        rows = []
        for seed in self.config.model_seeds:
            b0_map = self._b0_seed(int(seed))
            for item in items:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                pred = self._core_one(item, row, run, variant, lambda_m, lambda_t)
                pred["seed"] = int(seed)
                rows.append(pred)
        return rows

    def _core_one(self, item, b0_row, run: int, variant: str, lambda_m: float, lambda_t: float) -> dict[str, Any]:
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        scores = core_shortlist_scores(
            item, b0_row, qa_by_chunk, k=self.config.primary_k, min_overlap=self.config.source_overlap, variant=variant, lambda_m=lambda_m, lambda_t=lambda_t
        )
        ranked, fused = ranking_from_shortlist(item, b0_row, scores, k=self.config.primary_k)
        pred = self._predict(item, fused, qa_by_chunk, ranked, variant)
        pred["shortlist_ok"] = shortlist_set_ok(b0_row, ranked, k=self.config.primary_k)
        return pred

    def _selection_identity(self, left, right, extra=None) -> dict[str, Any]:
        index_left = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in left}
        index_right = {(str(row["question_id"]), int(row.get("seed") or 0)): row for row in right}
        keys = sorted(set(index_left) & set(index_right))
        n_chunk = sum(str(index_left[key].get("selected_chunk_id")) == str(index_right[key].get("selected_chunk_id")) for key in keys)
        n_ans = sum(str(index_left[key].get("qa_answer") or "") == str(index_right[key].get("qa_answer") or "") for key in keys)
        payload = {
            "ok": bool(keys) and n_chunk == len(keys) and n_ans == len(keys),
            "n": len(keys),
            "same_chunk": n_chunk,
            "same_answer": n_ans,
        }
        if extra:
            payload.update(extra)
        return payload

    def stage_att_reg(self) -> None:
        probe = span_length_bias_probe(seed=self.config.seed)
        write_json(self.output_dir / "regression" / "span_length_bias_test.json", probe)
        items = list(self.unlocked)[: min(4, len(self.unlocked))]
        logit_rows = []
        if self.config.smoke or self.config.skip_qa:
            for item in items:
                chunk = item.chunks[0]
                payload = synthetic_compact(item.question, chunk.text, predicted_start=0, predicted_end=min(8, len(chunk.text)), seed=self.config.seed)
                logit_rows.append(payload["logit_regression"])
            write_json(self.output_dir / "regression" / "logits_identity.json", {"ok": True, "smoke": True, "rows": logit_rows})
            write_json(self.output_dir / "regression" / "span_identity.json", {"ok": True, "smoke": True})
            write_json(self.output_dir / "regression" / "top5_identity.json", {"ok": True, "note": "B0 top-5 reused, never rewritten"})
            self.progress.log("attention regression smoke/skip-qa ok")
            return
        self._answer_fn()
        answerer = self.bundle.qa_answerer if self.bundle is not None else None
        if answerer is None:
            raise RuntimeError("frozen QA answerer missing for attention regression")
        n_ok = 0
        for item in items:
            cid = top_k_ids(self._b0_seed(self.config.seed).get(item.question_id) or {"ranked_ids": [item.chunks[0].chunk_id]}, self.config.primary_k)
            if not cid:
                continue
            qa = self.qa_store.get((item.question_id, int(cid[0]))) or {}
            chunk = item.chunk_by_id(int(cid[0]))
            payload = extract_compact_attention(
                answerer,
                item.question,
                chunk.text,
                predicted_start=int(qa.get("qa_chunk_start_char") or 0),
                predicted_end=int(qa.get("qa_chunk_end_char") or 0),
                predicted_answer=str(qa.get("qa_answer") or ""),
                eager=self.eager,
            )
            ok = logits_close(payload)
            if not ok and not self.eager:
                self.eager = True
                payload = extract_compact_attention(
                    answerer,
                    item.question,
                    chunk.text,
                    predicted_start=int(qa.get("qa_chunk_start_char") or 0),
                    predicted_end=int(qa.get("qa_chunk_end_char") or 0),
                    predicted_answer=str(qa.get("qa_answer") or ""),
                    eager=True,
                )
                ok = logits_close(payload)
            logit_rows.append({**payload["logit_regression"], "ok": ok, "question_id": item.question_id})
            n_ok += int(ok)
        write_json(self.output_dir / "regression" / "logits_identity.json", {"ok": n_ok == len(logit_rows) and logit_rows, "eager": self.eager, "rows": logit_rows})
        write_json(self.output_dir / "regression" / "span_identity.json", {"ok": all(row.get("identical_argmax_start") and row.get("identical_argmax_end") and row.get("identical_answer_text", True) for row in logit_rows), "n": len(logit_rows)})
        write_json(self.output_dir / "regression" / "top5_identity.json", {"ok": True, "note": "B0 ranked_ids are never rewritten"})
        if logit_rows and n_ok != len(logit_rows):
            self.progress.warn("attention logits identity not exact; continuing with recorded backend")
        self.progress.log("attention regression done", eager=self.eager, n_ok=n_ok)

    def _attn_key(self, qid: str, cid: int) -> str:
        from experiments.qa_chunk_selection.v2.cache import cache_key

        return cache_key(
            question_id=qid,
            chunk_id=int(cid),
            qa_model_hash=self._model_id(),
            tokenizer_hash=self._tokenizer_hash(),
            attention_backend="eager" if self.eager else "default",
            attention_feature_version=self.config.attention_feature_version,
        )

    def stage_att_extract(self) -> None:
        needed: set[tuple[str, int]] = set()
        by_qid = self._by_qid()
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                if qid in by_qid:
                    for cid in top_k_ids(row, self.config.primary_k):
                        needed.add((qid, int(cid)))
        started = time.time()
        answerer = None if (self.config.smoke or self.config.skip_qa) else (self.bundle.qa_answerer if self.bundle is not None else None)
        if not self.config.smoke and not self.config.skip_qa and answerer is None:
            self._answer_fn()
            answerer = self.bundle.qa_answerer
        for index, (qid, cid) in enumerate(sorted(needed), start=1):
            key = self._attn_key(qid, cid)
            hit = self.attn_cache.get("attn", key)
            if hit is not None:
                self.attn_store[(qid, cid)] = hit
                self.progress.count(index, max(len(needed), 1), item="attn_compact")
                continue
            item = by_qid[qid]
            chunk = item.chunk_by_id(int(cid))
            qa = self.qa_store.get((qid, cid)) or {}
            if self.config.smoke or self.config.skip_qa:
                payload = synthetic_compact(
                    item.question,
                    chunk.text,
                    predicted_start=int(qa.get("qa_chunk_start_char") or 0),
                    predicted_end=int(qa.get("qa_chunk_end_char") or min(8, len(chunk.text))),
                    seed=hash((qid, cid)) % 10**6,
                )
            else:
                payload = extract_compact_attention(
                    answerer,
                    item.question,
                    chunk.text,
                    predicted_start=int(qa.get("qa_chunk_start_char") or 0),
                    predicted_end=int(qa.get("qa_chunk_end_char") or 0),
                    predicted_answer=str(qa.get("qa_answer") or ""),
                    eager=self.eager,
                )
            payload["question_id"] = qid
            payload["chunk_id"] = int(cid)
            self.attn_cache.set("attn", key, payload)
            self.attn_store[(qid, cid)] = payload
            self.progress.count(index, max(len(needed), 1), item="attn_compact")
        elapsed = time.time() - started
        write_json(
            self.output_dir / "attention_cache" / "feature_contract.json",
            {
                "version": self.config.attention_feature_version,
                "primary": "Q->span enrichment + span->Q density, z-scored within shortlist, then mean selected heads",
                "raw_tensors_saved": False,
                "n": len(self.attn_store),
                "seconds": elapsed,
                "per_candidate_s": elapsed / max(len(needed), 1),
                "eager": self.eager,
                "att2": "IDF-weighted Q->span token mass, then |S|/|C| enrichment; span->Q unweighted",
            },
        )
        write_jsonl(
            self.output_dir / "attention_cache" / "compact_features.jsonl",
            [
                {
                    "question_id": qid,
                    "chunk_id": cid,
                    "valid_span": row.get("valid_span"),
                    "span_length_tokens": row.get("span_length_tokens"),
                    "context_length_tokens": row.get("context_length_tokens"),
                    "n_layers": row.get("n_layers"),
                    "n_heads": row.get("n_heads"),
                }
                for (qid, cid), row in sorted(self.attn_store.items())
            ],
        )
        self.progress.log("compact attention ready", n=len(self.attn_store), seconds=round(elapsed, 1))

    def _train_fold_items(self, run: int) -> dict[int, list]:
        spec = self.splits["runs"][int(run)]
        assignment = self.splits.get("assignment") or {}
        grouped: dict[int, list] = defaultdict(list)
        train_folds = {int(item) for item in spec["train"]}
        for item in self.unlocked:
            fold = int(assignment.get(item.document_id, -1))
            if fold in train_folds:
                grouped[fold].append(item)
        return dict(grouped)

    def stage_heads(self) -> None:
        n_runs = len(self.splits.get("runs") or [])
        by_seeds = self._b0_by_seeds()
        leakage = []
        for run in range(n_runs):
            self.progress.log(f"head selection outer run {run}")
            train_items = list(records_for_role(self.unlocked, self.splits, run=run, role="train"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            train_q = {item.question_id for item in train_items}
            eval_q = {item.question_id for item in eval_items}
            leak = sorted(train_q & eval_q)
            if leak:
                leakage.append({"run": run, "n": len(leak)})
            union = union_topk(train_items, by_seeds, self.config.model_seeds, k=self.config.primary_k)
            compact = {(qid, cid): self.attn_store[(qid, cid)] for qid, cids in union.items() for cid in cids if (qid, cid) in self.attn_store}
            pools, audit = build_pools(
                train_items,
                union,
                compact,
                self.flags,
                reader_easy_threshold=self.config.reader_easy_threshold,
                gold_f1=self._gold_f1,
                fallback=False,
                b0_by_seed=by_seeds,
                seeds=self.config.model_seeds,
                k=self.config.primary_k,
            )
            used_fallback = False
            if int(audit["n_questions_with_valid_pos_and_HN"]) < int(self.config.min_paired_questions):
                pools, audit = build_pools(
                    train_items,
                    union,
                    compact,
                    self.flags,
                    reader_easy_threshold=self.config.reader_easy_threshold,
                    gold_f1=self._gold_f1,
                    fallback=True,
                    b0_by_seed=by_seeds,
                    seeds=self.config.model_seeds,
                    k=self.config.primary_k,
                    )
                used_fallback = True
            unreliable = int(audit["n_pooled_questions"]) < int(self.config.min_paired_questions)
            run_dir = self._mkdir("head_selection", f"run{run}")
            write_json(run_dir / "feasibility.json", {**audit, "negative_fallback": used_fallback, "unreliable": unreliable})
            write_json(run_dir / "seed_union_dedup.json", {"n_questions": len(union), "n_pairs": sum(len(v) for v in union.values()), "unique_key": "question_id,chunk_id"})
            if unreliable:
                self.head_sets[run] = {"heads": [], "n": 0, "delta_forced_0": True, "unreliable": True}
                write_json(run_dir / "selected_heads.json", self.head_sets[run])
                self.progress.warn(f"run {run} head discovery unreliable; delta forced to 0")
                continue
            fold_items = self._train_fold_items(run)
            fold_pools = []
            fold_r = []
            for fold, items in sorted(fold_items.items()):
                fp, _a = build_pools(
                    items,
                    union_topk(items, by_seeds, self.config.model_seeds, k=self.config.primary_k),
                    compact,
                    self.flags,
                    reader_easy_threshold=self.config.reader_easy_threshold,
                    gold_f1=self._gold_f1,
                    fallback=used_fallback,
                    b0_by_seed=by_seeds,
                    seeds=self.config.model_seeds,
                    k=self.config.primary_k,
                )
                fold_pools.append(fp)
                fold_r.append(contrastive_r(fp, compact))
            n_choice = choose_n(fold_pools, compact, n_grid=self.config.n_head_grid)
            overall = contrastive_r(pools, compact)
            ranked = stable_heads(fold_r, overall) or sorted(overall, key=lambda key: (-float(overall[key]), key[0], key[1]))
            n = int((n_choice.get("selected") or {}).get("n") or 1)
            chosen = ranked[:n]
            coverage = {
                f"{a}-{b}": int(sum(1 for fold in fold_r if float(fold.get((a, b), 0.5)) > 0.5))
                for a, b in overall
            }
            payload = {
                "heads": [{"layer": int(a), "head": int(b)} for a, b in chosen],
                "n": n,
                "delta_forced_0": False,
                "unreliable": False,
                "negative_fallback": used_fallback,
                "overall_top": [{"layer": int(a), "head": int(b), "R": overall[(a, b)]} for a, b in ranked[:16]],
                "coverage_folds_gt_0.5": {f"{a}-{b}": coverage.get(f"{a}-{b}", 0) for a, b in chosen},
            }
            self.head_sets[run] = payload
            write_json(run_dir / "inner_n_selection.json", n_choice)
            write_json(
                run_dir / "head_scores.json",
                {
                    "overall": {f"{a}-{b}": v for (a, b), v in overall.items()},
                    "per_fold": [{f"{a}-{b}": v for (a, b), v in fold.items()} for fold in fold_r],
                    "coverage_folds_gt_0.5": coverage,
                },
            )
            write_json(run_dir / "selected_heads.json", payload)
        write_json(self.output_dir / "regression" / "seed_union_dedup.json", {"ok": True, "eval_leakage_runs": leakage})
        if leakage:
            self.progress.warn("EVAL ids overlapped TRAIN head discovery", n=len(leakage))
        self.progress.log("head selection done", n_runs=len(self.head_sets))

    def _compact_shortlist(self, item, b0_row) -> dict[int, dict[str, Any]]:
        out = {}
        for cid in top_k_ids(b0_row, self.config.primary_k):
            feat = self.attn_store.get((item.question_id, int(cid)))
            if feat is not None:
                out[int(cid)] = feat
        return out

    def _v6_and_rel(self, item, b0_row, run: int, *, idf: bool = False):
        weights = self.v6_weights.get(int(run), dict(V4_V6_WEIGHTS))
        qa_by_chunk = self._qa_map(item, b0_row, self.config.primary_k)
        ranked_v6, v6_scores, _feats = v6_shortlist_scores(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=self.config.primary_k, min_overlap=self.config.source_overlap
        )
        top = top_k_ids(b0_row, self.config.primary_k)
        v6_short = {int(cid): float(v6_scores.get(int(cid), 0.0)) for cid in top}
        compact = self._compact_shortlist(item, b0_row)
        spec = self.head_sets.get(int(run)) or {}
        selected = [(int(row["layer"]), int(row["head"])) for row in spec.get("heads") or []]
        if idf:
            texts = {int(cid): item.chunk_by_id(int(cid)).text for cid in top}
            idf_map = shortlist_idf(texts, k=self.config.primary_k)
            compact = {int(cid): apply_shortlist_idf(feat, idf_map) for cid, feat in compact.items()}
        raw = bidirectional_shortlist(compact, selected) if selected else {}
        for cid in top:
            raw.setdefault(int(cid), 0.0)
        a_rel = relative_attention(raw) if selected else {int(cid): 0.0 for cid in top}
        return qa_by_chunk, ranked_v6, v6_short, a_rel

    def _att_rows(self, items, run: int, *, delta: float, kind: str, tau: float = 0.0, lambda_v: float = 0.0) -> list[dict[str, Any]]:
        rows = []
        spec = self.head_sets.get(int(run)) or {}
        forced = bool(spec.get("delta_forced_0"))
        use_delta = 0.0 if forced else float(delta)
        for seed in self.config.model_seeds:
            b0_map = self._b0_seed(int(seed))
            for item in items:
                row = b0_map.get(item.question_id)
                if row is None:
                    continue
                qa_by_chunk, _rv, v6_short, a_rel = self._v6_and_rel(item, row, run, idf=(kind == "ATT2"))
                if kind == "ATT3":
                    scores = att3_scores(v6_short, a_rel, tau=float(tau), lambda_v=float(lambda_v))
                else:
                    scores = att1_scores(v6_short, a_rel, delta=use_delta)
                ranked, fused = ranking_from_shortlist(item, row, scores, k=self.config.primary_k)
                pred = self._predict(item, fused, qa_by_chunk, ranked, kind)
                pred["seed"] = int(seed)
                pred["shortlist_ok"] = shortlist_set_ok(row, ranked, k=self.config.primary_k)
                pred["delta"] = use_delta
                rows.append(pred)
        return rows

    def stage_att1(self) -> None:
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        chosen = {}
        v6_cal_hn_ref = []
        for run in range(n_runs):
            self.progress.log(f"ATT1 outer run {run}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_cal = self._att_rows(cal_items, run, delta=0.0, kind="ATT1")
            v6_hn = float(summarize_rows(v6_cal).get("hn@1") or 0.0)
            v6_cal_hn_ref.append(v6_hn)
            spec = self.head_sets.get(run) or {}
            if spec.get("delta_forced_0"):
                choice = {"selected": {"delta": 0.0, "cal_token_f1": mean_token_safe(v6_cal), "forced": True}, "tried": []}
            else:
                choice = choose_by_cal(
                    cal_items,
                    grid=[{"delta": d} for d in self.config.delta_grid],
                    predict_rows=lambda params, r=run: self._att_rows(cal_items, r, delta=float(params["delta"]), kind="ATT1"),
                    b0_hn=v6_hn,
                    epsilon_hn=self.config.epsilon_hn,
                    prefer_smaller=("delta",),
                    progress=self.progress,
                    item_name=f"att1_delta_run{run}",
                )
            run_dir = self._mkdir("att1", f"run{run}")
            write_json(run_dir / "selected_delta.json", choice)
            chosen[run] = choice["selected"]
            delta = float(choice["selected"].get("delta") or 0.0)
            zero_rows = self._att_rows(eval_items, run, delta=0.0, kind="ATT1")
            att_rows = self._att_rows(eval_items, run, delta=delta, kind="ATT1")
            write_json(run_dir / "delta0_identity.json", self._selection_identity(zero_rows, self.predictions.get("V6-K5") or [], extra={"selected_delta": delta}))
            if abs(delta) <= 1e-12:
                write_json(run_dir / "delta0_vs_selected.json", self._selection_identity(zero_rows, att_rows))
            for pred in att_rows:
                pred["cv_run"] = int(run)
                oof.append(pred)
        summary = self._record("ATT1", oof, self._mkdir("att1"))
        write_jsonl(self.output_dir / "att1" / "oof_predictions.jsonl", oof)
        v6 = self.predictions.get("V6-K5") or []
        delta = paired_delta(v6, oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if v6 else {}
        write_json(self.output_dir / "att1" / "bootstrap_vs_v6.json", delta)
        write_json(self.output_dir / "att1" / "transitions.json", transitions(v6, oof) if v6 else {})
        hn_v6 = float((self.summaries.get("V6-K5") or {}).get("hn@1") or 0.0)
        hn_att = float(summary.get("hn@1") or 0.0)
        self.att1_hn_worse = hn_att > hn_v6 + self.config.epsilon_hn
        low = delta.get("ci95_low")
        self.att1_gate = bool(v6) and float(delta.get("mean_delta") or 0.0) > 0 and low is not None and float(low) > 0 and not self.att1_hn_worse
        write_json(
            self.output_dir / "att1" / "gate.json",
            {"att1_gate": self.att1_gate, "hn_worse": self.att1_hn_worse, "delta": delta, "selected_delta": {str(k): v for k, v in chosen.items()}},
        )
        self.progress.log("ATT1 OOF", token_f1=summary.get("token_f1"), gate=self.att1_gate, dF1=delta.get("mean_delta"))

    def stage_att2(self) -> None:
        if not self.att1_gate and not self.config.force_att2 and not self.config.smoke:
            self.progress.log("ATT2 skipped: ATT1 did not pass primary gate (use --force-att2 for exploratory)")
            write_json(self.output_dir / "att2_idf" / "status.json", {"skipped": True, "reason": "ATT1 gate failed"})
            return
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_cal = self._att_rows(cal_items, run, delta=0.0, kind="ATT2")
            v6_hn = float(summarize_rows(v6_cal).get("hn@1") or 0.0)
            choice = choose_by_cal(
                cal_items,
                grid=[{"delta": d} for d in self.config.delta_grid],
                predict_rows=lambda params, r=run: self._att_rows(cal_items, r, delta=float(params["delta"]), kind="ATT2"),
                b0_hn=v6_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("delta",),
                progress=self.progress,
                item_name=f"att2_delta_run{run}",
            )
            write_json(self._mkdir("att2_idf", f"run{run}") / "selected_delta.json", choice)
            for pred in self._att_rows(eval_items, run, delta=float(choice["selected"]["delta"]), kind="ATT2"):
                pred["cv_run"] = int(run)
                oof.append(pred)
        summary = self._record("ATT2", oof, self._mkdir("att2_idf"))
        v6 = self.predictions.get("V6-K5") or []
        write_json(self.output_dir / "att2_idf" / "bootstrap_vs_v6.json", paired_delta(v6, oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if v6 else {})
        self.progress.log("ATT2 OOF", token_f1=summary.get("token_f1"), confirmatory=self.att1_gate)

    def stage_att3(self) -> None:
        if not self.att1_hn_worse and not self.config.force_att3 and not self.config.smoke:
            self.progress.log("ATT3 skipped: ATT1 HN did not exceed V6+eps")
            write_json(self.output_dir / "att3_veto" / "status.json", {"skipped": True, "reason": "no HN problem"})
            return
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        grid = [{"tau": t, "lambda_v": l} for t, l in product(self.config.att3_tau_grid, self.config.att3_lambda_grid)]
        for run in range(n_runs):
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            v6_cal = self._att_rows(cal_items, run, delta=0.0, kind="ATT1")
            v6_hn = float(summarize_rows(v6_cal).get("hn@1") or 0.0)
            choice = choose_by_cal(
                cal_items,
                grid=grid,
                predict_rows=lambda params, r=run: self._att_rows(cal_items, r, delta=0.0, kind="ATT3", tau=float(params["tau"]), lambda_v=float(params["lambda_v"])),
                b0_hn=v6_hn,
                epsilon_hn=self.config.epsilon_hn,
                prefer_smaller=("lambda_v", "tau"),
                progress=self.progress,
                item_name=f"att3_run{run}",
            )
            write_json(self._mkdir("att3_veto", f"run{run}") / "selected.json", choice)
            sel = choice["selected"]
            for pred in self._att_rows(eval_items, run, delta=0.0, kind="ATT3", tau=float(sel["tau"]), lambda_v=float(sel["lambda_v"])):
                pred["cv_run"] = int(run)
                oof.append(pred)
        self._record("ATT3", oof, self._mkdir("att3_veto"))

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
            delta = paired_delta(b0, rows, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed)
            deltas[name] = delta
            trans[name] = transitions(b0, rows)
            hn_ok = float(summary.get("hn@1") or 0.0) <= b0_hn + self.config.epsilon_hn
            auto = name in allowed and bool(design.get("three_seed")) and hn_ok
            if name == "ATT1" and not self.att1_gate:
                auto = False
            if name in {"ATT2", "ATT3", "CORE3"}:
                auto = False
            if name.startswith("CORE"):
                vs_v6 = paired_delta(v6, rows, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if v6 else {}
                deltas[f"{name}_vs_v6"] = vs_v6
                low_v6 = vs_v6.get("ci95_low")
                row["noninferior_vs_v6"] = low_v6 is not None and float(low_v6) > NONINFERIORITY_F1
                if name != "CORE3" and not row["noninferior_vs_v6"]:
                    auto = False
            row["admissible"] = auto
            row["hn_safeguard"] = hn_ok
            if auto:
                admissible.append(name)
            table.append(row)
        selected = "V6-K5" if "V6-K5" in self.predictions else "B0-live"
        best = float((self.summaries.get(selected) or {}).get("token_f1") or 0.0)
        for name in admissible:
            f1 = float((self.summaries.get(name) or {}).get("token_f1") or 0.0)
            dlt = deltas.get(name) or {}
            low = dlt.get("ci95_low")
            ci_ok = True
            if name in {"ATT1", "CORE1", "CORE2", "V6-K5"} and name != "V6-K5" and low is not None and float(low) <= 0:
                ci_ok = False
            if name == "ATT1" and not self.att1_gate:
                continue
            if f1 > best and ci_ok:
                best = f1
                selected = name
        att_vs_v6 = paired_delta(v6, self.predictions.get("ATT1") or [], n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if v6 and "ATT1" in self.predictions else {}
        payload = {
            "table": table,
            "deltas": deltas,
            "transitions": trans,
            "selected": selected,
            "admissible": admissible,
            "b0_token_f1": (self.summaries.get("B0-live") or {}).get("token_f1"),
            "v6_token_f1": (self.summaries.get("V6-K5") or {}).get("token_f1"),
            "core1_token_f1": (self.summaries.get("CORE1") or {}).get("token_f1"),
            "core2_token_f1": (self.summaries.get("CORE2") or {}).get("token_f1"),
            "att1_token_f1": (self.summaries.get("ATT1") or {}).get("token_f1"),
            "att1_vs_v6": att_vs_v6,
            "att1_gate": self.att1_gate,
            "internal_test_locked": True,
            "systems": list(self.predictions),
        }
        write_json(self.output_dir / "analysis" / "compare.json", payload)
        write_json(self.output_dir / "analysis" / "selection.json", {"selected": selected, "admissible": admissible, "att1_gate": self.att1_gate, "IN_SAMPLE_CONTAMINATED": False})
        write_json(self.output_dir / "analysis" / "transitions.json", trans)
        self._compare_payload = payload
        self._write_head_stability()
        self._write_aspect()
        self.progress.log("compare done", selected=selected, att1_gate=self.att1_gate)

    def _write_head_stability(self) -> None:
        sets = []
        for run, spec in sorted(self.head_sets.items()):
            heads = {(int(row["layer"]), int(row["head"])) for row in spec.get("heads") or []}
            sets.append(heads)
        jaccard = []
        for i, left in enumerate(sets):
            for j, right in enumerate(sets):
                if j <= i:
                    continue
                union = left | right
                jaccard.append(len(left & right) / max(len(union), 1))
        counts: dict[str, int] = defaultdict(int)
        for heads in sets:
            for layer, head in heads:
                counts[f"{layer}-{head}"] += 1
        write_json(
            self.output_dir / "analysis" / "head_stability.json",
            {"mean_jaccard": float(sum(jaccard) / max(len(jaccard), 1)) if jaccard else None, "recurrence": dict(counts), "n_runs": len(sets)},
        )

    def _write_aspect(self) -> None:
        payload = {}
        for name, rows in self.predictions.items():
            by = defaultdict(list)
            for row in rows:
                by[str(row.get("aspect") or "none")].append(row)
            payload[name] = {asp: summarize_rows(group) for asp, group in sorted(by.items())}
        write_json(self.output_dir / "analysis" / "aspect_metrics.json", payload)

    def stage_freeze(self) -> None:
        payload = getattr(self, "_compare_payload", None) or {}
        selected = payload.get("selected") or "V6-K5"
        write_json(
            self.output_dir / "analysis" / "efficiency.json",
            {
                "qa_cache": self.cache.stats(),
                "attn_cache": self.attn_cache.stats(),
                "n_attn": len(self.attn_store),
                "note": "live attention cost is att-extract wall time; V4.1 QA may be cached",
                "eager": self.eager,
            },
        )
        self._mkdir("final")
        write_json(self.output_dir / "final" / "model_selection.json", {"selected": selected, "att1_gate": self.att1_gate, "admissible": payload.get("admissible")})
        write_json(
            self.output_dir / "final" / "freeze_manifest.json",
            {
                "final_architecture": selected,
                "v6_retuned": False,
                "v6_weights": "V4.1 per-run",
                "primary_k": self.config.primary_k,
                "internal_test": "locked",
                "att1_gate": self.att1_gate,
                "b0_final": "NOT trained",
                "u80": "not in V4.2",
                "k10_attention": "not in V4.2",
            },
        )
        self.progress.log("freeze written; INTERNAL_TEST still locked", selected=selected)

    def stage_report(self) -> None:
        payload = getattr(self, "_compare_payload", None)
        if payload is None:
            path = self.output_dir / "analysis" / "compare.json"
            payload = load_json(path) if path.exists() else {"internal_test_locked": True}
        write_report(self.output_dir, payload)
        self.progress.log("report written", path=str(self.output_dir / "reports" / "results_v4_2.md"))


def mean_token_safe(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(summarize_rows(rows).get("token_f1") or 0.0)
