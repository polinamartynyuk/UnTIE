"""Isolated V4.1 pipeline. V4 and earlier artifact trees are read-only."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.split import build_cv_splits, records_for_role
from experiments.qa_chunk_selection.v3.train import score_fn_for
from experiments.qa_chunk_selection.v4.answers import normalize_predicted_answer
from experiments.qa_chunk_selection.v4.clusters import document_source_clusters, overlapping_ids, shortlist_source_clusters
from experiments.qa_chunk_selection.v4.heads import rank_by_scores, softmax
from experiments.qa_chunk_selection.v4.qa_shortlist import answer_one
from experiments.qa_chunk_selection.v4.shortlist import b0_rows_for_seed, index_b0_oof, load_b0_oof, scores_from_row, top_k_ids
from experiments.qa_chunk_selection.v4.v2_aggregate import aggregate
from experiments.qa_chunk_selection.v4.v3_utility import mine_utility_positives, train_utility_listwise, utility_index
from experiments.qa_chunk_selection.v4.v6_reader import apply_v6
from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device

from .cache_util import ReadThroughCache
from .config import ExperimentConfigV41
from .contract import (
    B0_IDENTITY_TOLERANCE,
    HISTORICAL_B0,
    HISTORICAL_V4_V2A4_TOKEN_F1,
    HISTORICAL_V4_V6_TOKEN_F1,
    PAIRWISE_TOLERANCE,
    PRIMARY_FINAL_SYSTEMS,
    REPRO_F1_TOLERANCE,
    SOURCE_OVERLAP,
    UTILITY_F1_PCT,
    V4_V6_WEIGHTS,
    experiment_contract_v4_1,
)
from .evaluate import (
    aggregation_prediction,
    chunk_prediction,
    metric_close,
    paired_delta,
    seed_design,
    summarize_rows,
    transitions,
)
from .features import collect_v6_feature_rows, feature_contract_payload
from .io_guard import assert_isolated_output, file_sha256, load_json, read_jsonl, write_json, write_jsonl
from .ranking import apply_v6_isolated, shortlist_set_ok
from .report import write_report
from .select import choose_lambda_agg, choose_v6_weights


class PipelineV41:
    def __init__(self, config: ExperimentConfigV41) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = Path(config.v3_artifact_dir)
        self.v4 = Path(config.v4_artifact_dir)
        primary = FeatureCache(self.output_dir / "cache")
        fallback = None
        if config.reuse_v4_cache and (self.v4 / "cache").exists() and not config.smoke:
            fallback = FeatureCache(self.v4 / "cache")
        self.cache = ReadThroughCache(primary, fallback)
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.flags = {}
        self.flags_rows: list[dict[str, Any]] = []
        self.b0_rows: list[dict[str, Any]] = []
        self.b0_index: dict[tuple[str, int], dict[str, Any]] = {}
        self.qa_store: dict[tuple[str, int], dict[str, Any]] = {}
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.r1_weights: dict[int, dict[str, float]] = {}
        self.r3_lambdas: dict[int, float] = {}
        self.r0_ok = True
        self.repro_ok = True
        self.bundle = None
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "r0": self.stage_r0,
            "cache": self.stage_cache,
            "r0-v6-repro": self.stage_r0_v6_repro,
            "r1": self.stage_r1,
            "r2": self.stage_r2,
            "r3": self.stage_r3,
            "r4": self.stage_r4,
            "r5": self.stage_r5,
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
        self.progress.log("E0/V2/V3/V3.1/D45/V4 are read-only")
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown V4.1 stage: {name}")
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
            self.progress.log("loading frozen extractive QA only (no generative teacher)")
            self.bundle = load_frozen_qa(device=self.config.device, smoke=False)
        return self.bundle.answer

    def _model_id(self) -> str:
        if self.config.smoke or self.bundle is None:
            return "lexical-smoke" if self.config.smoke else "qa-pending"
        return str(self.bundle.qa_model_name)

    def _by_qid(self):
        return {item.question_id: item for item in self.unlocked}

    def _b0_seed(self, seed: int) -> dict[str, dict[str, Any]]:
        return b0_rows_for_seed(self.b0_index, seed) if self.b0_index else {}

    def _b0_by_seeds(self) -> dict[int, dict[str, dict[str, Any]]]:
        return {int(seed): self._b0_seed(int(seed)) for seed in self.config.model_seeds}

    def _mkdir(self, *parts: str) -> Path:
        path = self.output_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _predict_chunk(self, item, scores, qa_by_chunk, ranked_ids, method: str) -> dict[str, Any]:
        top = int(ranked_ids[0]) if ranked_ids else int(item.chunks[0].chunk_id)
        qa = qa_by_chunk.get(top) or self.qa_store.get((item.question_id, top)) or {"qa_answer": "", "qa_chunk_id": top}
        return chunk_prediction(
            item,
            ranking_method=method,
            scores=scores,
            qa=qa,
            flags=self.flags,
            skip_bertscore=self.config.skip_bertscore,
        )

    def _record(self, name: str, rows: list[dict[str, Any]], dest: Path) -> dict[str, Any]:
        dest.mkdir(parents=True, exist_ok=True)
        self.predictions[name] = rows
        summary = summarize_rows(rows)
        self.summaries[name] = summary
        write_jsonl(dest / "predictions_scored.jsonl", rows)
        write_json(dest / "metrics.json", summary)
        self.progress.log(f"{name} done", n=len(rows), token_f1=summary.get("token_f1"))
        return summary

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v4_1())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        if self.config.smoke:
            records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
            self.splits = build_cv_splits(self.unlocked, n_folds=5, seed=self.config.seed)
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
                        "token_f1": 100.0 if ranked and ranked[0] in pos else 0.0,
                        "exact_match": 1.0 if ranked and ranked[0] in pos else 0.0,
                        "hit@1": 1.0 if ranked and ranked[0] in pos else 0.0,
                        "hit@5": 1.0 if pos & set(ranked[:5]) else 0.0,
                        "hit@10": 1.0 if pos & set(ranked[:10]) else 0.0,
                        "hn@1": 0.0,
                        "mrr": 1.0 if ranked and ranked[0] in pos else 0.0,
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
        summary = summarize_rows(self.b0_rows)
        self.predictions["B0-live"] = self.b0_rows
        self.summaries["B0-live"] = summary
        write_json(dest / "metrics.json", summary)
        write_json(
            dest / "manifest.json",
            {
                "name": "B0-live",
                "source": "artifacts/qa_chunk_selection_v3/analysis/oof/b0",
                "n": len(self.b0_rows),
                "seeds": sorted({int(row.get("seed") or 0) for row in self.b0_rows}),
                "note": "canonical pairwise from historical V3 OOF jsonl, not V4 rescored scale",
            },
        )
        identity = {}
        ok = True
        if not self.config.smoke:
            for key, expected in HISTORICAL_B0.items():
                got = summary.get(key) if key != "n" else len(self.b0_rows)
                close = metric_close(float(got), float(expected), tol=PAIRWISE_TOLERANCE if "pairwise" in key else B0_IDENTITY_TOLERANCE)
                identity[key] = {"expected": expected, "got": got, "ok": close}
                ok = ok and close
        clone_delta = paired_delta(self.b0_rows, self.b0_rows, n_boot=min(self.config.n_bootstrap, 200 if self.config.smoke else self.config.n_bootstrap), seed=self.config.seed)
        identity_cmp = {
            "mean_delta": clone_delta.get("mean_delta"),
            "win": clone_delta.get("win"),
            "lose": clone_delta.get("lose"),
            "tie": clone_delta.get("tie"),
            "n_rows": clone_delta.get("n_rows"),
            "ok": abs(float(clone_delta.get("mean_delta") or 0.0)) <= 1e-12
            and int(clone_delta.get("win") or 0) == 0
            and int(clone_delta.get("lose") or 0) == 0,
        }
        pairwise_ok = True
        if not self.config.smoke:
            pairwise_ok = metric_close(float(summary.get("pairwise_clean_hn") or 0.0), HISTORICAL_B0["pairwise_clean_hn"], tol=PAIRWISE_TOLERANCE)
        regression = {
            "b0_identity": identity,
            "b0_identity_ok": ok,
            "paired_identity": identity_cmp,
            "pairwise_identity": {
                "got": summary.get("pairwise_clean_hn"),
                "expected": HISTORICAL_B0["pairwise_clean_hn"],
                "ok": pairwise_ok,
                "hard_safeguard": pairwise_ok,
            },
            "seed_design": seed_design(self.b0_rows),
        }
        write_json(self.output_dir / "regression" / "b0_identity.json", identity)
        write_json(self.output_dir / "regression" / "paired_identity.json", identity_cmp)
        write_json(self.output_dir / "regression" / "pairwise_identity.json", regression["pairwise_identity"])
        write_json(self.output_dir / "regression" / "r0_summary.json", regression)
        self.r0_ok = bool(ok and identity_cmp["ok"] and pairwise_ok) or self.config.smoke
        if not self.r0_ok:
            self.exit_code = 2
            self.progress.warn("R0 regression failed; later stages should not be trusted")
        self.progress.log("R0 B0 identity", ok=self.r0_ok, token_f1=summary.get("token_f1"), pairwise=summary.get("pairwise_clean_hn"))

    def stage_cache(self) -> None:
        by_qid = self._by_qid()
        short_rows = []
        needed: set[tuple[str, int]] = set()
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                if qid not in by_qid:
                    continue
                payload = {
                    "question_id": qid,
                    "document_id": row.get("document_id"),
                    "seed": int(seed),
                    "top5": top_k_ids(row, 5),
                    "top10": top_k_ids(row, 10),
                }
                short_rows.append(payload)
                for cid in top_k_ids(row, max(self.config.top_k_values)):
                    needed.add((qid, int(cid)))
        write_jsonl(self.output_dir / "cache" / "shortlists.jsonl", short_rows)
        cluster_rows = []
        for index, item in enumerate(self.unlocked, start=1):
            cluster_rows.extend(document_source_clusters(item, min_overlap=self.config.source_overlap))
            self.progress.count(index, len(self.unlocked), item="source_clusters")
        write_jsonl(self.output_dir / "cache" / "source_overlap_clusters.jsonl", cluster_rows)
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
        write_jsonl(
            self.output_dir / "cache" / "qa_topk.jsonl",
            [{**row, "question_id": qid, "chunk_id": cid} for (qid, cid), row in self.qa_store.items()],
        )
        sample = collect_v6_feature_rows(
            list(self.unlocked)[: min(len(self.unlocked), 40)],
            self._b0_seed(self.config.seed),
            self.qa_store,
            k=self.config.primary_k,
            min_overlap=self.config.source_overlap,
            limit=200,
        )
        contract = feature_contract_payload(sample)
        write_json(self.output_dir / "regression" / "v6_feature_contract.json", contract)
        write_json(self.output_dir / "v2_a4_contract.json", {
            "variant": "V2-A4",
            "formula": "max B0 score + lambda * log(1+N_clusters)",
            "lambda": self.config.v2_a4_lambda,
            "answer_normalization": "experiments.qa_chunk_selection.v4.answers.normalize_predicted_answer",
            "source_overlap": self.config.source_overlap,
        })
        self.progress.log("cache ready", n_qa=len(self.qa_store), n_shortlists=len(short_rows), fallback_hits=self.cache.stats().get("fallback_hits"))

    def stage_r0_v6_repro(self) -> None:
        if not self.r0_ok and not self.config.allow_r1_without_repro:
            self.progress.warn("skipping V6 reproduction because R0 failed")
            self.repro_ok = False
            return
        by_qid = self._by_qid()
        rows = []
        invariant_ok = True
        n_checked = 0
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                item = by_qid.get(qid)
                if item is None:
                    continue
                qa_by_chunk = {
                    cid: self.qa_store[(qid, cid)]
                    for cid in top_k_ids(row, self.config.primary_k)
                    if (qid, cid) in self.qa_store
                }
                ranked, scores = apply_v6(
                    item,
                    row,
                    qa_by_chunk,
                    alpha=V4_V6_WEIGHTS["alpha"],
                    beta=V4_V6_WEIGHTS["beta"],
                    gamma=V4_V6_WEIGHTS["gamma"],
                    k=self.config.primary_k,
                    min_overlap=self.config.source_overlap,
                )
                pred = self._predict_chunk(item, scores, qa_by_chunk, ranked, "V4-V6-repro")
                pred["seed"] = int(seed)
                rows.append(pred)
                isolated, iso_scores, _ = apply_v6_isolated(
                    item,
                    row,
                    qa_by_chunk,
                    alpha=V4_V6_WEIGHTS["alpha"],
                    beta=V4_V6_WEIGHTS["beta"],
                    gamma=V4_V6_WEIGHTS["gamma"],
                    k=self.config.primary_k,
                    min_overlap=self.config.source_overlap,
                )
                n_checked += 1
                if not shortlist_set_ok(row, isolated, k=self.config.primary_k):
                    invariant_ok = False
        summary = summarize_rows(rows)
        expected = HISTORICAL_V4_V6_TOKEN_F1
        delta = abs(float(summary.get("token_f1") or 0.0) - expected)
        ok = self.config.smoke or delta <= REPRO_F1_TOLERANCE
        payload = {
            "n": len(rows),
            "token_f1": summary.get("token_f1"),
            "expected": expected,
            "abs_delta": delta,
            "ok": ok,
            "shortlist_invariant_ok": invariant_ok,
            "n_invariant_checked": n_checked,
            "weights": dict(V4_V6_WEIGHTS),
            "note": "reproduction uses V4 apply_v6 (legacy outside-topK scores). R1 uses isolated argmax.",
        }
        write_json(self.output_dir / "regression" / "v6_run0_reproduction.json", payload)
        write_json(self.output_dir / "regression" / "shortlist_invariants.json", {"ok": invariant_ok, "n": n_checked})
        self._record("V4-V6-repro", rows, self._mkdir("regression", "v6_repro"))
        self.repro_ok = bool(ok and invariant_ok)
        if not self.repro_ok:
            self.exit_code = 2
            self.progress.warn("V4 V6 reproduction failed", **{k: payload[k] for k in ("token_f1", "abs_delta", "ok")})
        else:
            self.progress.log("V4 V6 reproduction ok", token_f1=summary.get("token_f1"))

    def _v6_predict_item(self, item, b0_row, *, alpha, beta, gamma, k, knockout=None, method="V6"):
        qa_by_chunk = {
            cid: self.qa_store[(item.question_id, cid)]
            for cid in top_k_ids(b0_row, k)
            if (item.question_id, cid) in self.qa_store
        }
        ranked, scores, _feats = apply_v6_isolated(
            item, b0_row, qa_by_chunk, alpha=alpha, beta=beta, gamma=gamma, k=k, min_overlap=self.config.source_overlap, knockout=knockout
        )
        pred = self._predict_chunk(item, scores, qa_by_chunk, ranked, method)
        pred["v6_scores"] = {str(cid): float(scores[cid]) for cid in top_k_ids(b0_row, k) if cid in scores}
        pred["shortlist_ok"] = shortlist_set_ok(b0_row, ranked, k=k)
        return pred, ranked, scores, qa_by_chunk

    def _require_repro(self) -> bool:
        if self.config.smoke or self.config.allow_r1_without_repro:
            return True
        if not (self.r0_ok and self.repro_ok):
            self.progress.warn("R1+ blocked: R0/reproduction failed. Pass --allow-r1-without-repro to override.")
            self.exit_code = 2
            return False
        return True

    def _answer_groups(self, qa_by_chunk: dict[int, dict[str, Any]], top: list[int]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for cid in top:
            qa = qa_by_chunk.get(int(cid)) or {}
            key = str(qa.get("normalized_answer") or normalize_predicted_answer(str(qa.get("qa_answer") or "")))
            if key:
                groups[key].append(int(cid))
        return groups

    def _agg_prediction(self, item, *, method: str, answer: str, cids: list[int], provenance: int, n_clusters: int, scores, seed: int, qa_hint=None) -> dict[str, Any]:
        labels = []
        for cid in cids:
            chunk = item.chunk_by_id(int(cid))
            labels.append(chunk.label if chunk is not None else "none")
        pred = aggregation_prediction(
            item,
            ranking_method=method,
            answer=answer,
            supporting_chunk_ids=cids,
            provenance_chunk_id=int(provenance),
            n_source_clusters=int(n_clusters),
            source_labels=labels,
            scores=scores,
            skip_bertscore=self.config.skip_bertscore,
            min_overlap=self.config.source_overlap,
            qa_hint=qa_hint,
        )
        pred["seed"] = int(seed)
        return pred

    def _pick_answer_group(self, item, qa_by_chunk, groups: dict[str, list[int]], scored: list[tuple[float, str, list[int]]], p_map: dict[int, float], scores, *, method: str, seed: int) -> dict[str, Any]:
        if not scored:
            top = int(next(iter(qa_by_chunk), item.chunks[0].chunk_id))
            qa = qa_by_chunk.get(top) or {}
            return self._agg_prediction(
                item,
                method=method,
                answer=str(qa.get("qa_answer") or ""),
                cids=[top],
                provenance=top,
                n_clusters=1,
                scores=scores,
                seed=seed,
                qa_hint=qa,
            )
        scored.sort(key=lambda row: (-row[0], row[1]))
        _score, key, cids = scored[0]
        provenance = max(cids, key=lambda cid: p_map.get(int(cid), 0.0))
        answer = str((qa_by_chunk.get(int(cids[0])) or {}).get("qa_answer") or "")
        qa_hint = dict(qa_by_chunk.get(int(cids[0])) or {})
        for cid in cids:
            raw = str((qa_by_chunk.get(int(cid)) or {}).get("qa_answer") or "")
            if normalize_predicted_answer(raw) == key and raw:
                answer = raw
                qa_hint = dict(qa_by_chunk.get(int(cid)) or {})
                break
        clusters = shortlist_source_clusters(item, cids, min_overlap=self.config.source_overlap)
        return self._agg_prediction(
            item,
            method=method,
            answer=answer,
            cids=cids,
            provenance=int(provenance),
            n_clusters=len({row["evidence_cluster_id"] for row in clusters}),
            scores=scores,
            seed=seed,
            qa_hint=qa_hint,
        )

    def _global_v6_from_run_grids(self) -> dict[str, float] | None:
        buckets: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
        b0_hns: list[float] = []
        for run in sorted(self.r1_weights):
            selected_path = self.output_dir / "r1_v6_oof" / f"run{run}" / "selected_weights.json"
            grid_path = self.output_dir / "r1_v6_oof" / f"run{run}" / "cal_grid.jsonl"
            payload = load_json(selected_path) if selected_path.exists() else {}
            rows = read_jsonl(grid_path) if grid_path.exists() else list(payload.get("tried") or [])
            if payload.get("b0_cal_hn@1") is not None:
                b0_hns.append(float(payload["b0_cal_hn@1"]))
            for row in rows:
                buckets[(float(row["alpha"]), float(row["beta"]), float(row["gamma"]))].append(row)
        if not buckets:
            return None
        b0_hn = float(mean(b0_hns)) if b0_hns else 0.0
        ranked = []
        for (alpha, beta, gamma), rows in buckets.items():
            f1 = float(mean(float(row["cal_token_f1"]) for row in rows))
            hn = float(mean(float(row["cal_hn@1"]) for row in rows))
            hit = float(mean(float(row.get("cal_hit@1") or 0.0) for row in rows))
            ranked.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "cal_token_f1": f1,
                    "cal_hn@1": hn,
                    "cal_hit@1": hit,
                    "admissible": hn <= b0_hn + self.config.epsilon_hn,
                    "reader_mass": float(beta) + float(gamma),
                    "n_runs": len(rows),
                }
            )
        admissible = [row for row in ranked if row["admissible"]] or ranked
        admissible.sort(
            key=lambda row: (
                -float(row["cal_token_f1"]),
                float(row["cal_hn@1"]),
                -float(row["cal_hit@1"]),
                float(row["reader_mass"]),
                float(row["alpha"]),
                float(row["beta"]),
                float(row["gamma"]),
            )
        )
        best = admissible[0]
        write_json(self.output_dir / "final" / "global_v6_grid.json", {"selected": best, "b0_cal_hn@1": b0_hn, "n_triples": len(ranked)})
        return {"alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"]}

    def _global_lambda_from_run_grids(self) -> float | None:
        buckets: dict[float, list[float]] = defaultdict(list)
        for run in sorted(self.r3_lambdas):
            path = self.output_dir / "r3_v6_v2" / f"run{run}" / "selected_lambda.json"
            if not path.exists():
                continue
            payload = load_json(path)
            for row in payload.get("tried") or []:
                buckets[float(row["lambda_agg"])].append(float(row["cal_token_f1"]))
        if not buckets:
            return None
        best_lam, best_f1 = max(((lam, float(mean(vals))) for lam, vals in buckets.items()), key=lambda item: (item[1], -item[0]))
        write_json(self.output_dir / "final" / "global_lambda_grid.json", {"selected": best_lam, "cal_token_f1": best_f1})
        return float(best_lam)

    def stage_r1(self) -> None:
        if not self._require_repro():
            return
        oof: list[dict[str, Any]] = []
        cal_all: list[dict[str, Any]] = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            self.progress.log(f"R1 outer run {run}/{n_runs - 1}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            b0_cal = [row for row in self.b0_rows if any(row["question_id"] == item.question_id for item in cal_items)]
            choice = choose_v6_weights(
                cal_items,
                self._b0_by_seeds(),
                self.qa_store,
                b0_cal,
                k=self.config.primary_k,
                min_overlap=self.config.source_overlap,
                alpha_grid=self.config.v6_alpha_grid,
                beta_grid=self.config.v6_beta_grid,
                gamma_grid=self.config.v6_gamma_grid,
                epsilon_hn=self.config.epsilon_hn,
                predict_fn=lambda item, scores, qa_by_chunk, ranked: self._predict_chunk(item, scores, qa_by_chunk, ranked, "V6-cal"),
                seeds=self.config.model_seeds,
                progress=self.progress,
            )
            selected = choice["selected"]
            weights = {"alpha": selected["alpha"], "beta": selected["beta"], "gamma": selected["gamma"]}
            self.r1_weights[run] = weights
            run_dir = self._mkdir("r1_v6_oof", f"run{run}")
            write_json(run_dir / "selected_weights.json", choice)
            write_jsonl(run_dir / "cal_grid.jsonl", list(choice.get("tried") or []))
            for seed in self.config.model_seeds:
                seed_rows = []
                b0_map = self._b0_seed(int(seed))
                for item in eval_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred, ranked, _scores, _qa = self._v6_predict_item(item, row, **weights, k=self.config.primary_k, method="V6-K5")
                    pred["seed"] = int(seed)
                    pred["cv_run"] = int(run)
                    seed_rows.append(pred)
                    oof.append(pred)
                write_jsonl(run_dir / f"eval_predictions_seed{seed}.jsonl", seed_rows)
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in cal_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred, _, _, _ = self._v6_predict_item(item, row, **weights, k=self.config.primary_k, method="V6-K5-cal")
                    pred["seed"] = int(seed)
                    pred["cv_run"] = int(run)
                    cal_all.append(pred)
            self.progress.log(f"R1 run {run} weights", **weights, cal_f1=selected.get("cal_token_f1"))
        dest = self._mkdir("r1_v6_oof")
        summary = self._record("V6-K5", oof, dest)
        delta = paired_delta(self.b0_rows, oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed)
        hit5_ok = metric_close(float(summary.get("hit@5") or 0.0), float(self.summaries["B0-live"].get("hit@5") or 0.0), tol=1e-9) or self.config.smoke
        trans = transitions(self.b0_rows, oof)
        write_json(dest / "bootstrap.json", delta)
        write_json(dest / "transitions.json", trans)
        write_json(dest / "shortlist_invariant.json", {"hit@5_identical": hit5_ok, "n_bad": sum(1 for row in oof if not row.get("shortlist_ok", True))})
        write_json(dest / "cal_predictions_metrics.json", summarize_rows(cal_all))
        write_json(self.output_dir / "r1_v6_oof" / "weights_by_run.json", {str(k): v for k, v in self.r1_weights.items()})
        self.progress.log("R1 OOF merged", n=len(oof), token_f1=summary.get("token_f1"), dF1=delta.get("mean_delta"), hit5_ok=hit5_ok)

    def stage_r2(self) -> None:
        if "V6-K5" not in self.predictions:
            self.progress.warn("R2 skipped: no R1 OOF")
            return
        knockouts = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
        results = {}
        for knock in knockouts:
            rows = []
            for run, weights in self.r1_weights.items():
                eval_items = list(records_for_role(self.unlocked, self.splits, run=int(run), role="eval"))
                for seed in self.config.model_seeds:
                    b0_map = self._b0_seed(int(seed))
                    for item in eval_items:
                        row = b0_map.get(item.question_id)
                        if row is None:
                            continue
                        pred, _, _, _ = self._v6_predict_item(item, row, **weights, k=self.config.primary_k, knockout=knock, method=f"R2-{knock}")
                        pred["seed"] = int(seed)
                        pred["cv_run"] = int(run)
                        rows.append(pred)
            summary = self._record(f"R2-{knock}", rows, self._mkdir("r2_v6_ablation", knock))
            delta = paired_delta(self.b0_rows, rows, n_boot=min(self.config.n_bootstrap, 200 if self.config.smoke else self.config.n_bootstrap), seed=self.config.seed)
            results[knock] = {"metrics": summary, "delta_vs_b0": delta, "delta_vs_v6": paired_delta(self.predictions["V6-K5"], rows, n_boot=min(self.config.n_bootstrap, 200 if self.config.smoke else self.config.n_bootstrap), seed=self.config.seed)}
        write_json(self.output_dir / "r2_v6_ablation" / "knockout_results.json", results)
        by_label = defaultdict(list)
        by_aspect = defaultdict(list)
        for row in self.predictions["V6-K5"]:
            by_label[str(row.get("selected_chunk_label") or "none")].append(row)
            by_aspect[str(row.get("aspect") or "none")].append(row)
        write_json(
            self.output_dir / "r2_v6_ablation" / "aspect_diagnostics.json",
            {asp: summarize_rows(rows) for asp, rows in sorted(by_aspect.items())},
        )
        write_json(self.output_dir / "r2_v6_ablation" / "label_diagnostics.json", {lab: summarize_rows(rows) for lab, rows in sorted(by_label.items())})
        self.progress.log("R2 knockouts done", n=len(knockouts))

    def _aggregate_c1(self, item, b0_row, *, lam: float, seed: int, weights, k: int) -> dict[str, Any]:
        qa_by_chunk = {
            cid: self.qa_store[(item.question_id, cid)]
            for cid in top_k_ids(b0_row, k)
            if (item.question_id, cid) in self.qa_store
        }
        ranked, scores, _feats = apply_v6_isolated(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=k, min_overlap=self.config.source_overlap
        )
        top = top_k_ids(b0_row, k)
        probs = softmax([float(scores.get(cid, 0.0)) for cid in top])
        p_map = {int(cid): float(p) for cid, p in zip(top, probs)}
        fake_b0 = dict(b0_row)
        fake_b0["scores"] = {str(cid): p_map.get(int(cid), 0.0) for cid in top}
        fake_b0["ranked_ids"] = ranked[:k]
        agg = aggregate(item, fake_b0, qa_by_chunk, variant="V2-A4", k=k, lam=float(lam), min_overlap=self.config.source_overlap)
        pred = aggregation_prediction(
            item,
            ranking_method="C1",
            answer=agg["answer"],
            supporting_chunk_ids=agg["supporting_chunk_ids"],
            provenance_chunk_id=agg["provenance_chunk_id"],
            n_source_clusters=agg["n_source_clusters"],
            source_labels=agg["source_labels"],
            scores=scores,
            skip_bertscore=self.config.skip_bertscore,
            min_overlap=self.config.source_overlap,
            qa_hint=agg.get("qa"),
        )
        pred["seed"] = int(seed)
        pred["v6_top1_chunk_id"] = ranked[0] if ranked else None
        return pred

    def _aggregate_c2(self, item, b0_row, *, seed: int, weights, k: int) -> dict[str, Any]:
        qa_by_chunk = {
            cid: self.qa_store[(item.question_id, cid)]
            for cid in top_k_ids(b0_row, k)
            if (item.question_id, cid) in self.qa_store
        }
        ranked, scores, _feats = apply_v6_isolated(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=k, min_overlap=self.config.source_overlap
        )
        top = top_k_ids(b0_row, k)
        probs = softmax([float(scores.get(cid, 0.0)) for cid in top])
        p_map = {int(cid): float(p) for cid, p in zip(top, probs)}
        groups = self._answer_groups(qa_by_chunk, top)
        scored = [(sum(p_map.get(cid, 0.0) for cid in cids), key, cids) for key, cids in groups.items()]
        pred = self._pick_answer_group(item, qa_by_chunk, groups, scored, p_map, scores, method="C2", seed=int(seed))
        pred["v6_top1_chunk_id"] = ranked[0] if ranked else None
        return pred

    def _aggregate_c3(self, item, b0_row, *, lam: float, seed: int, weights, k: int) -> dict[str, Any]:
        qa_by_chunk = {
            cid: self.qa_store[(item.question_id, cid)]
            for cid in top_k_ids(b0_row, k)
            if (item.question_id, cid) in self.qa_store
        }
        ranked, scores, _feats = apply_v6_isolated(
            item, b0_row, qa_by_chunk, alpha=weights["alpha"], beta=weights["beta"], gamma=weights["gamma"], k=k, min_overlap=self.config.source_overlap
        )
        top = top_k_ids(b0_row, k)
        probs = softmax([float(scores.get(cid, 0.0)) for cid in top])
        p_map = {int(cid): float(p) for cid, p in zip(top, probs)}
        groups = self._answer_groups(qa_by_chunk, top)
        scored = []
        for key, cids in groups.items():
            clusters = shortlist_source_clusters(item, cids, min_overlap=self.config.source_overlap)
            n_clusters = len({row["evidence_cluster_id"] for row in clusters})
            score = max((p_map.get(cid, 0.0) for cid in cids), default=0.0) + float(lam) * math.log(1.0 + n_clusters) + float(len(cids))
            scored.append((score, key, cids))
        pred = self._pick_answer_group(item, qa_by_chunk, groups, scored, p_map, scores, method="C3", seed=int(seed))
        pred["v6_top1_chunk_id"] = ranked[0] if ranked else None
        return pred

    def stage_r3(self) -> None:
        if not self.r1_weights:
            self.progress.warn("R3 skipped: no R1 weights")
            return
        by_qid = self._by_qid()
        c0_rows = []
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                item = by_qid.get(qid)
                if item is None:
                    continue
                qa_by_chunk = {
                    cid: self.qa_store[(qid, cid)]
                    for cid in top_k_ids(row, self.config.primary_k)
                    if (qid, cid) in self.qa_store
                }
                agg = aggregate(
                    item, row, qa_by_chunk, variant="V2-A4", k=self.config.primary_k, lam=self.config.v2_a4_lambda, min_overlap=self.config.source_overlap
                )
                pred = aggregation_prediction(
                    item,
                    ranking_method="C0-V2-A4",
                    answer=agg["answer"],
                    supporting_chunk_ids=agg["supporting_chunk_ids"],
                    provenance_chunk_id=agg["provenance_chunk_id"],
                    n_source_clusters=agg["n_source_clusters"],
                    source_labels=agg["source_labels"],
                    scores=agg["scores"],
                    skip_bertscore=self.config.skip_bertscore,
                    min_overlap=self.config.source_overlap,
                    qa_hint=agg.get("qa"),
                )
                pred["seed"] = int(seed)
                c0_rows.append(pred)
        c0_summary = self._record("C0-V2-A4", c0_rows, self._mkdir("r3_v6_v2", "C0"))
        c0_ok = self.config.smoke or metric_close(float(c0_summary.get("token_f1") or 0.0), HISTORICAL_V4_V2A4_TOKEN_F1, tol=0.25)
        write_json(
            self.output_dir / "r3_v6_v2" / "control_v2_a4.json",
            {"token_f1": c0_summary.get("token_f1"), "expected": HISTORICAL_V4_V2A4_TOKEN_F1, "ok": c0_ok, "note": "tolerance 0.25 F1 for cache/rounding; V4 used lambda=1.0"},
        )
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            weights = self.r1_weights[run]
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            choice = choose_lambda_agg(
                cal_items,
                self._b0_by_seeds(),
                self.qa_store,
                weights=weights,
                k=self.config.primary_k,
                min_overlap=self.config.source_overlap,
                grid=self.config.lambda_agg_grid,
                aggregate_fn=lambda item, row, lam, seed: self._aggregate_c1(item, row, lam=lam, seed=seed, weights=weights, k=self.config.primary_k),
                seeds=self.config.model_seeds,
                progress=self.progress,
            )
            lam = float(choice["selected"]["lambda_agg"])
            self.r3_lambdas[run] = lam
            run_dir = self._mkdir("r3_v6_v2", f"run{run}")
            write_json(run_dir / "selected_lambda.json", choice)
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                seed_rows = []
                for item in eval_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred = self._aggregate_c1(item, row, lam=lam, seed=int(seed), weights=weights, k=self.config.primary_k)
                    pred["cv_run"] = int(run)
                    seed_rows.append(pred)
                    oof.append(pred)
                write_jsonl(run_dir / f"eval_predictions_seed{seed}.jsonl", seed_rows)
            self.progress.log(f"R3 run {run} lambda_agg={lam}")
        summary = self._record("C1", oof, self._mkdir("r3_v6_v2"))
        write_json(self.output_dir / "r3_v6_v2" / "bootstrap.json", paired_delta(self.predictions["V6-K5"], oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed))
        write_json(self.output_dir / "r3_v6_v2" / "bootstrap_vs_b0.json", paired_delta(self.b0_rows, oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed))
        c2_rows = []
        c3_rows = []
        for run in range(n_runs):
            weights = self.r1_weights[run]
            lam = float(self.r3_lambdas.get(run, self.config.v2_a4_lambda))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in eval_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    c2 = self._aggregate_c2(item, row, seed=int(seed), weights=weights, k=self.config.primary_k)
                    c2["cv_run"] = int(run)
                    c2_rows.append(c2)
                    c3 = self._aggregate_c3(item, row, lam=lam, seed=int(seed), weights=weights, k=self.config.primary_k)
                    c3["cv_run"] = int(run)
                    c3_rows.append(c3)
        self._record("C2", c2_rows, self._mkdir("r3_v6_v2", "C2"))
        self._record("C3", c3_rows, self._mkdir("r3_v6_v2", "C3"))
        write_json(
            self.output_dir / "r3_v6_v2" / "diagnostic_note.json",
            {"C2": "sum p_V6 over answer group; diagnostic only", "C3": "C1 + answer frequency; diagnostic only", "primary": "C1"},
        )
        self.progress.log("R3 C1 OOF", token_f1=summary.get("token_f1"))

    def stage_r4(self) -> None:
        if not self.r1_weights:
            self.progress.warn("R4 skipped: no R1")
            return
        k10 = 10
        oof_v6 = []
        oof_c1 = []
        weights_k10: dict[int, dict[str, float]] = {}
        n_runs = len(self.splits.get("runs") or [])
        for run in range(n_runs):
            self.progress.log(f"R4 K=10 outer run {run}")
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            b0_cal = [row for row in self.b0_rows if any(row["question_id"] == item.question_id for item in cal_items)]
            choice = choose_v6_weights(
                cal_items,
                self._b0_by_seeds(),
                self.qa_store,
                b0_cal,
                k=k10,
                min_overlap=self.config.source_overlap,
                alpha_grid=self.config.v6_alpha_grid,
                beta_grid=self.config.v6_beta_grid,
                gamma_grid=self.config.v6_gamma_grid,
                epsilon_hn=self.config.epsilon_hn,
                predict_fn=lambda item, scores, qa_by_chunk, ranked: self._predict_chunk(item, scores, qa_by_chunk, ranked, "V6-K10-cal"),
                seeds=self.config.model_seeds,
                progress=self.progress,
            )
            weights = {"alpha": choice["selected"]["alpha"], "beta": choice["selected"]["beta"], "gamma": choice["selected"]["gamma"]}
            weights_k10[run] = weights
            run_dir = self._mkdir("r4_k", "k10", f"run{run}")
            write_json(run_dir / "selected_weights.json", choice)
            lam_choice = choose_lambda_agg(
                cal_items,
                self._b0_by_seeds(),
                self.qa_store,
                weights=weights,
                k=k10,
                min_overlap=self.config.source_overlap,
                grid=self.config.lambda_agg_grid,
                aggregate_fn=lambda item, row, lam, seed, w=weights: self._aggregate_c1(item, row, lam=lam, seed=seed, weights=w, k=k10),
                seeds=self.config.model_seeds,
                progress=self.progress,
            )
            lam = float(lam_choice["selected"]["lambda_agg"])
            write_json(run_dir / "selected_lambda.json", lam_choice)
            for seed in self.config.model_seeds:
                b0_map = self._b0_seed(int(seed))
                for item in eval_items:
                    row = b0_map.get(item.question_id)
                    if row is None:
                        continue
                    pred, _, _, _ = self._v6_predict_item(item, row, **weights, k=k10, method="V6-K10")
                    pred["seed"] = int(seed)
                    pred["cv_run"] = int(run)
                    oof_v6.append(pred)
                    c1 = self._aggregate_c1(item, row, lam=lam, seed=int(seed), weights=weights, k=k10)
                    c1["cv_run"] = int(run)
                    oof_c1.append(c1)
        self._record("V6-K10", oof_v6, self._mkdir("r4_k", "k10"))
        if oof_c1:
            self._record("C1-K10", oof_c1, self._mkdir("r4_k", "c1_k10"))
        k5 = self.predictions.get("V6-K5") or []
        write_json(
            self.output_dir / "r4_k" / "paired_comparison.json",
            {
                "k5_vs_b0": paired_delta(self.b0_rows, k5, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if k5 else {},
                "k10_vs_b0": paired_delta(self.b0_rows, oof_v6, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed),
                "k10_vs_k5": paired_delta(k5, oof_v6, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed) if k5 else {},
                "weights_k10": {str(k): v for k, v in weights_k10.items()},
            },
        )
        write_json(
            self.output_dir / "r4_k" / "efficiency.json",
            {
                "qa_store_unique": len(self.qa_store),
                "cache": self.cache.stats(),
                "note": "union cache is not live per-question cost; live inference still scales with K",
            },
        )
        write_json(self.output_dir / "r4_k" / "k5" / "note.json", {"alias": "R1 V6-K5 OOF", "metrics": self.summaries.get("V6-K5")})
        self.progress.log("R4 K=10 OOF", token_f1=self.summaries.get("V6-K10", {}).get("token_f1"))

    def stage_r5(self) -> None:
        if not self._require_repro():
            return
        answer_fn = self._answer_fn()
        oof = []
        n_runs = len(self.splits.get("runs") or [])
        seed = int(self.config.trainable_seeds[0])
        exploratory = tuple(self.config.trainable_seeds) != tuple(self.config.model_seeds)
        for run in range(n_runs):
            self.progress.log(f"R5 utility run {run} seed={seed}")
            train_items = list(records_for_role(self.unlocked, self.splits, run=run, role="train"))
            cal_items = list(records_for_role(self.unlocked, self.splits, run=run, role="calibration"))
            eval_items = list(records_for_role(self.unlocked, self.splits, run=run, role="eval"))
            mined = mine_utility_positives(
                train_items,
                answer_fn,
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash="v4.1-utility-u80",
                f1_threshold=self.config.utility_f1_pct,
                progress=self.progress,
            )
            primary = []
            diagnostic_overlap = []
            train_by_qid = {item.question_id: item for item in train_items}
            for row in mined:
                item = train_by_qid.get(str(row["question_id"]))
                if item is None:
                    continue
                pos_ids = {int(chunk.chunk_id) for chunk in item.positives}
                overlap = overlapping_ids(item, int(row["chunk_id"]), min_overlap=self.config.source_overlap)
                row = dict(row)
                row["overlaps_positive_cluster"] = bool(overlap & pos_ids)
                if row["overlaps_positive_cluster"]:
                    diagnostic_overlap.append(row)
                else:
                    primary.append(row)
            run_dir = self._mkdir("r5_utility", f"run{run}")
            write_jsonl(run_dir / "utility_candidates.jsonl", mined)
            write_json(
                run_dir / "utility_summary.json",
                {
                    "n_mined": len(mined),
                    "n_primary_new_region": len(primary),
                    "n_overlap_diagnostic": len(diagnostic_overlap),
                    "threshold_pct": self.config.utility_f1_pct,
                    "scale": "0..100",
                    "note": "utility_positive_candidate, not target_evidence",
                },
            )
            best = None
            for eta in self.config.eta_grid:
                out = run_dir / "models" / f"eta{eta}" / f"seed{seed}"
                trained = train_utility_listwise(
                    train_items,
                    cal_items,
                    self.flags_rows,
                    utility_index(primary),
                    seed=seed,
                    device=self._resolved_device(),
                    model_name=self.config.cross_encoder_model,
                    output_dir=out,
                    epochs=self.config.b0_epochs,
                    patience=self.config.b0_patience,
                    learning_rate=self.config.b0_learning_rate,
                    max_length=self.config.b0_max_length,
                    temperature=self.config.listwise_temperature,
                    easy_cap=self.config.easy_negative_cap,
                    eta=float(eta),
                    smoke=self.config.smoke,
                    progress=self.progress,
                )
                scorer = score_fn_for(trained["scorer"])
                cal_rows = []
                from experiments.qa_chunk_selection.v2.schemas import to_runtime

                for item in cal_items:
                    runtime = to_runtime(item)
                    scores = {int(chunk.chunk_id): float(score) for chunk, score in zip(item.chunks, scorer(runtime))}
                    ranked = rank_by_scores(list(scores), [scores[cid] for cid in scores])
                    pred = self._predict_chunk(item, scores, {}, ranked, "R5-cal")
                    pred["seed"] = seed
                    cal_rows.append(pred)
                f1 = float(summarize_rows(cal_rows).get("token_f1") or 0.0)
                payload = {"eta": float(eta), "cal_token_f1": f1, "scorer": trained["scorer"]}
                if best is None or f1 > float(best["cal_token_f1"]):
                    best = payload
            assert best is not None
            write_json(run_dir / "selected_eta.json", {"eta": best["eta"], "cal_token_f1": best["cal_token_f1"]})
            scorer = score_fn_for(best["scorer"])
            from experiments.qa_chunk_selection.v2.schemas import to_runtime

            for item in eval_items:
                runtime = to_runtime(item)
                scores = {int(chunk.chunk_id): float(score) for chunk, score in zip(item.chunks, scorer(runtime))}
                ranked = rank_by_scores(list(scores), [scores[cid] for cid in scores])
                top = ranked[0]
                qa = self.qa_store.get((item.question_id, top)) or answer_one(
                    item, top, self._answer_fn(), cache=self.cache, model_id=self._model_id(), inference_hash="v4.1-utility-eval"
                )
                pred = chunk_prediction(item, ranking_method="R5-U80", scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)
                pred["seed"] = seed
                pred["cv_run"] = int(run)
                oof.append(pred)
        summary = self._record("R5-U80", oof, self._mkdir("r5_utility"))
        b0_s = [row for row in self.b0_rows if int(row.get("seed") or 0) == seed]
        write_json(
            self.output_dir / "r5_utility" / "bootstrap.json",
            paired_delta(b0_s, oof, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed),
        )
        write_json(
            self.output_dir / "r5_utility" / "status.json",
            {
                "exploratory_single_seed": exploratory,
                "seeds": list(self.config.trainable_seeds),
                "IN_SAMPLE_CONTAMINATED": False,
                "eval_only": True,
                "note": "cannot automatically beat three-seed primary without protocol amendment",
            },
        )
        self.progress.log("R5 OOF", n=len(oof), token_f1=summary.get("token_f1"), exploratory=exploratory)

    def stage_compare(self) -> None:
        b0 = self.predictions.get("B0-live") or []
        b0_design = seed_design(b0)
        b0_hn = float((self.summaries.get("B0-live") or {}).get("hn@1") or 0.0)
        pairwise_ok = bool((load_json(self.output_dir / "regression" / "pairwise_identity.json") or {}).get("ok")) if (self.output_dir / "regression" / "pairwise_identity.json").exists() else False
        table = []
        deltas = {}
        trans = {}
        admissible = []
        excluded = []
        allowed = set(PRIMARY_FINAL_SYSTEMS)
        for name, rows in self.predictions.items():
            summary = self.summaries.get(name) or summarize_rows(rows)
            design = seed_design(rows)
            row = {
                "system": name,
                **{k: summary.get(k) for k in ("n", "token_f1", "exact_match", "hit@1", "mrr", "hn@1", "pairwise_clean_hn", "ambiguous_at_1_original", "gsrn_at_1_original")},
                **design,
            }
            if name == "B0-live":
                row["admissible"] = True
                table.append(row)
                continue
            delta = paired_delta(b0, rows, n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed)
            deltas[name] = delta
            trans[name] = transitions(b0, rows)
            hn_ok = float(summary.get("hn@1") or 0.0) <= b0_hn + self.config.epsilon_hn
            row["hn_safeguard"] = hn_ok
            row["pairwise_safeguard_used"] = bool(pairwise_ok)
            leak = False
            if name.startswith("R5"):
                status_path = self.output_dir / "r5_utility" / "status.json"
                leak = bool((load_json(status_path) or {}).get("IN_SAMPLE_CONTAMINATED")) if status_path.exists() else False
            comparable = bool(design.get("three_seed"))
            auto_winner_ok = name in allowed and comparable and not leak and hn_ok
            row["admissible"] = auto_winner_ok
            row["comparable_matched_seed"] = comparable
            if auto_winner_ok:
                admissible.append(name)
            else:
                excluded.append(name)
            table.append(row)
        selected = "B0-live"
        best = float((self.summaries.get("B0-live") or {}).get("token_f1") or 0.0)
        for name in admissible:
            f1 = float((self.summaries.get(name) or {}).get("token_f1") or 0.0)
            delta = deltas.get(name) or {}
            low = delta.get("ci95_low")
            ci_ok = True
            if name in {"V6-K5", "C1", "V6-K10", "C1-K10"} and low is not None and float(low) <= 0:
                ci_ok = False
            if f1 > best and ci_ok:
                best = f1
                selected = name
        payload = {
            "table": table,
            "deltas": deltas,
            "transitions": trans,
            "selected": selected,
            "admissible": admissible,
            "excluded_in_sample": excluded,
            "b0_seed_design": b0_design,
            "b0_token_f1": (self.summaries.get("B0-live") or {}).get("token_f1"),
            "v6_k5_token_f1": (self.summaries.get("V6-K5") or {}).get("token_f1"),
            "c1_token_f1": (self.summaries.get("C1") or {}).get("token_f1"),
            "systems": list(self.predictions),
            "internal_test_locked": True,
            "selection_rule": "primary final systems only; three-seed; HN-ok; CI excluding 0 for V6/C1",
        }
        write_json(self.output_dir / "analysis" / "compare.json", payload)
        write_json(
            self.output_dir / "analysis" / "selection.json",
            {
                "selected": selected,
                "admissible": admissible,
                "b0_live_kept": True,
                "seed_design_checked": True,
                "pairing": "question_id+seed",
                "IN_SAMPLE_CONTAMINATED": False,
            },
        )
        self._compare_payload = payload
        self.progress.log("compare done", selected=selected, n_systems=len(self.predictions))

    def stage_freeze(self) -> None:
        payload = getattr(self, "_compare_payload", None)
        if payload is None and (self.output_dir / "analysis" / "compare.json").exists():
            payload = load_json(self.output_dir / "analysis" / "compare.json")
        selected = (payload or {}).get("selected") or "B0-live"
        self._mkdir("final")
        global_weights = self._global_v6_from_run_grids() if self.r1_weights else None
        if global_weights is None and self.r1_weights:
            triples = [tuple(round(self.r1_weights[run][k], 6) for k in ("alpha", "beta", "gamma")) for run in sorted(self.r1_weights)]
            counts: dict[tuple[float, float, float], int] = defaultdict(int)
            for triple in triples:
                counts[triple] += 1
            picked = max(counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1], -item[0][2]))[0]
            global_weights = {"alpha": picked[0], "beta": picked[1], "gamma": picked[2]}
        global_lambda = self._global_lambda_from_run_grids()
        k_choice = 5
        if "V6-K10" in self.summaries and "V6-K5" in self.summaries:
            d = paired_delta(self.predictions["V6-K5"], self.predictions["V6-K10"], n_boot=self.config.n_bootstrap if not self.config.smoke else 50, seed=self.config.seed)
            if float(d.get("mean_delta") or 0.0) > 0 and float(d.get("ci95_low") or -1) > 0:
                k_choice = 10
        r5_status = "excluded"
        if (self.output_dir / "r5_utility" / "status.json").exists():
            r5_status = "exploratory_excluded_from_final_list"
        hashes = load_json(self.output_dir / "source_hashes.json") if (self.output_dir / "source_hashes.json").exists() else {}
        manifest = {
            "dataset_hash": hashes.get("dataset"),
            "split_hash": hashes.get("splits"),
            "b0_live": "historical V3-B0 max5 OOF",
            "b0_final": "NOT trained in this default run; INTERNAL_TEST remains locked",
            "qa_inference_hash": self.config.qa_inference_hash,
            "answer_normalization": "experiments.qa_chunk_selection.v4.answers.normalize_predicted_answer",
            "source_overlap": self.config.source_overlap,
            "final_architecture": selected,
            "final_k": k_choice,
            "final_v6_weights_global_from_runs": global_weights,
            "final_lambda_agg_global": global_lambda,
            "per_run_v6_weights": {str(k): v for k, v in self.r1_weights.items()},
            "v6_feature_contract": "regression/v6_feature_contract.json",
            "v2_a4_contract": "v2_a4_contract.json",
            "utility_branch_status": r5_status,
            "bootstrap": "mean over seeds within question, cluster document_id, n_boot>=5000",
            "primary_endpoint": "Token F1",
            "hn_safeguard": self.config.epsilon_hn,
            "internal_test": "locked",
            "no_config_changes_after_this_file": True,
        }
        self._mkdir("final")
        write_json(self.output_dir / "final" / "model_selection.json", {"selected": selected, "admissible": (payload or {}).get("admissible")})
        write_json(self.output_dir / "final" / "hyperparameter_selection.json", {"k": k_choice, "v6_weights": global_weights, "lambda_agg": global_lambda, "lambdas_by_run": {str(k): v for k, v in self.r3_lambdas.items()}})
        write_json(self.output_dir / "final" / "b0_final_training_manifest.json", {"trained": False, "reason": "INTERNAL_TEST not opened; epoch rule not applied on test"})
        write_json(self.output_dir / "final" / "freeze_manifest.json", manifest)
        self.progress.log("freeze manifest written; INTERNAL_TEST still locked", selected=selected, k=k_choice)

    def stage_report(self) -> None:
        payload = getattr(self, "_compare_payload", None)
        if payload is None:
            path = self.output_dir / "analysis" / "compare.json"
            payload = load_json(path) if path.exists() else {"internal_test_locked": True}
        write_report(self.output_dir, payload)
        self.progress.log("report written", path=str(self.output_dir / "reports" / "results_v4_1.md"))
