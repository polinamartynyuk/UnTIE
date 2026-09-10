"""Isolated V4 pipeline. Prior experiment artifacts are read-only."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from experiments.qa_chunk_selection.models import load_frozen_qa
from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.hn_flags import flag_index
from experiments.qa_chunk_selection.v3.contract import LOCKED_SPLITS
from experiments.qa_chunk_selection.v3.data import records_from_jsonl, split_unlocked, synthetic_records
from experiments.qa_chunk_selection.v3.split import records_for_role
from experiments.qa_chunk_selection.v3.train import load_ranker, score_fn_for

from .answers import normalize_predicted_answer
from .clusters import document_source_clusters
from .config import ExperimentConfigV4
from .contract import experiment_contract_v4
from .evaluate import aggregation_prediction, chunk_prediction, paired_delta, qa_fields_from_row, summarize_rows, transitions
from .heads import LinearHead, rank_by_scores
from .io_guard import assert_isolated_output, file_sha256, load_json, read_jsonl, write_json, write_jsonl
from .keybert_features import KeyphraseExtractor, document_idf, document_profile_phrases, resolve_encoder_device
from .qa_shortlist import answer_one
from .report import write_report
from .shortlist import b0_rows_for_seed, index_b0_oof, load_b0_oof, scores_from_row, top_k_ids
from .v1_keybert import VARIANTS as V1_VARIANTS, choose_lambda, rerank_variant
from .v2_aggregate import VARIANTS as V2_VARIANTS, aggregate
from .v3_utility import mine_utility_positives, train_utility_listwise, utility_index
from .v4_ownership import pair_documents, probe_pair, profile_vector, summarize_diagnostic
from .v5_calibrate import FEATURE_NAMES, apply_v5, collect_training
from .v6_reader import apply_v6, choose_v6_weights
from .v7_reconsider import score_v7, train_v7


class PipelineV4:
    def __init__(self, config: ExperimentConfigV4) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(Path(config.output_dir))
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.v3 = Path(config.v3_artifact_dir)
        self.cache = FeatureCache(self.output_dir / "cache")
        self.unlocked = ()
        self.splits: dict[str, Any] = {}
        self.flags_rows: list[dict[str, Any]] = []
        self.flags = {}
        self.b0_rows: list[dict[str, Any]] = []
        self.b0_index: dict[tuple[str, int], dict[str, Any]] = {}
        self.phrases: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.idf: dict[str, float] = {}
        self.qa_store: dict[tuple[str, int], dict[str, Any]] = {}
        self.profiles: dict[str, list[str]] = {}
        self.profile_vecs: dict[str, list[float]] = {}
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.summaries: dict[str, dict[str, Any]] = {}
        self.bundle = None
        self.extractor: KeyphraseExtractor | None = None
        self.exit_code = 0
        self._handlers: dict[str, Callable[[], None]] = {
            "init": self.stage_init,
            "freeze-b0": self.stage_freeze_b0,
            "shortlist": self.stage_shortlist,
            "clusters": self.stage_clusters,
            "keybert": self.stage_keybert,
            "qa-topk": self.stage_qa_topk,
            "v1": self.stage_v1,
            "v2": self.stage_v2,
            "v3": self.stage_v3,
            "v4": self.stage_v4,
            "v5": self.stage_v5,
            "v6": self.stage_v6,
            "v7": self.stage_v7,
            "compare": self.stage_compare,
            "report": self.stage_report,
        }

    def run(self, stages: list[str] | None = None) -> int:
        names = list(stages or self.config.stages)
        self.progress.stage(
            "start",
            f"output={self.output_dir} smoke={self.config.smoke} device={self.config.device}->{self._resolved_device()} stages={names}",
        )
        self.progress.log("E0/V2/V3/V3.1/D45 are read-only", write_target=str(self.output_dir))
        for name in names:
            handler = self._handlers.get(name)
            if handler is None:
                raise ValueError(f"unknown V4 stage: {name}")
            self.progress.stage(name)
            handler()
        self.progress.stage("done", f"cache_hit_rate={self.cache.stats().get('hit_rate')}")
        return self.exit_code

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

    def _resolved_device(self) -> str:
        return "cpu" if self.config.smoke else resolve_encoder_device(self.config.device)

    def _extractor(self) -> KeyphraseExtractor:
        if self.extractor is None:
            self.extractor = KeyphraseExtractor(
                backend=self.config.keybert_backend,
                top_n=self.config.keybert_n,
                ngram_low=self.config.keybert_ngram_low,
                ngram_high=self.config.keybert_ngram_high,
                use_mmr=self.config.keybert_use_mmr,
                smoke=self.config.smoke,
                cache=self.cache,
                device=self._resolved_device(),
                progress=self.progress,
            )
        return self.extractor

    def _by_qid(self) -> dict[str, Any]:
        return {item.question_id: item for item in self.unlocked}

    def _b0_seed(self, seed: int) -> dict[str, dict[str, Any]]:
        if self.b0_index:
            return b0_rows_for_seed(self.b0_index, seed)
        return {}

    def _synthetic_b0(self) -> list[dict[str, Any]]:
        rows = []
        for seed in self.config.model_seeds:
            for item in self.unlocked:
                scores = {}
                for chunk in item.chunks:
                    # label-free: longer + earlier chunks score slightly higher
                    scores[int(chunk.chunk_id)] = float(len(chunk.text.split())) - 0.01 * float(chunk.position)
                ranked = rank_by_scores(list(scores), [scores[cid] for cid in scores])
                rows.append(
                    {
                        "question_id": item.question_id,
                        "document_id": item.document_id,
                        "aspect": item.aspect,
                        "seed": int(seed),
                        "cv_run": 0,
                        "scores": {str(k): v for k, v in scores.items()},
                        "ranked_ids": ranked,
                        "selected_chunk_id": ranked[0] if ranked else None,
                        "token_f1": 0.0,
                    }
                )
        return rows

    def stage_init(self) -> None:
        write_json(self.output_dir / "contract.json", experiment_contract_v4())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        if self.config.smoke:
            records = synthetic_records(n_docs=8)
            self.unlocked = tuple(item for item in split_unlocked(records) if item.positives)
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

    def stage_freeze_b0(self) -> None:
        if self.config.smoke:
            self.b0_rows = self._synthetic_b0()
        else:
            self.b0_rows = load_b0_oof(self.v3)
        self.b0_index = index_b0_oof(self.b0_rows)
        dest = self.output_dir / "reference" / "b0_live"
        dest.mkdir(parents=True, exist_ok=True)
        write_jsonl(dest / "predictions_scored.jsonl", self.b0_rows)
        write_json(
            dest / "manifest.json",
            {
                "name": "B0-live",
                "source": "artifacts/qa_chunk_selection_v3/analysis/oof/b0",
                "n": len(self.b0_rows),
                "seeds": sorted({int(row.get("seed") or 0) for row in self.b0_rows}),
                "note": "historical V3-B0 max5; not retrained",
            },
        )
        by_qid = self._by_qid()
        scored = []
        for seed in self.config.model_seeds:
            for row in self._b0_seed(seed).values():
                item = by_qid.get(row["question_id"])
                if item is None:
                    continue
                qa = qa_fields_from_row(row)
                if not qa.get("qa_chunk_id"):
                    qa["qa_chunk_id"] = row.get("selected_chunk_id")
                if "qa_span_score" not in qa:
                    qa["qa_span_score"] = row.get("final_ranking_score") or 0.0
                scored.append(
                    chunk_prediction(
                        item,
                        ranking_method="B0-live",
                        scores=scores_from_row(row),
                        qa=qa,
                        flags=self.flags,
                        skip_bertscore=self.config.skip_bertscore,
                    )
                    | {"seed": int(seed)}
                )
        self.predictions["B0-live"] = scored
        self.summaries["B0-live"] = summarize_rows(scored)
        write_json(dest / "metrics.json", self.summaries["B0-live"])
        self.progress.log("B0-live frozen", n=len(self.b0_rows), token_f1=self.summaries["B0-live"].get("token_f1"))

    def stage_shortlist(self) -> None:
        rows = []
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                payload = {
                    "question_id": qid,
                    "document_id": row.get("document_id"),
                    "seed": int(seed),
                    "cv_run": row.get("cv_run"),
                    "top5": top_k_ids(row, 5),
                    "top10": top_k_ids(row, 10),
                    "ranked_ids": top_k_ids(row, 10),
                }
                rows.append(payload)
        write_jsonl(self.output_dir / "cache" / "shortlists.jsonl", rows)
        self.progress.log("shortlists written", n=len(rows), k=list(self.config.top_k_values))

    def stage_clusters(self) -> None:
        rows = []
        for index, item in enumerate(self.unlocked, start=1):
            rows.extend(document_source_clusters(item, min_overlap=self.config.source_overlap))
            self.progress.count(index, len(self.unlocked), item="source_clusters")
        write_jsonl(self.output_dir / "cache" / "source_overlap_clusters.jsonl", rows)
        self.progress.log("source-overlap clusters", n=len(rows), threshold=self.config.source_overlap)

    def stage_keybert(self) -> None:
        extractor = self._extractor()
        write_json(self.output_dir / "cache" / "keybert_spec.json", extractor.spec())
        total = sum(len(item.chunks) for item in self.unlocked)
        current = 0
        phrases_by_q: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
        for item in self.unlocked:
            for chunk in item.chunks:
                current += 1
                phrases = extractor.extract(chunk.text)
                self.phrases[(item.question_id, int(chunk.chunk_id))] = phrases
                phrases_by_q[item.question_id][int(chunk.chunk_id)] = phrases
                self.progress.count(current, max(total, 1), item="keybert_chunk")
        vocab = sorted({row["phrase"] for phrases in self.phrases.values() for row in phrases})
        for index, phrase in enumerate(vocab, start=1):
            self.embeddings[phrase] = extractor.embed_phrases([phrase])[0]
            if index == 1 or index == len(vocab) or index % max(1, len(vocab) // 20) == 0:
                self.progress.count(index, max(len(vocab), 1), item="keybert_embed")
        train_records = self.unlocked
        if self.splits.get("runs"):
            train_records = records_for_role(self.unlocked, self.splits, run=0, role="train")
        self.idf = document_idf(train_records, self.phrases)
        for item in self.unlocked:
            by_chunk = phrases_by_q[item.question_id]
            self.profiles[item.document_id] = document_profile_phrases(by_chunk)
            self.profile_vecs[item.document_id] = profile_vector(self.profiles[item.document_id], self.embeddings)
        write_json(self.output_dir / "cache" / "train_idf.json", self.idf)
        write_json(self.output_dir / "cache" / "document_profiles.json", self.profiles)
        self.progress.log("KeyBERT cache ready", n_chunks=len(self.phrases), n_phrases=len(self.embeddings))

    def stage_qa_topk(self) -> None:
        answer_fn = self._answer_fn()
        needed: set[tuple[str, int]] = set()
        by_qid = self._by_qid()
        for seed in self.config.model_seeds:
            for qid, row in self._b0_seed(seed).items():
                if qid not in by_qid:
                    continue
                for cid in top_k_ids(row, max(self.config.top_k_values)):
                    needed.add((qid, int(cid)))
        for index, (qid, cid) in enumerate(sorted(needed), start=1):
            item = by_qid[qid]
            self.qa_store[(qid, cid)] = answer_one(
                item,
                cid,
                answer_fn,
                cache=self.cache,
                model_id=self._model_id(),
                inference_hash="v4-topk",
            )
            self.progress.count(index, max(len(needed), 1), item="qa_topk")
        write_jsonl(self.output_dir / "cache" / "qa_topk.jsonl", [{**row, "question_id": qid, "chunk_id": cid} for (qid, cid), row in self.qa_store.items()])
        self.progress.log("frozen QA on union(top-K)", n=len(self.qa_store))

    def _feature_bundle(self, item):
        by_chunk = {int(chunk.chunk_id): self.phrases.get((item.question_id, int(chunk.chunk_id)), []) for chunk in item.chunks}
        aspect_vec = None
        if self.extractor is not None:
            try:
                aspect_vec = self.extractor.embed_phrases([item.question])[0]
            except Exception:
                aspect_vec = None
        return {
            "phrases_by_chunk": by_chunk,
            "embeddings": self.embeddings,
            "idf": self.idf,
            "min_overlap": self.config.source_overlap,
            "aspect_embedding": aspect_vec,
        }

    def _record_chunk_system(self, name: str, rows: list[dict[str, Any]]) -> None:
        self.predictions[name] = rows
        self.summaries[name] = summarize_rows(rows)
        dest = self.output_dir / "analysis" / name
        dest.mkdir(parents=True, exist_ok=True)
        write_jsonl(dest / "predictions_scored.jsonl", rows)
        write_json(dest / "metrics.json", self.summaries[name])
        self.progress.log(f"{name} done", n=len(rows), token_f1=self.summaries[name].get("token_f1"))

    def stage_v1(self) -> None:
        by_qid = self._by_qid()
        cal_items = list(self.unlocked)
        if self.splits.get("runs"):
            cal_items = list(records_for_role(self.unlocked, self.splits, run=0, role="calibration"))
        lambdas = {}
        b0_cal = self._b0_seed(self.config.seed)
        for variant in ("V1-A3", "V1-A4", "V1-A5"):
            lambdas[variant] = choose_lambda(
                cal_items,
                b0_cal,
                variant=variant,
                k=self.config.primary_k,
                feature_fn=self._feature_bundle,
            )
            self.progress.log(f"{variant} lambda from CAL", lam=lambdas[variant])
        write_json(self.output_dir / "analysis" / "v1_lambdas.json", lambdas)
        for variant in V1_VARIANTS:
            rows = []
            for seed in self.config.model_seeds:
                for qid, row in self._b0_seed(seed).items():
                    item = by_qid.get(qid)
                    if item is None:
                        continue
                    ranked, scores, _feats = rerank_variant(
                        item,
                        row,
                        variant=variant,
                        k=self.config.primary_k,
                        lam=float(lambdas.get(variant, 0.5)),
                        **self._feature_bundle(item),
                    )
                    top = ranked[0] if ranked else int(item.chunks[0].chunk_id)
                    qa = self.qa_store.get((qid, top)) or {
                        "qa_answer": row.get("qa_answer"),
                        "qa_chunk_id": top,
                    }
                    pred = chunk_prediction(
                        item,
                        ranking_method=variant,
                        scores=scores,
                        qa=qa,
                        flags=self.flags,
                        skip_bertscore=self.config.skip_bertscore,
                    )
                    pred["seed"] = int(seed)
                    rows.append(pred)
            self._record_chunk_system(variant, rows)

    def stage_v2(self) -> None:
        by_qid = self._by_qid()
        for variant in V2_VARIANTS:
            rows = []
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
                        item,
                        row,
                        qa_by_chunk,
                        variant=variant,
                        k=self.config.primary_k,
                        lam=1.0,
                        min_overlap=self.config.source_overlap,
                    )
                    pred = aggregation_prediction(
                        item,
                        ranking_method=variant,
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
                    rows.append(pred)
            self._record_chunk_system(variant, rows)

    def stage_v3(self) -> None:
        answer_fn = self._answer_fn()
        train_records = list(self.unlocked)
        cal_records = list(self.unlocked)
        if self.splits.get("runs"):
            train_records = list(records_for_role(self.unlocked, self.splits, run=0, role="train"))
            cal_records = list(records_for_role(self.unlocked, self.splits, run=0, role="calibration"))
        mined = mine_utility_positives(
            train_records,
            answer_fn,
            cache=self.cache,
            model_id=self._model_id(),
            inference_hash="v4-utility",
            f1_threshold=self.config.utility_f1,
            progress=self.progress,
        )
        write_jsonl(self.output_dir / "analysis" / "v3_utility_candidates.jsonl", mined)
        write_json(
            self.output_dir / "analysis" / "v3_utility_summary.json",
            {
                "n": len(mined),
                "n_new_region": sum(1 for row in mined if not row["overlaps_original_positive"]),
                "threshold": self.config.utility_f1,
                "note": "utility_positive_candidate, not target_evidence",
            },
        )
        utility = utility_index(mined)
        seed = int(self.config.trainable_seeds[0])
        out = self.output_dir / "models" / "v3_utility" / f"seed{seed}"
        trained = train_utility_listwise(
            train_records,
            cal_records,
            self.flags_rows,
            utility,
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
            eta=self.config.utility_eta,
            smoke=self.config.smoke,
            progress=self.progress,
        )
        scorer = score_fn_for(trained["scorer"])
        rows = []
        from experiments.qa_chunk_selection.v2.schemas import to_runtime

        for item in self.unlocked:
            runtime = to_runtime(item)
            scores = {int(chunk.chunk_id): float(score) for chunk, score in zip(item.chunks, scorer(runtime))}
            ranked = rank_by_scores(list(scores), [scores[cid] for cid in scores])
            top = ranked[0]
            qa = self.qa_store.get((item.question_id, top)) or answer_one(
                item, top, self._answer_fn(), cache=self.cache, model_id=self._model_id(), inference_hash="v4-v3"
            )
            pred = chunk_prediction(item, ranking_method="V3-A3", scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)
            pred["seed"] = seed
            rows.append(pred)
        self._record_chunk_system("V3-A3", rows)

    def stage_v4(self) -> None:
        records = list(self.unlocked)
        if self.splits.get("runs"):
            records = list(records_for_role(self.unlocked, self.splits, run=0, role="train"))
        pairs = pair_documents(records, self.profile_vecs, seed=self.config.seed)
        answer_fn = self._answer_fn()
        target_s: list[float] = []
        mismatch_s: list[float] = []
        rows = []
        for index, (target, donor, sim) in enumerate(pairs, start=1):
            own = target.positives[0]
            other = donor.positives[0]
            phrases = self.profiles.get(target.document_id) or []
            own_probe = probe_pair(
                target, target, own.chunk_id, answer_fn,
                cache=self.cache, model_id=self._model_id(), inference_hash="v4-own", phrases=phrases,
            )
            syn_probe = probe_pair(
                target, donor, other.chunk_id, answer_fn,
                cache=self.cache, model_id=self._model_id(), inference_hash="v4-syn", phrases=phrases,
            )
            target_s.append(own_probe["anchored_score"])
            mismatch_s.append(syn_probe["anchored_score"])
            rows.append(
                {
                    "target_document_id": target.document_id,
                    "donor_document_id": donor.document_id,
                    "aspect": target.aspect,
                    "profile_similarity": sim,
                    "own": own_probe,
                    "synthetic": syn_probe,
                    "anchor_phrases": phrases,
                    "question": target.question,
                    "question_id": target.question_id,
                    "chunk_id": int(other.chunk_id),
                    "chunk_text": other.text,
                }
            )
            self.progress.count(index, max(len(pairs), 1), item="v4_pairs")
        summary = summarize_diagnostic(target_s, mismatch_s)
        summary["gate_pass"] = float(summary["pairwise_target_gt_mismatch"]) >= float(self.config.ownership_gate_pairwise)
        write_jsonl(self.output_dir / "analysis" / "v4_pairs.jsonl", rows)
        write_json(self.output_dir / "analysis" / "v4_diagnostic.json", summary)
        self.summaries["V4-diagnostic"] = summary
        self.progress.log("V4 diagnostic", **{k: summary[k] for k in ("auroc", "pairwise_target_gt_mismatch", "gate_pass")})

    def stage_v5(self) -> None:
        train_records = list(self.unlocked)
        if self.splits.get("runs"):
            train_records = list(records_for_role(self.unlocked, self.splits, run=0, role="train"))
        b0_train = self._b0_seed(self.config.seed)
        matrix, labels = collect_training(
            train_records, b0_train, self.qa_store, self.flags, k=self.config.primary_k, easy_cap=self.config.easy_negative_cap
        )
        head = LinearHead().fit(matrix, labels, FEATURE_NAMES)
        write_json(self.output_dir / "models" / "v5_linear.json", head.to_dict())
        by_qid = self._by_qid()
        for variant in ("V5-A0", "V5-A1", "V5-A2", "V5-A3"):
            rows = []
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
                    _ranked, scores = apply_v5(item, row, qa_by_chunk, variant=variant, k=self.config.primary_k, head=head)
                    if variant == "V5-A1":
                        top = int(top_k_ids(row, 1)[0])
                    else:
                        top = max(scores, key=lambda cid: (scores[cid], -cid))
                    qa = self.qa_store.get((qid, top)) or {"qa_answer": "", "qa_chunk_id": top}
                    pred = chunk_prediction(item, ranking_method=variant, scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)
                    pred["seed"] = int(seed)
                    rows.append(pred)
            self._record_chunk_system(variant, rows)

    def stage_v6(self) -> None:
        cal_items = list(self.unlocked)
        if self.splits.get("runs"):
            cal_items = list(records_for_role(self.unlocked, self.splits, run=0, role="calibration"))
        alpha, beta, gamma = choose_v6_weights(
            cal_items, self._b0_seed(self.config.seed), self.qa_store, k=self.config.primary_k, min_overlap=self.config.source_overlap
        )
        write_json(self.output_dir / "analysis" / "v6_weights.json", {"alpha": alpha, "beta": beta, "gamma": gamma})
        by_qid = self._by_qid()
        rows = []
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
                _ranked, scores = apply_v6(
                    item, row, qa_by_chunk, alpha=alpha, beta=beta, gamma=gamma, k=self.config.primary_k, min_overlap=self.config.source_overlap
                )
                top = max(scores, key=lambda cid: (scores[cid], -cid))
                qa = self.qa_store.get((qid, int(top))) or {"qa_answer": "", "qa_chunk_id": top}
                pred = chunk_prediction(item, ranking_method="V6", scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)
                pred["seed"] = int(seed)
                rows.append(pred)
        self._record_chunk_system("V6", rows)

    def stage_v7(self) -> None:
        diag = {}
        path = self.output_dir / "analysis" / "v4_diagnostic.json"
        if path.exists():
            diag = load_json(path)
        use_anchor = bool(diag.get("gate_pass"))
        if not use_anchor:
            self.progress.warn("V7 skips anchored synthetics because V4 gate failed or was not run")
        train_records = list(self.unlocked)
        cal_records = list(self.unlocked)
        if self.splits.get("runs"):
            train_records = list(records_for_role(self.unlocked, self.splits, run=0, role="train"))
            cal_records = list(records_for_role(self.unlocked, self.splits, run=0, role="calibration"))
        synthetics = []
        pairs_path = self.output_dir / "analysis" / "v4_pairs.jsonl"
        if use_anchor and pairs_path.exists():
            synthetics = read_jsonl(pairs_path)
        seed = int(self.config.trainable_seeds[0])
        trained = train_v7(
            train_records,
            cal_records,
            self.flags_rows,
            self._b0_seed(seed),
            synthetics,
            seed=seed,
            device=self._resolved_device(),
            model_name=self.config.cross_encoder_model,
            output_dir=self.output_dir / "models" / "v7" / f"seed{seed}",
            epochs=self.config.b0_epochs,
            patience=self.config.b0_patience,
            batch_size=self.config.b0_batch_size,
            learning_rate=self.config.b0_learning_rate,
            max_length=self.config.b0_max_length,
            k=self.config.primary_k,
            easy_cap=self.config.easy_negative_cap,
            use_anchor=use_anchor,
            profiles=self.profiles,
            smoke=self.config.smoke,
            progress=self.progress,
        )
        rows = []
        for item in self.unlocked:
            scores = score_v7(trained["scorer"], item, use_anchor=use_anchor, phrases=self.profiles.get(item.document_id) or [])
            top = max(scores, key=lambda cid: (scores[cid], -cid))
            qa = self.qa_store.get((item.question_id, int(top))) or answer_one(
                item, int(top), self._answer_fn(), cache=self.cache, model_id=self._model_id(), inference_hash="v4-v7"
            )
            pred = chunk_prediction(item, ranking_method="V7", scores=scores, qa=qa, flags=self.flags, skip_bertscore=True)
            pred["seed"] = seed
            rows.append(pred)
        self._record_chunk_system("V7", rows)

    def stage_compare(self) -> None:
        if "B0-live" not in self.predictions and (self.output_dir / "reference" / "b0_live" / "predictions_scored.jsonl").exists():
            self.predictions["B0-live"] = read_jsonl(self.output_dir / "reference" / "b0_live" / "predictions_scored.jsonl")
            self.summaries["B0-live"] = summarize_rows(self.predictions["B0-live"])
        baseline = self.predictions.get("B0-live") or []
        table = []
        deltas = {}
        trans = {}
        b0_hn = float((self.summaries.get("B0-live") or {}).get("hn@1") or 0.0)
        b0_pair = float((self.summaries.get("B0-live") or {}).get("pairwise_clean_hn") or 0.0)
        admissible = []
        for name, rows in self.predictions.items():
            summary = self.summaries.get(name) or summarize_rows(rows)
            self.summaries[name] = summary
            row = {"system": name, **{k: summary.get(k) for k in ("n", "token_f1", "exact_match", "hit@1", "mrr", "hn@1", "pairwise_clean_hn", "ambiguous_at_1_original", "gsrn_at_1_original")}}
            if name != "B0-live" and baseline:
                deltas[name] = paired_delta(baseline, rows, metric="token_f1", n_boot=min(self.config.n_bootstrap, 200 if self.config.smoke else self.config.n_bootstrap), seed=self.config.seed)
                trans[name] = transitions(baseline, rows)
                hn_ok = True
                pair_ok = True
                if "hn@1" in summary and summary["hn@1"] == summary["hn@1"]:
                    hn_ok = float(summary["hn@1"]) <= b0_hn + self.config.epsilon_hn
                if "pairwise_clean_hn" in summary and summary["pairwise_clean_hn"] == summary["pairwise_clean_hn"]:
                    pair_ok = float(summary["pairwise_clean_hn"]) >= b0_pair - self.config.epsilon_pairwise
                row["hn_safeguard"] = hn_ok
                row["pairwise_safeguard"] = pair_ok
                if name.startswith("V2"):
                    row["admissible"] = True
                else:
                    row["admissible"] = bool(hn_ok and pair_ok)
                if row["admissible"]:
                    admissible.append(name)
            else:
                row["admissible"] = True
            table.append(row)
        selected = "B0-live"
        best = float((self.summaries.get("B0-live") or {}).get("token_f1") or 0.0)
        for name in admissible:
            f1 = float((self.summaries.get(name) or {}).get("token_f1") or 0.0)
            if f1 > best:
                best = f1
                selected = name
        payload = {
            "table": table,
            "deltas": deltas,
            "transitions": trans,
            "selected": selected,
            "admissible": admissible,
            "b0_token_f1": (self.summaries.get("B0-live") or {}).get("token_f1"),
            "systems": list(self.predictions),
            "internal_test_locked": True,
        }
        write_json(self.output_dir / "analysis" / "compare.json", payload)
        write_json(self.output_dir / "analysis" / "selection.json", {"selected": selected, "admissible": admissible, "b0_live_kept": True})
        self._compare_payload = payload
        self.progress.log("compare done", selected=selected, n_systems=len(self.predictions))

    def stage_report(self) -> None:
        payload = getattr(self, "_compare_payload", None)
        if payload is None:
            path = self.output_dir / "analysis" / "compare.json"
            payload = load_json(path) if path.exists() else {"systems": list(self.predictions), "internal_test_locked": True}
        write_report(self.output_dir, payload)
        self.progress.log("report written", path=str(self.output_dir / "reports" / "results_v4.md"))
