"""Orchestrate A–D on frozen SciREX candidate sets with progress logs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Sequence

from .answerability import (
    TfidfAnswerabilityScorer,
    collect_pairs,
    evaluate_answerability_scores,
    train_answerability,
)
from .candidates import apply_candidate_set, document_split_payload, freeze_candidate_sets
from .config import ExperimentConfig
from .contract import experiment_contract
from .dataset import filter_answerable, load_split_records, select_documents
from .evaluate import evaluate_ranking, write_run
from .io_util import assert_isolated_output, load_json, write_json, write_jsonl
from .metrics import paired_delta
from .models import ModelBundle, load_models
from .progress import ProgressLogger
from .ranking import (
    bm25_chunk_scores,
    cross_encoder_scores,
    dense_chunk_scores,
    rank_ids,
    rrf_fuse,
    section_scores,
    score_map,
)
from .sanity import audit_records
from .schema import ChunkRecord, InstanceRecord
from .structure import aggregate_answers, expand_neighbors, sentence_evidence, structural_mix


def _score_dense(bundle: ModelBundle) -> Callable[[str, Sequence[ChunkRecord]], list[float]]:
    def score_fn(question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
        return dense_chunk_scores(question, chunks, bundle.encode)

    return score_fn


def _score_ce(reranker) -> Callable[[str, Sequence[ChunkRecord]], list[float]]:
    def score_fn(question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
        return cross_encoder_scores(question, chunks, reranker)

    return score_fn


def _score_rrf(bundle: ModelBundle, rrf_k: int) -> Callable[[str, Sequence[ChunkRecord]], list[float]]:
    def score_fn(question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
        dense = dense_chunk_scores(question, chunks, bundle.encode)
        bm25 = bm25_chunk_scores(question, chunks)
        dense_rank = rank_ids(chunks, dense)
        bm25_rank = rank_ids(chunks, bm25)
        fused = rrf_fuse([dense_rank, bm25_rank], k=rrf_k)
        return [float(fused.get(chunk.chunk_id, 0.0)) for chunk in chunks]

    return score_fn


class Pipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = assert_isolated_output(config.output_dir)
        self.progress = ProgressLogger(self.output_dir / "progress.jsonl")
        self.records: dict[str, tuple[InstanceRecord, ...]] = {}
        self.bundle: ModelBundle | None = None
        self.answerability = None

    def run(self) -> int:
        self.progress.stage("start", f"output={self.output_dir}")
        write_json(self.output_dir / "contract.json", experiment_contract())
        write_json(self.output_dir / "config_resolved.json", self.config.to_dict())
        stages = set(self.config.stages)
        if "prepare" in stages or not (self.output_dir / "data" / "dataset.jsonl").exists():
            self.prepare()
        else:
            self._load_prepared()
        need_models = stages & {
            "a0",
            "a",
            "b-train",
            "b",
            "c",
            "d1",
            "d2",
            "d3",
            "d4",
            "d5",
        }
        if need_models:
            self.progress.stage("models", "loading encoders / QA / CE")
            self.bundle = load_models(
                device=self.config.device,
                cross_encoder_model=self.config.cross_encoder_model,
                smoke=self.config.smoke,
            )
            self.progress.log(
                "models ready",
                device=self.bundle.device,
                qa=self.bundle.qa_model_name,
                ce=self.bundle.ce_model_name,
                smoke=self.bundle.smoke,
            )
        if "b-train" in stages:
            self.train_b()
        elif "b" in stages or "c" in stages or "d2" in stages:
            self._load_b_if_available()
        for name in self.config.stages:
            if name in {"prepare", "b-train", "compare"}:
                continue
            self.run_experiment(name)
        if "compare" in self.config.stages:
            self.compare()
        self.progress.stage("done", "all requested stages finished")
        return 0

    def prepare(self) -> None:
        cfg = self.config
        self.progress.stage("prepare", "loading official SciREX Q5 labels")
        if cfg.allow_test:
            raise ValueError("TEST is reserved until the feature rule is frozen")
        train = load_split_records(
            cfg.train_label_dir,
            cfg.train_preprocess_dir,
            split_name="train",
            progress=self.progress,
        )
        dev = load_split_records(
            cfg.dev_label_dir,
            cfg.dev_preprocess_dir,
            split_name="dev",
            progress=self.progress,
        )
        if cfg.smoke:
            limit = cfg.limit_docs or min(2, cfg.debug_docs or 2)
            train = select_documents(train, limit_docs=limit, seed=cfg.seed)
            dev = select_documents(dev, limit_docs=max(1, limit // 2 or 1), seed=cfg.seed)
            self.progress.log(f"smoke subset train_docs={len({r.document_id for r in train})} dev_docs={len({r.document_id for r in dev})}")
        elif cfg.limit_docs or cfg.debug_docs:
            limit = cfg.limit_docs or cfg.debug_docs
            train = select_documents(train, limit_docs=limit, seed=cfg.seed)
            dev = select_documents(dev, limit_docs=max(1, limit // 3 or 1), seed=cfg.seed)
        train = filter_answerable(train, include_unanswerable=cfg.include_unanswerable)
        dev = filter_answerable(dev, include_unanswerable=cfg.include_unanswerable)
        records = tuple(list(train) + list(dev))
        self.progress.stage("prepare", "sanity audit (labels are not rewritten)")
        audit = audit_records(records)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        write_json(data_dir / "hard_negative_verification.json", audit)
        if audit["n_blocking_errors"]:
            raise ValueError(f"blocking label errors: {audit['blocking_errors'][:3]}")
        splits = document_split_payload(records)
        write_json(data_dir / "splits.json", splits)
        candidate_rows = freeze_candidate_sets(
            records,
            mode=cfg.candidate_mode,
            n_easy=cfg.n_easy_negatives,
            seed=cfg.seed,
        )
        write_jsonl(data_dir / "candidate_sets.jsonl", candidate_rows)
        restricted: list[InstanceRecord] = []
        by_key = {(row["question_id"], row["split"]): row for row in candidate_rows}
        for item in records:
            row = by_key[(item.question_id, item.split)]
            restricted.append(apply_candidate_set(item, list(row["chunk_ids"])))
        write_jsonl(data_dir / "dataset.jsonl", [item.to_dict() for item in restricted])
        self.records = {
            "train": tuple(item for item in restricted if item.split == "train"),
            "dev": tuple(item for item in restricted if item.split == "dev"),
        }
        self.progress.log(
            "prepare done",
            train=len(self.records["train"]),
            dev=len(self.records["dev"]),
            hn_gold_warnings=audit["n_hn_literal_gold_warnings"],
        )

    def _load_prepared(self) -> None:
        from .schema import ChunkRecord, GoldSpan, InstanceRecord

        rows = []
        path = self.output_dir / "data" / "dataset.jsonl"
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            import json

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
        self.records = {
            "train": tuple(item for item in rows if item.split == "train"),
            "dev": tuple(item for item in rows if item.split == "dev"),
        }
        self.progress.log("loaded prepared dataset", train=len(self.records["train"]), dev=len(self.records["dev"]))

    def train_b(self) -> None:
        cfg = self.config
        self.progress.stage("b-train", "fitting answerability on train labels only")
        model_dir = self.output_dir / "models" / "answerability_ce"
        if cfg.resume and (model_dir / "wrapper.json").exists():
            self._load_b_if_available()
            self.progress.log("reused trained answerability model")
            return
        best: dict[str, Any] | None = None
        for ratio in cfg.answerability_ratios:
            self.progress.log(f"trying pos:hn:easy = {ratio[0]}:{ratio[1]}:{ratio[2]}")
            pairs = collect_pairs(self.records["train"], ratio=ratio, seed=cfg.seed)
            if cfg.smoke:
                scorer = TfidfAnswerabilityScorer().fit(pairs)
            else:
                run_dir = model_dir / f"ratio_{ratio[0]}_{ratio[1]}_{ratio[2]}"
                scorer = train_answerability(
                    pairs,
                    model_name=cfg.cross_encoder_model,
                    device=self.bundle.device if self.bundle else "cpu",
                    output_dir=run_dir,
                    epochs=cfg.b_epochs,
                    batch_size=cfg.b_batch_size,
                    learning_rate=cfg.b_learning_rate,
                    max_length=cfg.b_max_length,
                    loss_name=cfg.b_loss,
                    margin=cfg.pairwise_margin,
                    progress=self.progress,
                )
            metrics = evaluate_answerability_scores(self.records["dev"], scorer)
            self.progress.log(
                f"ratio {ratio} dev PR-AUC={metrics['pr_auc']:.4f} pairwise={metrics['pairwise_pos_gt_hn']:.4f}"
            )
            score = (
                float(metrics["pr_auc"]) if metrics["pr_auc"] == metrics["pr_auc"] else -1.0,
                float(metrics["pairwise_pos_gt_hn"])
                if metrics["pairwise_pos_gt_hn"] == metrics["pairwise_pos_gt_hn"]
                else -1.0,
            )
            if best is None or score > best["score"]:
                best = {"ratio": ratio, "metrics": metrics, "scorer": scorer, "score": score}
        assert best is not None
        self.answerability = best["scorer"]
        if hasattr(self.answerability, "save"):
            self.answerability.save(model_dir)
        write_json(
            self.output_dir / "models" / "answerability_selection.json",
            {"ratio": list(best["ratio"]), "dev_metrics": best["metrics"]},
        )
        self.progress.log("selected answerability ratio", ratio=best["ratio"])

    def _load_b_if_available(self) -> None:
        model_dir = self.output_dir / "models" / "answerability_ce"
        wrapper = model_dir / "wrapper.json"
        if not wrapper.exists():
            return
        kind = load_json(wrapper)
        if kind.get("kind") == "tfidf_logistic":
            self.answerability = TfidfAnswerabilityScorer.load(model_dir)
        else:
            from .answerability import AnswerabilityModel

            device = self.bundle.device if self.bundle else "cpu"
            self.answerability = AnswerabilityModel.load(model_dir, device=device)

    def run_experiment(self, name: str) -> None:
        if self.bundle is None:
            raise RuntimeError("models are not loaded")
        self.progress.stage(name, f"evaluating on frozen candidates ({self.config.candidate_mode})")
        score_fn = self._scorer_for(name)
        select_k = 1
        qa_mode = "top1"
        if name == "c":
            select_k = self.config.default_passage_m
            qa_mode = "global"
        for split in ("train", "dev"):
            records = self.records[split]
            if name == "c":
                payload = evaluate_ranking(
                    records,
                    score_fn,
                    experiment=name,
                    split=split,
                    ks=self.config.top_k_values,
                    progress=self.progress,
                    answer_fn=self.bundle.answer,
                    skip_qa=self.config.skip_qa,
                    skip_bertscore=self.config.skip_bertscore,
                    select_k=select_k,
                    qa_mode=qa_mode,
                    qa_answerer=self.bundle.qa_answerer,
                )
                if split == "dev" and not self.config.skip_qa:
                    self._c_diagnostics(records, score_fn)
            elif name == "d5":
                payload = self._eval_d5(records, score_fn, split)
            elif name == "d4":
                payload = self._eval_d4(records, score_fn, split)
            elif name == "d3":
                payload = self._eval_d3(records, score_fn, split)
            else:
                payload = evaluate_ranking(
                    records,
                    score_fn,
                    experiment=name,
                    split=split,
                    ks=self.config.top_k_values,
                    progress=self.progress,
                    answer_fn=self.bundle.answer,
                    skip_qa=self.config.skip_qa,
                    skip_bertscore=self.config.skip_bertscore,
                    select_k=select_k,
                    qa_mode=qa_mode,
                    qa_answerer=self.bundle.qa_answerer,
                )
            write_run(
                self.config.run_dir(name, self.config.seed) / split,
                experiment=name,
                split=split,
                payload=payload,
                config=self.config.to_dict(),
            )
            self.progress.log(
                f"{name}/{split} hit@1={payload['summary'].get('hit@1')} "
                f"mrr={payload['summary'].get('mrr')} "
                f"token_f1={payload['summary'].get('token_f1')}"
            )

    def _scorer_for(self, name: str):
        bundle = self.bundle
        assert bundle is not None
        if name == "a0":
            return _score_dense(bundle)
        if name == "a":
            return _score_ce(bundle.relevance_reranker)
        if name in {"b", "c", "d2", "d3", "d4", "d5"}:
            if self.answerability is None and name != "d1":
                self.progress.warn(f"{name} needs B; falling back to generic CE")
                return _score_ce(bundle.relevance_reranker)
            if name == "b" or name == "c":
                return _score_ce(self.answerability)
            if name == "d2":
                return self._score_d2()
            if name in {"d3", "d4", "d5"}:
                return self._score_d2()
        if name == "d1":
            return _score_rrf(bundle, self.config.rrf_k)
        raise ValueError(f"no scorer for {name}")

    def _score_d2(self):
        bundle = self.bundle
        assert bundle is not None
        answerability = self.answerability or bundle.relevance_reranker

        def score_fn(question: str, chunks: Sequence[ChunkRecord]) -> list[float]:
            retrieval = _score_rrf(bundle, self.config.rrf_k)(question, chunks)
            ans = cross_encoder_scores(question, chunks, answerability)
            section = section_scores(question, chunks, None)
            return structural_mix(retrieval, ans, section)

        return score_fn

    def _eval_d3(self, records, score_fn, split: str) -> dict[str, Any]:
        assert self.bundle is not None

        def expanded(instance: InstanceRecord) -> list[float]:
            base = score_fn(instance.question, instance.chunks)
            ranked = rank_ids(instance.chunks, base)
            seeds = ranked[: self.config.n_seed_chunks]
            keep = set(
                expand_neighbors(
                    instance,
                    seeds,
                    encode=self.bundle.encode,
                    max_chunks=self.config.max_evidence_chunks,
                    max_neighbors=self.config.max_graph_neighbors,
                    hops=self.config.graph_hops,
                )
            )
            return [
                float(score if chunk.chunk_id in keep else score - 1e6)
                for chunk, score in zip(instance.chunks, base)
            ]

        return evaluate_ranking(
            records,
            None,
            experiment="d3",
            split=split,
            ks=self.config.top_k_values,
            progress=self.progress,
            answer_fn=self.bundle.answer,
            skip_qa=self.config.skip_qa,
            skip_bertscore=self.config.skip_bertscore,
            select_k=self.config.max_evidence_chunks,
            qa_mode="global",
            qa_answerer=self.bundle.qa_answerer,
            instance_score_fn=expanded,
        )

    def _eval_d4(self, records, score_fn, split: str) -> dict[str, Any]:
        assert self.bundle is not None
        rows = []
        score_rows = []
        from .metrics import ranking_row, qa_row, summarize_ranking
        import time

        started = time.time()
        for index, instance in enumerate(records, start=1):
            scores = score_fn(instance.question, instance.chunks)
            ranked = rank_ids(instance.chunks, scores)
            mapped = score_map(instance.chunks, scores)
            top = [instance.chunk_by_id(cid) for cid in ranked[: self.config.max_evidence_chunks]]
            compact, mapping = sentence_evidence(
                instance.question,
                top,
                _score_ce(self.answerability or self.bundle.relevance_reranker),
                window=self.config.sentence_window,
            )
            result = self.bundle.answer(instance.question, compact) if compact else {"answer": ""}
            row = ranking_row(instance, ranked, mapped, ks=self.config.top_k_values)
            row["experiment"] = "d4"
            row["latency_s"] = 0.0
            row["qa_answer"] = str(result.get("answer") or "")
            row["evidence_mapping"] = mapping
            row.update(qa_row(row["qa_answer"], instance.gold_answers, skip_bertscore=self.config.skip_bertscore))
            rows.append(row)
            self.progress.count(index, len(records), item=f"d4/{split}")
        summary = summarize_ranking(rows, ks=self.config.top_k_values)
        summary.update({"experiment": "d4", "split": split, "elapsed_s": time.time() - started})
        return {"rows": rows, "chunk_scores": score_rows, "summary": summary}

    def _eval_d5(self, records, score_fn, split: str) -> dict[str, Any]:
        assert self.bundle is not None
        from .metrics import ranking_row, qa_row, summarize_ranking
        import time

        started = time.time()
        rows = []
        for index, instance in enumerate(records, start=1):
            scores = score_fn(instance.question, instance.chunks)
            ranked = rank_ids(instance.chunks, scores)
            mapped = score_map(instance.chunks, scores)
            spans = []
            for chunk_id in ranked[: self.config.max_evidence_chunks]:
                chunk = instance.chunk_by_id(chunk_id)
                result = self.bundle.answer(instance.question, chunk.text)
                spans.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": result.get("answer") or "",
                        "start_char": result.get("start") or 0,
                        "end_char": result.get("end") or 0,
                        "score": float(result.get("start_logit") or 0.0)
                        + float(result.get("end_logit") or 0.0),
                    }
                )
            agg = aggregate_answers(spans, instance.chunks)
            row = ranking_row(instance, ranked, mapped, ks=self.config.top_k_values)
            row["experiment"] = "d5"
            row["latency_s"] = 0.0
            row["qa_answer"] = str(agg["text"])
            row.update(qa_row(row["qa_answer"], instance.gold_answers, skip_bertscore=self.config.skip_bertscore))
            rows.append(row)
            self.progress.count(index, len(records), item=f"d5/{split}")
        summary = summarize_ranking(rows, ks=self.config.top_k_values)
        summary.update({"experiment": "d5", "split": split, "elapsed_s": time.time() - started})
        return {"rows": rows, "chunk_scores": [], "summary": summary}

    def _c_diagnostics(self, records, score_fn) -> None:
        rng = random.Random(self.config.seed)
        m = self.config.default_passage_m
        self.progress.log(f"C diagnostics: random-{m} and oracle-{m} on dev")

        def random_fn(instance: InstanceRecord) -> list[float]:
            order = list(range(len(instance.chunks)))
            rng.shuffle(order)
            scores = [0.0] * len(instance.chunks)
            for rank, index in enumerate(order):
                scores[index] = float(len(instance.chunks) - rank)
            return scores

        def oracle_fn(instance: InstanceRecord) -> list[float]:
            pos = {chunk.chunk_id for chunk in instance.positives}
            base = score_fn(instance.question, instance.chunks)
            return [
                float(score) + (1000.0 if chunk.chunk_id in pos else 0.0)
                for chunk, score in zip(instance.chunks, base)
            ]

        for tag, fn in (("c_random", random_fn), ("c_oracle", oracle_fn)):
            payload = evaluate_ranking(
                records,
                None,
                experiment=tag,
                split="dev",
                ks=self.config.top_k_values,
                progress=self.progress,
                answer_fn=self.bundle.answer if self.bundle else None,
                skip_qa=self.config.skip_qa,
                skip_bertscore=self.config.skip_bertscore,
                select_k=m,
                qa_mode="global",
                qa_answerer=self.bundle.qa_answerer if self.bundle else None,
                instance_score_fn=fn,
            )
            write_run(
                self.config.run_dir(tag, self.config.seed) / "dev",
                experiment=tag,
                split="dev",
                payload=payload,
                config=self.config.to_dict(),
            )

    def compare(self) -> None:
        self.progress.stage("compare", "paired clustered bootstrap on DEV")
        analysis = self.output_dir / "analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        names = [item for item in ("a0", "a", "b", "c", "d1", "d2", "d3", "d4", "d5") if item in self.config.stages or item in {"a0", "a", "b", "c"}]
        loaded: dict[str, list[dict[str, Any]]] = {}
        for name in names:
            path = self.config.run_dir(name, self.config.seed) / "dev" / "predictions.jsonl"
            if not path.exists():
                continue
            from .io_util import read_jsonl

            loaded[name] = read_jsonl(path)
        table = []
        for name, rows in loaded.items():
            from .metrics import summarize_ranking

            summary = summarize_ranking(rows, ks=self.config.top_k_values)
            summary["model"] = name
            table.append(summary)
        write_json(analysis / "dev_summary.json", {"models": table})
        comparisons = {}
        if "a" in loaded and "b" in loaded:
            comparisons["B-A_token_f1"] = paired_delta(loaded["a"], loaded["b"], "token_f1", seed=self.config.seed)
            comparisons["B-A_hit@1"] = paired_delta(loaded["a"], loaded["b"], "hit@1", seed=self.config.seed)
        if "b" in loaded and "c" in loaded:
            comparisons["C-B_token_f1"] = paired_delta(loaded["b"], loaded["c"], "token_f1", seed=self.config.seed)
        write_json(analysis / "paired_comparison.json", comparisons)
        self.progress.log("wrote analysis tables", n_models=len(table))
