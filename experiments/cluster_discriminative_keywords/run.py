"""Isolated hyp4 run: cluster-specific discriminative keywords.

Does not modify production UnTIE. Does not rewrite frozen artifact trees.
Does not re-run RoBERTa; ranking and QA use cached extractive spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster-specific discriminative QA keywords (hyp4)"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/keybert_topic_clustering/en_full/evidence",
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/keybert_topic_clustering/en_full",
    )
    parser.add_argument(
        "--static-model",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Which task was solved?")
    parser.add_argument("--field-id", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cluster_keyword_profiles.csv and chunk_recall_metrics.csv; skip H_m.",
    )
    parser.add_argument("--topic-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1e6)
    except Exception:
        return 0.0


def _gpu_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:
        pass
    return 0.0


def _texts_for(
    pairs: Sequence[tuple[str, int]],
    evidence: Mapping[str, Any],
) -> list[str]:
    out = []
    for doc_id, chunk_id in pairs:
        item = evidence[doc_id]
        chunk = next((row for row in item.chunks if int(row.index) == int(chunk_id)), None)
        out.append("" if chunk is None else chunk.text)
    return out


def _doc_text(evidence_item) -> str:
    return " ".join(chunk.text for chunk in evidence_item.chunks)


def _load_recall_summaries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"method", "split"}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            if "n" in parsed:
                parsed["n"] = int(parsed["n"])
            if "n_with_gold" in parsed:
                parsed["n_with_gold"] = int(parsed["n_with_gold"])
            rows.append(parsed)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from experiments.cluster_discriminative_keywords.candidates import (
        attention_pool,
        build_named_pools,
        cluster_of,
        importance_weight,
        load_assignments,
        load_document_keyphrases,
        load_routing,
        load_topic_terms,
        question_keybert_pool,
        topic_pool,
    )
    from experiments.cluster_discriminative_keywords.config import (
        FROZEN_CLUSTERING,
        ExperimentParams,
        POOL_SPECS,
    )
    from experiments.cluster_discriminative_keywords.contract import experiment_contract
    from experiments.cluster_discriminative_keywords.discrimination import (
        KeywordProfile,
        blend,
        load_profiles_by_pool,
        score_profiles,
        shrinkage_lambda,
    )
    from experiments.cluster_discriminative_keywords.io_util import (
        append_timing,
        assert_isolated_output,
        write_csv,
        write_json,
    )
    from experiments.cluster_discriminative_keywords.labels import (
        cluster_chunk_groups,
        flatten_hard_negative_table,
        hard_negative_indices,
        label_document_chunks,
    )
    from experiments.cluster_discriminative_keywords.qa_eval import (
        keyword_chunk_scores,
        predict_consensus,
        predict_hybrid,
        predict_linear,
        predict_prefilter_centroid,
        predict_q0,
    )
    from experiments.cluster_discriminative_keywords.ranking import (
        gold_chunk_index,
        mix_cluster_scores,
        profiles_to_keywords,
        rank_metrics,
        score_document_chunks,
        summarize_ranks,
    )
    from experiments.cluster_discriminative_keywords.report import write_summary
    from experiments.keybert_topic_clustering.documents import load_papers, split_groups
    from experiments.keybert_topic_clustering.downstream import bootstrap_mean_ci
    from experiments.keybert_topic_clustering.encoder import CachedSentenceEncoder
    from experiments.keybert_topic_clustering.routing import RoutingResult
    from untie.cli import load_static_keywords
    from untie.extraction_metrics import char_f1, normalize_pred
    from untie.keyword_evidence import EvidenceStore, ExtractionMetricCache
    from untie.models import resolve_device
    from untie.ranking import WeightedKeyword

    output_dir = assert_isolated_output(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_dir / "timings.csv"
    params = ExperimentParams()
    device = resolve_device(args.device)
    papers = load_papers(args.dataset, seed=args.seed, quick=args.quick)
    groups = split_groups(papers)
    train, dev, test = groups["train"], groups["dev"], groups["test"]
    by_id = {paper.doc_id: paper for paper in papers}

    write_json(
        output_dir / "experiment_config.json",
        {
            "dataset": str(args.dataset),
            "output_dir": str(output_dir),
            "evidence_dir": str(args.evidence_dir),
            "cluster_dir": str(args.cluster_dir),
            "question": args.question,
            "field_id": args.field_id,
            "frozen_clustering": FROZEN_CLUSTERING,
            "contract": experiment_contract(),
            "n_train": len(train),
            "n_dev": len(dev),
            "n_test": len(test),
            "quick": args.quick,
            "resume": bool(args.resume),
            "device": device,
            "anti_leakage": "dictionaries and H_m fit on train; thresholds on dev; test once",
        },
    )

    evidence_store = EvidenceStore(args.evidence_dir)
    evidence = {}
    missing = []
    for paper in papers:
        loaded = evidence_store.load(paper.doc_id)
        if loaded is None:
            missing.append(paper.doc_id)
        else:
            evidence[paper.doc_id] = loaded
    if missing:
        raise SystemExit(
            f"missing cached QA evidence for {len(missing)} documents; "
            "pass --evidence-dir to hyp2/hyp3. This experiment does not re-run RoBERTa."
        )

    cluster_dir = Path(args.cluster_dir)
    assignments = load_assignments(cluster_dir / "C_keybert_cluster_assignments.csv")
    routes = load_routing(cluster_dir / "C_keybert_routing.json")
    chosen_raw = json.loads((cluster_dir / "chosen_configs.json").read_text(encoding="utf-8"))
    topic_terms = load_topic_terms(chosen_raw)
    phrases = load_document_keyphrases(cluster_dir / "C_keybert_document_keyphrases.csv")
    static_keywords, static_strategy = load_static_keywords(args.static_model, field_id=args.field_id)

    def route_for(paper) -> RoutingResult:
        if paper.split == "train":
            cluster_id = cluster_of(paper.doc_id, "train", assignments, routes)
            return RoutingResult(
                paper.doc_id, paper.split, cluster_id, 1.0, (cluster_id,), (1.0,), (1.0,)
            )
        if paper.doc_id not in routes:
            raise SystemExit(f"missing frozen routing for {paper.doc_id}")
        return routes[paper.doc_id]

    assignment_rows = []
    for paper in papers:
        route = route_for(paper)
        assignment_rows.append(
            {
                "doc_id": paper.doc_id,
                "split": paper.split,
                "cluster_id": int(route.top1),
                "top_ids": ",".join(str(item) for item in route.top_ids),
                "weights": ",".join(f"{item:.6f}" for item in route.weights),
            }
        )
    write_csv(output_dir / "cluster_assignments.csv", assignment_rows)

    train_by_cluster: dict[int, list[str]] = defaultdict(list)
    for paper in train:
        train_by_cluster[int(route_for(paper).top1)].append(paper.doc_id)
    cluster_ids = sorted(train_by_cluster)

    profiles_path = output_dir / "cluster_keyword_profiles.csv"
    recall_path = output_dir / "chunk_recall_metrics.csv"
    resume = bool(args.resume)
    fitted: dict[tuple[str, str], dict[int, list[KeywordProfile]]] = {}
    pools_by_cluster: dict[int, dict[str, dict[str, KeywordProfile]]] = {}
    fit_h = None
    apply_shrinkage = None
    dev_rank_summaries: list[dict[str, Any]] = []
    dev_rank_rows: list[dict[str, Any]] = []

    if resume:
        if not profiles_path.is_file() or not recall_path.is_file():
            raise SystemExit(
                "--resume requires cluster_keyword_profiles.csv and "
                "chunk_recall_metrics.csv in --output-dir"
            )
        started = time.perf_counter()
        loaded_profiles = load_profiles_by_pool(profiles_path)
        fitted = {(pool, "H_hard"): clusters for pool, clusters in loaded_profiles.items()}
        dev_rank_summaries = _load_recall_summaries(recall_path)
        append_timing(
            timings_path,
            {
                "stage": "resume_dictionaries",
                "seconds": time.perf_counter() - started,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
            },
        )
    else:
            started = time.perf_counter()
            encoder = CachedSentenceEncoder(
                args.topic_encoder,
                device=device if str(device).startswith("cuda") else "cpu",
                cache_dir=output_dir / "cache" / "topic_embeddings",
            )
            question_vec = encoder.encode([args.question])[0]
            unique_phrases = sorted({phrase for items in phrases.values() for phrase, _w in items})
            phrase_matrix = encoder.encode(unique_phrases) if unique_phrases else np.zeros((0, encoder.embedding_dimension))
            phrase_q = {
                phrase: float(vec @ question_vec)
                for phrase, vec in zip(unique_phrases, phrase_matrix)
            }
            append_timing(
                timings_path,
                {"stage": "encoder_phrases", "seconds": time.perf_counter() - started, "rss_mb": _rss_mb(), "gpu_mb": _gpu_mb()},
            )

            started = time.perf_counter()
            labels_by_doc: dict[str, list[dict[str, object]]] = {}
            all_label_rows = []
            chunk_embeddings: dict[str, np.ndarray] = {}
            for paper in papers:
                route = route_for(paper)
                item = evidence[paper.doc_id]
                rows = label_document_chunks(item, cluster_id=int(route.top1), split=paper.split, params=params)
                labels_by_doc[paper.doc_id] = rows
                all_label_rows.extend(rows)
                if paper.split == "train":
                    texts = [chunk.text for chunk in item.chunks]
                    chunk_embeddings[paper.doc_id] = encoder.encode(texts) if texts else np.zeros((0, encoder.embedding_dimension))
            write_csv(
                output_dir / "positive_chunks.csv",
                [row for row in all_label_rows if int(row["is_positive"])],
            )
            append_timing(
                timings_path,
                {"stage": "labels", "seconds": time.perf_counter() - started, "rss_mb": _rss_mb(), "gpu_mb": _gpu_mb()},
            )

            started = time.perf_counter()
            hard_by_doc: dict[str, dict[str, list[int]]] = {}
            for paper in train:
                hard_by_doc[paper.doc_id] = hard_negative_indices(
                    evidence[paper.doc_id],
                    labels_by_doc[paper.doc_id],
                    global_keywords=static_keywords,
                    chunk_embeddings=chunk_embeddings.get(paper.doc_id),
                    question_embedding=question_vec,
                    params=params,
                )
            write_csv(
                output_dir / "hard_negative_chunks.csv",
                flatten_hard_negative_table(hard_by_doc, labels_by_doc, strategy="HN5"),
            )
            append_timing(
                timings_path,
                {"stage": "hard_negatives", "seconds": time.perf_counter() - started, "rss_mb": _rss_mb(), "gpu_mb": _gpu_mb()},
            )

            started = time.perf_counter()
            groups_hn = {
                name: cluster_chunk_groups(
                    [row for row in all_label_rows if row["split"] == "train"],
                    hard_by_doc,
                    strategy=name,
                )
                for name in ("HN1", "HN2", "HN3", "HN4", "HN5")
            }
            groups_default = groups_hn["HN5"]

            source_csvs = {
                "attention": [],
                "attention_diff": [],
                "question_keybert": [],
                "topic": [],
            }
            pools_by_cluster: dict[int, dict[str, dict[str, KeywordProfile]]] = {}
            for cluster_id in cluster_ids:
                ids = train_by_cluster[cluster_id]
                att = attention_pool(cluster_id, ids, evidence, require_diff=False)
                att_diff = attention_pool(cluster_id, ids, evidence, require_diff=True)
                qpool = question_keybert_pool(cluster_id, ids, phrases, phrase_q)
                tpool = topic_pool(cluster_id, topic_terms.get(cluster_id, ()))
                for key, profile in att.items():
                    source_csvs["attention"].append({"cluster_id": cluster_id, "keyword": key, **profile.to_row()})
                for key, profile in att_diff.items():
                    source_csvs["attention_diff"].append({"cluster_id": cluster_id, "keyword": key, **profile.to_row()})
                for key, profile in qpool.items():
                    source_csvs["question_keybert"].append({"cluster_id": cluster_id, "keyword": key, **profile.to_row()})
                for key, profile in tpool.items():
                    source_csvs["topic"].append({"cluster_id": cluster_id, "keyword": key, **profile.to_row()})
                pools_by_cluster[cluster_id] = build_named_pools(
                    cluster_id, attention=att, attention_diff=att_diff, question=qpool, topic=tpool
                )
            write_csv(output_dir / "candidate_keywords_attention.csv", source_csvs["attention"])
            write_csv(output_dir / "candidate_keywords_attention_diff.csv", source_csvs["attention_diff"])
            write_csv(output_dir / "candidate_keywords_question_keybert.csv", source_csvs["question_keybert"])
            write_csv(output_dir / "candidate_keywords_topic.csv", source_csvs["topic"])

            def cluster_texts(cluster_id: int, kind: str, strategy: str = "HN5") -> list[str]:
                grouped = groups_hn[strategy][cluster_id]
                return _texts_for(grouped[kind], evidence)

            def fit_h(
                pool: dict[str, KeywordProfile],
                cluster_id: int,
                *,
                strategy: str = "HN5",
                negative: str = "hard",
                local_params: ExperimentParams | None = None,
            ) -> list[KeywordProfile]:
                cfg = local_params or params
                pos = cluster_texts(cluster_id, "positive", strategy)
                hard = cluster_texts(cluster_id, "hard", strategy)
                easy = cluster_texts(cluster_id, "easy", strategy)
                docs = [_doc_text(evidence[doc_id]) for doc_id in train_by_cluster[cluster_id]]
                copies = [
                    KeywordProfile(**{**profile.to_row(), "cluster_id": cluster_id, "keyword": profile.keyword})
                    for profile in pool.values()
                ]
                return score_profiles(copies, pos, hard, easy, docs, params=cfg, negative=negative)

            global_pos = []
            global_hard = []
            global_easy = []
            global_docs = []
            for cluster_id in cluster_ids:
                global_pos.extend(cluster_texts(cluster_id, "positive"))
                global_hard.extend(cluster_texts(cluster_id, "hard"))
                global_easy.extend(cluster_texts(cluster_id, "easy"))
                global_docs.extend(_doc_text(evidence[doc_id]) for doc_id in train_by_cluster[cluster_id])

            def apply_shrinkage(profiles: list[KeywordProfile], cluster_id: int) -> list[KeywordProfile]:
                lam = shrinkage_lambda(len(train_by_cluster[cluster_id]), n0=params.shrinkage_n0)
                global_h = {
                    profile.keyword: profile.discriminative_score
                    for profile in score_profiles(
                        [KeywordProfile(cluster_id=-1, keyword=item.keyword) for item in profiles],
                        global_pos,
                        global_hard,
                        global_easy,
                        global_docs,
                        params=params,
                        negative="hard",
                    )
                }
                for profile in profiles:
                    profile.global_discriminative_score = float(global_h.get(profile.keyword, 0.0))
                    profile.final_weight = blend(
                        profile.discriminative_score, profile.global_discriminative_score, lam
                    )
                profiles.sort(key=lambda item: -abs(item.final_weight))
                return profiles[: params.max_keywords_per_cluster]

            fitted: dict[tuple[str, str], dict[int, list[KeywordProfile]]] = {}
            profile_rows = []
            for pool_name, _sources in POOL_SPECS:
                per_cluster = {}
                for cluster_id in cluster_ids:
                    scored = fit_h(pools_by_cluster[cluster_id][pool_name], cluster_id)
                    scored = apply_shrinkage(scored, cluster_id)
                    per_cluster[cluster_id] = scored
                    for profile in scored:
                        row = profile.to_row()
                        row["pool"] = pool_name
                        profile_rows.append(row)
                fitted[(pool_name, "H_hard")] = per_cluster
                imp = {}
                for cluster_id in cluster_ids:
                    items = []
                    for profile in pools_by_cluster[cluster_id][pool_name].values():
                        copy = KeywordProfile(**profile.to_row())
                        copy.final_weight = importance_weight(profile)
                        if copy.final_weight > 0:
                            items.append(copy)
                    items.sort(key=lambda item: -item.final_weight)
                    imp[cluster_id] = items[: params.max_keywords_per_cluster]
                fitted[(pool_name, "importance")] = imp
            write_csv(output_dir / "cluster_keyword_profiles.csv", profile_rows)
            append_timing(
                timings_path,
                {"stage": "dictionaries", "seconds": time.perf_counter() - started, "rss_mb": _rss_mb(), "gpu_mb": _gpu_mb()},
            )

    def keywords_for(method: tuple[str, str], cluster_id: int, *, keep_negative: bool = True) -> list[WeightedKeyword]:
        profiles = fitted[method].get(cluster_id, [])
        return profiles_to_keywords(profiles, keep_negative=keep_negative)

    def score_method(
        paper,
        method: str,
        *,
        keep_negative: bool = True,
        use_modifiers: bool = True,
        soft: bool = False,
    ) -> list[float]:
        item = evidence[paper.doc_id]
        route = route_for(paper)
        n = len(item.chunks)
        if method == "qa_confidence":
            return [float(chunk.confidence) for chunk in item.chunks]
        if method == "global_keywords":
            return score_document_chunks(item, static_keywords, use_modifiers=use_modifiers)
        pool, kind = method.split(":", 1)
        key = (pool, kind)
        if key not in fitted:
            return [0.0] * n
        per_cluster = {}
        for cluster_id in cluster_ids:
            per_cluster[cluster_id] = score_document_chunks(
                item,
                keywords_for(key, cluster_id, keep_negative=keep_negative),
                use_modifiers=use_modifiers,
            )
        return mix_cluster_scores(per_cluster, route, soft=soft)

    if resume:
        ranking_method_names = ["qa_confidence", "global_keywords"]
        for pool_name, _sources in POOL_SPECS:
            if (pool_name, "H_hard") in fitted:
                ranking_method_names.append(f"{pool_name}:H_hard")
    else:
        ranking_method_names = ["qa_confidence", "global_keywords"]
        for pool_name, _sources in POOL_SPECS:
            ranking_method_names.append(f"{pool_name}:importance")
            ranking_method_names.append(f"{pool_name}:H_hard")

    def evaluate_ranking(method: str, papers_split, **kwargs) -> tuple[dict[str, float], list[dict[str, Any]]]:
        rows = []
        for paper in papers_split:
            scores = score_method(paper, method, **kwargs)
            gold = gold_chunk_index(evidence[paper.doc_id])
            metrics = rank_metrics(scores, gold)
            baseline = rank_metrics(
                score_method(paper, "global_keywords"), gold
            )
            rows.append(
                {
                    "document_id": paper.doc_id,
                    "split": paper.split,
                    "method": method,
                    "cluster_id": int(route_for(paper).top1),
                    **metrics,
                    "rank_up": float(
                        metrics["gold_rank"] > 0
                        and baseline["gold_rank"] > 0
                        and metrics["gold_rank"] < baseline["gold_rank"]
                    ),
                    "rank_down": float(
                        metrics["gold_rank"] > 0
                        and baseline["gold_rank"] > 0
                        and metrics["gold_rank"] > baseline["gold_rank"]
                    ),
                }
            )
        summary = summarize_ranks(rows)
        summary["method"] = method
        summary["split"] = papers_split[0].split if papers_split else ""
        summary["rank_up"] = float(np.mean([row["rank_up"] for row in rows])) if rows else 0.0
        summary["rank_down"] = float(np.mean([row["rank_down"] for row in rows])) if rows else 0.0
        return summary, rows

    started = time.perf_counter()
    if not resume:
        dev_rank_summaries = []
        dev_rank_rows = []
        for name in ranking_method_names:
            summary, rows = evaluate_ranking(name, dev)
            dev_rank_summaries.append(summary)
            dev_rank_rows.extend(rows)
        write_csv(output_dir / "chunk_ranking_results.csv", dev_rank_rows)

    h_rows = [row for row in dev_rank_summaries if str(row["method"]).endswith(":H_hard")]
    best_h = max(h_rows, key=lambda row: (float(row["evidence_recall@5"]), float(row["mrr"]))) if h_rows else None
    best_pool = str(best_h["method"]).split(":")[0] if best_h else "K1"

    # Ablations for the winning pool on DEV only. Skip on resume: HN dicts are not in the CSV.
    ablation_names = []
    if (not resume) and (best_pool, "H_hard") in fitted:
        easy_fit = {}
        for cluster_id in cluster_ids:
            scored = fit_h(pools_by_cluster[cluster_id][best_pool], cluster_id, negative="easy")
            easy_fit[cluster_id] = apply_shrinkage(scored, cluster_id)
        fitted[(best_pool, "H_easy")] = easy_fit
        ablation_names.append(f"{best_pool}:H_easy")
        for hn_name in ("HN1", "HN2", "HN3", "HN4"):
            per_cluster = {}
            for cluster_id in cluster_ids:
                scored = fit_h(
                    pools_by_cluster[cluster_id][best_pool], cluster_id, strategy=hn_name
                )
                per_cluster[cluster_id] = apply_shrinkage(scored, cluster_id)
            fitted[(best_pool, f"H_{hn_name}")] = per_cluster
            ranking_method_names.append(f"{best_pool}:H_{hn_name}")
            ablation_names.append(f"{best_pool}:H_{hn_name}")
        for extra in ablation_names:
            if extra.split(":", 1)[0] == best_pool:
                summary, rows = evaluate_ranking(extra, dev)
                dev_rank_summaries.append(summary)
                dev_rank_rows.extend(rows)
        summary_pos, rows_pos = evaluate_ranking(
            f"{best_pool}:H_hard", dev, keep_negative=False
        )
        summary_pos["method"] = f"{best_pool}:H_hard_positive_only"
        dev_rank_summaries.append(summary_pos)
        for row in rows_pos:
            row["method"] = summary_pos["method"]
        dev_rank_rows.extend(rows_pos)
        summary_pure, rows_pure = evaluate_ranking(
            f"{best_pool}:H_hard", dev, use_modifiers=False
        )
        summary_pure["method"] = f"{best_pool}:H_hard_pure"
        dev_rank_summaries.append(summary_pure)
        for row in rows_pure:
            row["method"] = summary_pure["method"]
        dev_rank_rows.extend(rows_pure)
        summary_soft, rows_soft = evaluate_ranking(
            f"{best_pool}:H_hard", dev, soft=True
        )
        summary_soft["method"] = f"{best_pool}:H_hard_soft"
        dev_rank_summaries.append(summary_soft)
        for row in rows_soft:
            row["method"] = summary_soft["method"]
        dev_rank_rows.extend(rows_soft)

    if not resume:
        write_csv(output_dir / "chunk_recall_metrics.csv", dev_rank_summaries)
        write_csv(output_dir / "chunk_ranking_results.csv", dev_rank_rows)
    append_timing(
        timings_path,
        {
            "stage": "ranking_dev" if not resume else "resume_ranking",
            "seconds": time.perf_counter() - started,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        },
    )

    core_dev = [
        row
        for row in dev_rank_summaries
        if row["method"] in ranking_method_names
        and not str(row["method"]).startswith("qa_confidence")
    ]
    core_dev.sort(key=lambda row: (-float(row["evidence_recall@5"]), -float(row["mrr"])))
    top_methods = [row["method"] for row in core_dev[:4]]
    global_dev = next(row for row in dev_rank_summaries if row["method"] == "global_keywords")
    gate_methods = [
        name
        for name in top_methods
        if float(next(row for row in dev_rank_summaries if row["method"] == name)["evidence_recall@5"])
        + 1e-12
        >= float(global_dev["evidence_recall@5"]) - 0.02
    ]
    if not gate_methods:
        gate_methods = top_methods[:1]
    gate_methods = [
        name
        for name in gate_methods
        if name in {"qa_confidence", "global_keywords"}
        or (":" in name and tuple(name.split(":", 1)) in fitted)
    ]
    if not gate_methods:
        gate_methods = [
            name
            for name in top_methods
            if name in {"qa_confidence", "global_keywords"}
            or (":" in name and tuple(name.split(":", 1)) in fitted)
        ][:1]

    metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")

    def exact_match(prediction: str | None, refs: Sequence[str]) -> float:
        predicted = normalize_pred(prediction).casefold()
        if not predicted:
            return 0.0
        return float(any(predicted == str(item).strip().casefold() for item in refs if str(item).strip()))

    def score_pred(prediction: str | None, refs: tuple[str, ...]) -> dict[str, float]:
        metrics = metric_cache.score(
            prediction,
            list(refs),
            language=args.language,
            include_bertscore=not args.skip_bertscore,
        )
        metrics["exact_match"] = exact_match(prediction, refs)
        nonempty = bool(str(prediction or "").strip())
        metrics["coverage"] = float(nonempty)
        metrics["false_positive"] = float(nonempty and float(metrics.get("char_f1", 0.0)) <= 0.0)
        return metrics

    def qa_for_method(paper, method: str, recipe: str, **kwargs) -> tuple[str | None, dict[str, Any], list[float]]:
        item = evidence[paper.doc_id]
        if recipe == "Q0":
            pred, meta = predict_q0(item)
            scores = [float(chunk.confidence) for chunk in item.chunks]
            return pred, meta, scores
        scores = score_method(paper, method) if method != "global_keywords" else score_method(paper, "global_keywords")
        if recipe == "B_consensus":
            pred, meta = predict_consensus(item, scores)
        elif recipe == "B_linear":
            pred, meta = predict_linear(
                item, scores, alpha=float(kwargs["alpha"]), beta=float(kwargs["beta"])
            )
        elif recipe == "C":
            pred, meta = predict_prefilter_centroid(item, scores, top_k=int(kwargs["top_k"]))
        elif recipe == "D":
            pred, meta = predict_hybrid(
                item,
                scores,
                top_k=int(kwargs["top_k"]),
                alpha=float(kwargs.get("alpha", 0.5)),
                beta=float(kwargs.get("beta", 0.5)),
                use_consensus=bool(kwargs.get("use_consensus", True)),
            )
        else:
            raise ValueError(recipe)
        return pred, meta, scores

    mix_grid = ((1.0, 0.0), (0.0, 1.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7))
    k_grid = (5, 10, 20) if not args.quick else (5, 10)
    best_method = gate_methods[0]
    best_mix = (0.5, 0.5)
    best_mix_score = -1.0
    for alpha, beta in mix_grid:
        scores = []
        for paper in dev:
            pred, _meta, _s = qa_for_method(paper, best_method, "B_linear", alpha=alpha, beta=beta)
            scores.append(char_f1(pred, list(paper.references)))
        mean = float(np.mean(scores)) if scores else 0.0
        if mean > best_mix_score:
            best_mix_score = mean
            best_mix = (alpha, beta)

    def mean_dev_recipe(recipe: str, **kwargs) -> float:
        scores = []
        for paper in dev:
            pred, _meta, _s = qa_for_method(paper, best_method, recipe, **kwargs)
            scores.append(char_f1(pred, list(paper.references)))
        return float(np.mean(scores)) if scores else 0.0

    consensus_dev = mean_dev_recipe("B_consensus")
    use_consensus_b = consensus_dev >= best_mix_score
    best_k = 10
    best_k_score = -1.0
    allowed_k = []
    for top_k in k_grid:
        rec = next(row for row in dev_rank_summaries if row["method"] == best_method)
        key = f"evidence_recall@{top_k}" if top_k != 10 else "evidence_recall@10"
        recall = float(rec.get(key, rec["evidence_recall@5"]))
        global_recall = float(global_dev.get(key, global_dev["evidence_recall@5"]))
        if recall + 1e-12 < global_recall - 0.02:
            continue
        allowed_k.append(top_k)
        score = mean_dev_recipe("C", top_k=top_k)
        if score > best_k_score:
            best_k_score = score
            best_k = top_k
    if not allowed_k:
        best_k = None

    qa_specs = [("Q0", "qa_confidence", "Q0", {}), ("Q1", "global_keywords", "B_consensus", {})]
    for method in gate_methods:
        qa_specs.append(
            (
                f"{method}/B",
                method,
                "B_consensus" if (method != best_method or use_consensus_b) else "B_linear",
                {"alpha": best_mix[0], "beta": best_mix[1]},
            )
        )
    if best_k is not None:
        qa_specs.append((f"{best_method}/C_k{best_k}", best_method, "C", {"top_k": best_k}))
        qa_specs.append(
            (
                f"{best_method}/D_k{best_k}",
                best_method,
                "D",
                {
                    "top_k": best_k,
                    "alpha": best_mix[0],
                    "beta": best_mix[1],
                    "use_consensus": use_consensus_b,
                },
            )
        )

    started = time.perf_counter()
    test_rank_summaries = []
    test_rank_rows = []
    for name in ["qa_confidence", "global_keywords", *top_methods]:
        summary, rows = evaluate_ranking(name, test)
        summary["split"] = "test"
        test_rank_summaries.append(summary)
        test_rank_rows.extend(rows)

    q0_chunks = [len(evidence[paper.doc_id].chunks) for paper in test]
    q0_tokens = [sum(chunk.token_count for chunk in evidence[paper.doc_id].chunks) for paper in test]
    baseline_chunks = float(np.mean(q0_chunks)) if q0_chunks else 1.0
    baseline_tokens = float(np.mean(q0_tokens)) if q0_tokens else 1.0

    qa_pred_rows = []
    qa_metric_rows = []
    cluster_qa: dict[tuple[str, int], list[float]] = defaultdict(list)
    for spec_name, method, recipe, kwargs in qa_specs:
        per_doc_f1 = []
        per_recall = []
        compute_rows = []
        t0 = time.perf_counter()
        for paper in test:
            t_doc = time.perf_counter()
            pred, meta, scores = qa_for_method(paper, method, recipe, **kwargs)
            metrics = score_pred(pred, paper.references)
            gold = gold_chunk_index(evidence[paper.doc_id])
            retrieval = rank_metrics(scores, gold)
            per_doc_f1.append(float(metrics["char_f1"]))
            per_recall.append(float(retrieval["evidence_recall@5"]))
            cluster_qa[(spec_name, int(route_for(paper).top1))].append(float(metrics["char_f1"]))
            qa_pred_rows.append(
                {
                    "document_id": paper.doc_id,
                    "split": "test",
                    "method": spec_name,
                    "ranking_method": method,
                    "recipe": recipe,
                    "prediction": pred or "",
                    "cluster_id": int(route_for(paper).top1),
                    **metrics,
                    **retrieval,
                    "qa_chunks": meta.get("qa_chunks", ""),
                    "qa_tokens": meta.get("qa_tokens", ""),
                    "fallback": meta.get("fallback", ""),
                    "latency_s": time.perf_counter() - t_doc,
                }
            )
            compute_rows.append(meta)
        mean, lo, hi = bootstrap_mean_ci(per_doc_f1, seed=args.seed)
        qa_chunks = [float(item.get("qa_chunks") or 0) for item in compute_rows]
        qa_tokens = [float(item.get("qa_tokens") or 0) for item in compute_rows]
        qa_metric_rows.append(
            {
                "method": spec_name,
                "ranking_method": method,
                "recipe": recipe,
                "n": len(test),
                "char_f1": mean,
                "char_f1_ci95_low": lo,
                "char_f1_ci95_high": hi,
                "token_f1": float(np.mean([float(row["token_f1"]) for row in qa_pred_rows if row["method"] == spec_name])),
                "rouge_l_f1": float(np.mean([float(row["rouge_l_f1"]) for row in qa_pred_rows if row["method"] == spec_name])),
                "bertscore_f1": float(np.mean([float(row.get("bertscore_f1") or 0.0) for row in qa_pred_rows if row["method"] == spec_name])),
                "exact_match": float(np.mean([float(row["exact_match"]) for row in qa_pred_rows if row["method"] == spec_name])),
                "evidence_recall@5": float(np.mean(per_recall)) if per_recall else 0.0,
                "qa_chunks_mean": float(np.mean(qa_chunks)) if qa_chunks else 0.0,
                "qa_tokens_mean": float(np.mean(qa_tokens)) if qa_tokens else 0.0,
                "qa_chunk_ratio": (float(np.mean(qa_chunks)) / baseline_chunks) if baseline_chunks else 0.0,
                "qa_token_ratio": (float(np.mean(qa_tokens)) / baseline_tokens) if baseline_tokens else 0.0,
                "seconds": time.perf_counter() - t0,
            }
        )
    write_csv(output_dir / "qa_predictions.csv", qa_pred_rows)
    write_csv(output_dir / "qa_metrics.csv", qa_metric_rows)
    metric_cache.save()

    cluster_rows = []
    q0_by_cluster: dict[int, list[float]] = defaultdict(list)
    for paper in test:
        q0_by_cluster[int(route_for(paper).top1)].append(
            char_f1(evidence[paper.doc_id].baseline_answer, list(paper.references))
        )
    for spec_name, _method, _recipe, _kw in qa_specs:
        for cluster_id in cluster_ids:
            values = cluster_qa.get((spec_name, cluster_id), [])
            base = q0_by_cluster.get(cluster_id, [])
            if not values:
                continue
            terms = [term for term, _w in topic_terms.get(cluster_id, [])[:5]]
            cluster_rows.append(
                {
                    "method": spec_name,
                    "cluster_id": cluster_id,
                    "document_count": len(values),
                    "representative_terms": ", ".join(terms),
                    "baseline_f1": float(np.mean(base)) if base else 0.0,
                    "new_f1": float(np.mean(values)),
                    "delta_f1": float(np.mean(values) - (np.mean(base) if base else 0.0)),
                    "harm_rate": float(
                        np.mean(
                            [
                                float(new < old - 1e-12)
                                for new, old in zip(values, base)
                            ]
                        )
                    )
                    if base and len(base) == len(values)
                    else "",
                }
            )
    write_csv(output_dir / "qa_metrics_by_cluster.csv", cluster_rows)

    compute_rows = [
        {
            "method": row["method"],
            "qa_chunks_mean": row["qa_chunks_mean"],
            "qa_tokens_mean": row["qa_tokens_mean"],
            "qa_chunk_ratio": row["qa_chunk_ratio"],
            "qa_token_ratio": row["qa_token_ratio"],
            "seconds": row["seconds"],
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        }
        for row in qa_metric_rows
    ]
    write_csv(output_dir / "compute_metrics.csv", compute_rows)
    write_csv(output_dir / "chunk_recall_metrics_test.csv", test_rank_summaries)
    append_timing(
        timings_path,
        {"stage": "qa_test", "seconds": time.perf_counter() - started, "rss_mb": _rss_mb(), "gpu_mb": _gpu_mb()},
    )

    def lookup_dev(name: str) -> float:
        row = next((item for item in dev_rank_summaries if item["method"] == name), None)
        return float(row["evidence_recall@5"]) if row else 0.0

    hypotheses = {
        "H1 ClusterAttention > GlobalAttention": (
            "supported"
            if lookup_dev("K1:H_hard") > lookup_dev("global_keywords")
            else "not supported"
        )
        + f" (dev R@5 {lookup_dev('K1:H_hard'):.3f} vs {lookup_dev('global_keywords'):.3f})",
        "H2 ClusterAttentionDiff > ClusterAttention": (
            "supported" if lookup_dev("K2:H_hard") > lookup_dev("K1:H_hard") else "not supported"
        )
        + f" (dev R@5 {lookup_dev('K2:H_hard'):.3f} vs {lookup_dev('K1:H_hard'):.3f})",
        "H3 QuestionKeyBERT > ClusterTopic": (
            "supported" if lookup_dev("K3:H_hard") > lookup_dev("K4:H_hard") else "not supported"
        )
        + f" (dev R@5 {lookup_dev('K3:H_hard'):.3f} vs {lookup_dev('K4:H_hard'):.3f})",
        "H4 Attention+QuestionKeyBERT > Attention": (
            "supported" if lookup_dev("K5:H_hard") > lookup_dev("K1:H_hard") else "not supported"
        )
        + f" (dev R@5 {lookup_dev('K5:H_hard'):.3f} vs {lookup_dev('K1:H_hard'):.3f})",
        "H5 HardNegative > CandidateImportance": (
            "supported"
            if lookup_dev(f"{best_pool}:H_hard") > lookup_dev(f"{best_pool}:importance")
            else "not supported"
        )
        + f" (dev R@5 {lookup_dev(f'{best_pool}:H_hard'):.3f} vs {lookup_dev(f'{best_pool}:importance'):.3f})",
        "H6 ClusterSpecific > Global": (
            "supported" if lookup_dev(best_method) > lookup_dev("global_keywords") else "not supported"
        )
        + f" ({best_method} {lookup_dev(best_method):.3f} vs global {lookup_dev('global_keywords'):.3f})",
        "H7 Hybrid Pareto": "see qa_metrics.csv quality vs qa_chunk_ratio",
    }
    chosen = {
        "ranking_method": best_method,
        "top_ranking_methods": top_methods,
        "gate_methods": gate_methods,
        "qa_recipe": (
            f"B_consensus={use_consensus_b}, mix={best_mix}, prefilter_k={best_k}"
        ),
        "best_pool": best_pool,
    }
    write_json(output_dir / "chosen_config.json", chosen)
    write_json(output_dir / "hypotheses.json", hypotheses)
    write_summary(
        output_dir / "summary.md",
        question=args.question,
        clustering=FROZEN_CLUSTERING,
        ranking_rows=dev_rank_summaries,
        qa_rows=qa_metric_rows,
        chosen=chosen,
        hypotheses=hypotheses,
    )
    print(f"Wrote cluster-discriminative artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
