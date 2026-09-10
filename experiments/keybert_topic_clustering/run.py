"""Isolated KeyBERT topic-clustering + cluster-keyword QA experiment.

Does not modify production UnTIE behaviour. Writes only under --output-dir.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KeyBERT topic clustering experiment with downstream extractive QA"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument(
        "--topic-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "clustering", "downstream"),
        default="all",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    return parser


def _versions() -> dict[str, str]:
    out = {}
    try:
        import keybert

        out["keybert"] = keybert.__version__
    except Exception:
        out["keybert"] = "not-installed"
    try:
        import sentence_transformers

        out["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        out["sentence_transformers"] = "missing"
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:
        out["sklearn"] = "missing"
    return out


def _hardware(device: str) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"{device}; {name}"
    except Exception:
        pass
    return device


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
        return 0.0
    return 0.0


def _mean_words(texts) -> float:
    lengths = [len(str(text).split()) for text in texts]
    return float(sum(lengths) / max(len(lengths), 1))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.keybert_topic_clustering.ce_contract import ce_contract
    from experiments.keybert_topic_clustering.documents import load_papers, split_groups
    from experiments.keybert_topic_clustering.downstream import (
        QUESTION,
        bootstrap_mean_ci,
        collect_evidence,
        fit_cluster_dictionary,
        predict_d0,
        predict_keyword_mode,
        predict_q0,
        predict_q1,
        retrieval_from_rows,
    )
    from experiments.keybert_topic_clustering.encoder import CachedSentenceEncoder
    from experiments.keybert_topic_clustering.io_util import (
        append_timing,
        assert_isolated_output,
        write_csv,
        write_json,
    )
    from experiments.keybert_topic_clustering.keyphrases import KeyphraseConfig
    from experiments.keybert_topic_clustering.phrase_analysis import (
        cluster_keyphrases,
        document_phrase_overlap,
    )
    from experiments.keybert_topic_clustering.report import (
        conclude,
        markdown_table,
        pca_scatter,
        size_histogram,
        write_summary,
    )
    from experiments.keybert_topic_clustering.search import (
        SearchContext,
        freeze_and_describe,
        run_staged_search,
    )
    from untie.attention import AttentionKeywordExtractor
    from untie.cli import load_static_keywords
    from untie.config import PipelineConfig
    from untie.keyword_evidence import EvidenceStore, ExtractionMetricCache
    from untie.keyword_training import StrategyConfig
    from untie.models import ModelFactory, profile_for_language, resolve_device
    from untie.pipelines import DocumentProcessor

    output_dir = assert_isolated_output(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_dir / "timings.csv"
    versions = _versions()
    device = resolve_device(args.device)
    papers = load_papers(args.dataset, seed=args.seed, quick=args.quick)
    groups = split_groups(papers)
    train, dev, test = groups["train"], groups["dev"], groups["test"]
    config_payload = {
        "dataset": str(args.dataset),
        "output_dir": str(output_dir),
        "static_model": str(args.static_model),
        "language": args.language,
        "seed": args.seed,
        "question": args.question or QUESTION,
        "quick": args.quick,
        "topic_encoder": args.topic_encoder,
        "ce_contract": ce_contract(),
        "anti_leakage": {
            "train": "clustering, centroids, KeyBERT vocab, cluster keywords",
            "dev": "representation/K/algorithm/MMR/alpha/routing/keyword count",
            "test": "final evaluation only",
        },
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "metadata_empty_rate": sum(item.metadata_empty for item in papers) / max(len(papers), 1),
        "versions": versions,
        "hardware": _hardware(device),
    }
    write_json(output_dir / "experiment_config.json", config_payload)

    encoder = CachedSentenceEncoder(
        args.topic_encoder,
        device=device,
        cache_dir=output_dir / "cache" / "topic_embeddings",
        normalize=True,
    )
    write_json(output_dir / "encoder_spec.json", encoder.spec())
    default_kp = KeyphraseConfig(1, 3, 10, False, 0.5)
    ctx = SearchContext(
        train,
        dev,
        encoder,
        keyphrase_dir=output_dir / "cache" / "keyphrases",
        default_phrases=default_kp,
    )

    chosen_path = output_dir / "chosen_configs.json"
    if args.stage in {"all", "clustering"}:
        started = time.perf_counter()
        search = run_staged_search(
            ctx,
            k_grid=(4, 8) if args.quick else (4, 8, 16),
            stage_k=4 if args.quick else 8,
            seed=args.seed,
            quick=args.quick,
        )
        append_timing(
            timings_path,
            {
                "stage": "clustering_search",
                "seconds": time.perf_counter() - started,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
            },
        )
        candidate_rows = []
        for score, item in search["ranked"]:
            candidate_rows.append(
                {
                    "config_id": item.config_id,
                    "family": item.spec.family,
                    "text_field": item.spec.text_field,
                    "alpha": item.spec.alpha,
                    "keyphrase_config": item.spec.keyphrase_config,
                    "algorithm": item.algorithm,
                    "k": item.k,
                    "silhouette": item.model.silhouette,
                    "davies_bouldin": item.model.davies_bouldin,
                    "calinski_harabasz": item.model.calinski_harabasz,
                    "cluster_size_min": item.model.size_min,
                    "cluster_size_max": item.model.size_max,
                    "cluster_size_mean": item.model.size_mean,
                    "cluster_size_median": item.model.size_median,
                    "mean_dev_top1": item.mean_dev_top1,
                    "train_time_s": item.train_time_s,
                    "selection_score": score,
                    "uses_keybert": item.uses_keybert,
                }
            )
        write_csv(output_dir / "cluster_metrics.csv", candidate_rows)
        frozen = {}
        for tag, cand in (("C_plain", search["c_plain"]), ("C_keybert", search["c_keybert"])):
            started = time.perf_counter()
            payload = freeze_and_describe(
                ctx,
                cand,
                all_papers=papers,
                output_dir=output_dir,
                stability_seeds=(1, 2) if args.quick else (1, 2, 3, 4, 5),
            )
            append_timing(
                timings_path,
                {
                    "stage": f"freeze_{tag}",
                    "seconds": time.perf_counter() - started,
                    "rss_mb": _rss_mb(),
                    "gpu_mb": _gpu_mb(),
                },
            )
            np.save(output_dir / f"{tag}_centroids.npy", payload["centroids"])
            np.save(output_dir / f"{tag}_train_vectors.npy", payload["train_vectors"])
            write_csv(output_dir / f"{tag}_cluster_assignments.csv", payload["assignments"])
            routing_rows = [
                {
                    "document_id": item.doc_id,
                    "split": item.split,
                    "cluster_id": item.top1,
                    "similarity": item.top1_similarity,
                    "normalized_weight": item.weights[0] if item.weights else 1.0,
                    "top_ids": list(item.top_ids),
                    "weights": list(item.weights),
                }
                for item in payload["routing"]
            ]
            write_csv(output_dir / f"{tag}_routing_simulation.csv", routing_rows)
            term_rows = []
            for cluster_id, terms in payload["representative_terms"].items():
                for item in terms:
                    term_rows.append({"cluster_id": cluster_id, **item})
            write_csv(output_dir / f"{tag}_cluster_representative_terms.csv", term_rows)
            phrase_rows = [
                {
                    "document_id": doc_id,
                    "phrase": item.phrase,
                    "weight": item.weight,
                    "config": payload["keyphrase_config"].key,
                }
                for doc_id, items in payload["phrases"].items()
                for item in items
            ]
            write_csv(output_dir / f"{tag}_document_keyphrases.csv", phrase_rows)
            phrase_clusters = cluster_keyphrases(payload["phrases"], encoder, k=8 if args.quick else 16)
            write_json(output_dir / f"{tag}_keyphrase_clusters.json", phrase_clusters)
            overlap = document_phrase_overlap(
                train,
                payload["train_labels"],
                phrase_clusters["phrases"],
                payload["phrases"],
            )
            write_csv(output_dir / f"{tag}_document_keyphrase_overlap.csv", overlap)
            pca_scatter(
                output_dir / "plots" / f"{tag}_pca.png",
                payload["train_vectors"],
                np.asarray(list(payload["train_labels"].values())),
                f"{tag} train PCA",
            )
            size_histogram(
                output_dir / "plots" / f"{tag}_sizes.png",
                np.asarray(list(payload["train_labels"].values())),
            )
            frozen[tag] = {
                "config_id": payload["config_id"],
                "spec": payload["spec"],
                "algorithm": payload["algorithm"],
                "k": payload["k"],
                "metrics": payload["metrics"],
                "train_labels": payload["train_labels"],
                "nearest_papers": payload["nearest_papers"],
                "representative_terms": payload["representative_terms"],
            }
            # Keep routing objects for downstream via a sidecar pickle-free JSON of routes
            write_json(
                output_dir / f"{tag}_routing.json",
                [
                    {
                        "doc_id": item.doc_id,
                        "split": item.split,
                        "top1": item.top1,
                        "top1_similarity": item.top1_similarity,
                        "top_ids": list(item.top_ids),
                        "weights": list(item.weights),
                    }
                    for item in payload["routing"]
                ],
            )
            write_json(
                output_dir / f"{tag}_train_labels.json",
                payload["train_labels"],
            )
        write_json(
            chosen_path,
            {
                "C_plain": frozen["C_plain"]["config_id"],
                "C_keybert": frozen["C_keybert"]["config_id"],
                "details": {
                    tag: {key: value for key, value in payload.items() if key not in {"nearest_papers"}}
                    for tag, payload in frozen.items()
                },
            },
        )
        if args.stage == "clustering":
            write_summary(
                output_dir / "summary.md",
                {
                    **_summary_dataset(papers, train, dev, test, args, versions, device, encoder),
                    "c_plain": frozen["C_plain"]["config_id"],
                    "c_keybert": frozen["C_keybert"]["config_id"],
                    "clustering_table_md": markdown_table(
                        ["id", "sil", "DB", "CH", "dev_top1"],
                        [
                            [
                                row["config_id"],
                                f"{row['silhouette']:.3f}",
                                f"{row['davies_bouldin']:.3f}",
                                f"{row['calinski_harabasz']:.1f}",
                                f"{row['mean_dev_top1']:.3f}",
                            ]
                            for row in candidate_rows[:12]
                        ],
                    ),
                    "downstream_table_md": "Downstream not run (stage=clustering).",
                    "ablations_md": "",
                    "conclusion_md": "Clustering stage complete. Run --stage downstream next.",
                },
            )
            print(f"Wrote clustering artifacts to {output_dir}")
            return 0

    if args.stage in {"all", "downstream"}:
        from experiments.keybert_topic_clustering.routing import RoutingResult

        if not chosen_path.is_file():
            raise SystemExit("chosen_configs.json missing; run --stage clustering first")
        static_keywords, static_strategy = load_static_keywords(args.static_model, args.field_id)
        profile = profile_for_language(args.language)
        pipeline_config = PipelineConfig(profile=profile, device=args.device)
        models = ModelFactory(pipeline_config)
        processor = DocumentProcessor(
            models.tokenizer, pipeline_config, sentence_encoder=models.sentence_encoder
        )
        attention = AttentionKeywordExtractor(
            models.qa_model, models.tokenizer, models.device
        )
        store = EvidenceStore(output_dir / "evidence")
        started = time.perf_counter()
        evidence = collect_evidence(
            papers,
            store=store,
            processor=processor,
            answerer=models.answerer,
            encoder=models.sentence_encoder,
            attention=attention,
            language=args.language,
            collect_candidates_splits={"train"},
            question=args.question or QUESTION,
        )
        append_timing(
            timings_path,
            {
                "stage": "collect_evidence",
                "seconds": time.perf_counter() - started,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
            },
        )
        metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")
        d1_strategy = StrategyConfig("only_weight", "weighted_score", "combined_score")
        d2_strategy = StrategyConfig(
            "equal_weight_score_diff", "weighted_score", "combined_score"
        )
        predictions: list[dict[str, Any]] = []
        chunk_rows: list[dict[str, Any]] = []
        compute_rows: list[dict[str, Any]] = []
        cluster_quality: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        def score_pred(prediction: str | None, refs: tuple[str, ...]) -> dict[str, float]:
            return metric_cache.score(
                prediction,
                refs,
                language=args.language,
                include_bertscore=not args.skip_bertscore,
            )

        def composite(metrics: dict[str, float]) -> float:
            return (
                float(metrics.get("char_f1", 0.0))
                + float(metrics.get("token_f1", 0.0)) / 100.0
                + float(metrics.get("rouge_l_f1", 0.0))
                + float(metrics.get("bertscore_f1", 0.0))
            ) / 4.0

        # Q0 / Q1 do not depend on clustering
        mode_metrics: dict[str, list[float]] = defaultdict(list)
        for paper in test:
            item = evidence[paper.doc_id]
            started = time.perf_counter()
            for mode, pred, meta in (
                ("Q0", *predict_q0(item)),
                ("Q1", *predict_q1(item, static_keywords, static_strategy)),
            ):
                metrics = score_pred(pred, paper.references)
                row = {
                    "document_id": paper.doc_id,
                    "split": "test",
                    "clustering": "none",
                    "mode": mode,
                    "prediction": pred or "",
                    **metrics,
                    "composite_quality": composite(metrics),
                    "fallback": meta.get("fallback", False),
                    "qa_chunks": meta.get("qa_chunks", 0),
                    "qa_tokens": meta.get("qa_tokens", 0),
                    "latency_s": time.perf_counter() - started,
                }
                predictions.append(row)
                mode_metrics[mode].append(float(metrics["char_f1"]))
                compute_rows.append(
                    {
                        "document_id": paper.doc_id,
                        "mode": mode,
                        "qa_chunks": meta.get("qa_chunks", 0),
                        "qa_tokens": meta.get("qa_tokens", 0),
                        "latency_s": row["latency_s"],
                    }
                )

        for tag in ("C_plain", "C_keybert"):
            train_labels = {
                str(key): int(value)
                for key, value in __import__("json").loads(
                    (output_dir / f"{tag}_train_labels.json").read_text(encoding="utf-8")
                ).items()
            }
            routing_raw = __import__("json").loads(
                (output_dir / f"{tag}_routing.json").read_text(encoding="utf-8")
            )
            routes = {
                item["doc_id"]: RoutingResult(
                    doc_id=item["doc_id"],
                    split=item["split"],
                    top1=int(item["top1"]),
                    top1_similarity=float(item["top1_similarity"]),
                    top_ids=tuple(item["top_ids"]),
                    similarities=tuple(),
                    weights=tuple(item["weights"]),
                )
                for item in routing_raw
            }
            cluster_ids = sorted(set(train_labels.values()))
            train_by_cluster: dict[int, list[str]] = defaultdict(list)
            for doc_id, cluster_id in train_labels.items():
                train_by_cluster[cluster_id].append(doc_id)
            dev_by_cluster: dict[int, list[str]] = defaultdict(list)
            for paper in dev:
                route = routes[paper.doc_id]
                dev_by_cluster[int(route.top1)].append(paper.doc_id)

            started = time.perf_counter()
            d1_dicts: dict[int, list] = {}
            d1_rows = []
            d2_dicts: dict[int, list] = {}
            d2_rows = []
            for cluster_id in cluster_ids:
                kw, info = fit_cluster_dictionary(
                    cluster_id,
                    train_by_cluster[cluster_id],
                    dev_by_cluster.get(cluster_id, []),
                    evidence,
                    strategy=d1_strategy,
                    language=args.language,
                    seed=args.seed,
                    metric_cache=metric_cache,
                )
                if kw:
                    d1_dicts[cluster_id] = kw
                for word in info.get("keyword_metadata", []):
                    d1_rows.append(
                        {
                            "cluster_id": cluster_id,
                            "keyword": word.get("word"),
                            "document_support": word.get("document_support"),
                            "attention_weight": word.get("attention_weight"),
                            "score_difference": "",
                            "combined_weight": word.get("attention_weight"),
                        }
                    )
                kw2, info2 = fit_cluster_dictionary(
                    cluster_id,
                    train_by_cluster[cluster_id],
                    dev_by_cluster.get(cluster_id, []),
                    evidence,
                    strategy=d2_strategy,
                    language=args.language,
                    seed=args.seed,
                    metric_cache=metric_cache,
                )
                if kw2:
                    d2_dicts[cluster_id] = kw2
                for word in info2.get("keyword_metadata", []):
                    att = float(word.get("attention_weight") or 0.0)
                    diff = float(word.get("score_difference") or 0.0)
                    d2_rows.append(
                        {
                            "cluster_id": cluster_id,
                            "keyword": word.get("word"),
                            "document_support": word.get("document_support"),
                            "attention_weight": att,
                            "score_difference": diff,
                            "combined_weight": 0.5 * att + 0.5 * diff,
                        }
                    )
            write_csv(output_dir / f"{tag}_cluster_attention_keywords.csv", d1_rows)
            write_csv(output_dir / f"{tag}_cluster_attention_diff_keywords.csv", d2_rows)
            append_timing(
                timings_path,
                {
                    "stage": f"fit_keywords_{tag}",
                    "seconds": time.perf_counter() - started,
                    "rss_mb": _rss_mb(),
                    "gpu_mb": _gpu_mb(),
                },
            )

            # pick hard vs soft on dev using D1 char-F1
            def eval_split(split_papers, hard: bool, dictionaries, ratio: float, mode: str):
                local_rows = []
                for paper in split_papers:
                    route = routes[paper.doc_id]
                    pred, meta, chunks = predict_keyword_mode(
                        evidence[paper.doc_id],
                        route,
                        dictionaries,
                        static_keywords,
                        weight_ratio=ratio,
                        hard=hard,
                    )
                    metrics = score_pred(pred, paper.references)
                    local_rows.append((paper, pred, meta, chunks, metrics, route))
                mean = float(np.mean([item[4]["char_f1"] for item in local_rows])) if local_rows else 0.0
                return mean, local_rows

            dev_hard, _ = eval_split(dev, True, d1_dicts, 1.0, "D1")
            dev_soft, _ = eval_split(dev, False, d1_dicts, 1.0, "D1")
            use_hard = dev_hard >= dev_soft
            clustering_name = "plain" if tag == "C_plain" else "keybert"
            for mode, dictionaries, ratio in (
                ("D0", None, None),
                ("D1", d1_dicts, 1.0),
                ("D2", d2_dicts, 0.5),
            ):
                for paper in test:
                    route = routes[paper.doc_id]
                    started = time.perf_counter()
                    if mode == "D0":
                        pred, meta = predict_d0(evidence[paper.doc_id])
                        chunks = []
                    else:
                        pred, meta, chunks = predict_keyword_mode(
                            evidence[paper.doc_id],
                            route,
                            dictionaries or {},
                            static_keywords,
                            weight_ratio=float(ratio),
                            hard=use_hard,
                        )
                    metrics = score_pred(pred, paper.references)
                    row = {
                        "document_id": paper.doc_id,
                        "split": "test",
                        "clustering": clustering_name,
                        "mode": mode,
                        "prediction": pred or "",
                        **metrics,
                        "composite_quality": composite(metrics),
                        "fallback": meta.get("fallback", False),
                        "qa_chunks": meta.get("qa_chunks", 0),
                        "qa_tokens": meta.get("qa_tokens", 0),
                        "routed_cluster_ids": list(route.top_ids),
                        "routing_weights": list(route.weights),
                        "latency_s": time.perf_counter() - started,
                        "hard_routing": use_hard,
                    }
                    predictions.append(row)
                    key = f"{clustering_name}/{mode}"
                    mode_metrics[key].append(float(metrics["char_f1"]))
                    cluster_quality[f"{clustering_name}/{mode}"][int(route.top1)].append(
                        float(metrics["char_f1"])
                    )
                    retrieval = retrieval_from_rows(chunks) if chunks else {}
                    compute_rows.append(
                        {
                            "document_id": paper.doc_id,
                            "clustering": clustering_name,
                            "mode": mode,
                            "qa_chunks": meta.get("qa_chunks", 0),
                            "qa_tokens": meta.get("qa_tokens", 0),
                            "latency_s": row["latency_s"],
                            **retrieval,
                        }
                    )
                    for chunk in chunks:
                        chunk_rows.append(
                            {
                                "document_id": paper.doc_id,
                                "split": "test",
                                "clustering": clustering_name,
                                "mode": mode,
                                "routed_cluster_ids": list(route.top_ids),
                                "routing_weights": list(route.weights),
                                **chunk,
                            }
                        )

        write_csv(output_dir / "downstream_predictions.csv", predictions)
        write_csv(output_dir / "downstream_chunk_scores.csv", chunk_rows)
        write_csv(output_dir / "downstream_compute.csv", compute_rows)

        summary_rows = []
        baseline_vals = mode_metrics.get("Q0", [])
        by_doc_q0 = {
            row["document_id"]: row
            for row in predictions
            if row["mode"] == "Q0"
        }
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[f"{row['clustering']}/{row['mode']}"].append(row)
        metric_map = {}
        for key, rows in grouped.items():
            f1s = [float(row["char_f1"]) for row in rows]
            mean, lo, hi = bootstrap_mean_ci(f1s)
            deltas = []
            wins = harms = 0
            for row in rows:
                base = float(by_doc_q0[row["document_id"]]["char_f1"])
                delta = float(row["char_f1"]) - base
                deltas.append(delta)
                if delta > 1e-12:
                    wins += 1
                elif delta < -1e-12:
                    harms += 1
            rec5 = [
                float(item.get("evidence_recall@5", 0.0) or 0.0)
                for item in compute_rows
                if f"{item.get('clustering', 'none')}/{item['mode']}" == key
                or (key.endswith("/Q0") and item["mode"] == "Q0")
                or (key.endswith("/Q1") and item["mode"] == "Q1")
            ]
            summary_rows.append(
                {
                    "clustering": key.split("/")[0],
                    "mode": key.split("/")[1],
                    "mean_char_f1": mean,
                    "char_f1_ci_low": lo,
                    "char_f1_ci_high": hi,
                    "mean_delta_char_f1": float(np.mean(deltas)) if deltas else 0.0,
                    "median_delta_char_f1": float(np.median(deltas)) if deltas else 0.0,
                    "win_rate": wins / max(len(rows), 1),
                    "harm_rate": harms / max(len(rows), 1),
                    "fallback_rate": sum(bool(row["fallback"]) for row in rows) / max(len(rows), 1),
                    "mean_qa_chunks": float(np.mean([row["qa_chunks"] for row in rows])),
                    "mean_latency_s": float(np.mean([row["latency_s"] for row in rows])),
                    "documents": len(rows),
                }
            )
            metric_map[key.replace("none/", "")] = {"mean_char_f1": mean}
        write_csv(output_dir / "downstream_metrics.csv", summary_rows)
        by_cluster_rows = []
        for key, clusters in cluster_quality.items():
            clustering, mode = key.split("/")
            for cluster_id, values in clusters.items():
                by_cluster_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "clustering": clustering,
                        "mode": mode,
                        "document_count": len(values),
                        "mean_char_f1": float(np.mean(values)),
                    }
                )
        write_csv(output_dir / "downstream_metrics_by_cluster.csv", by_cluster_rows)
        metric_cache.save()

        table = markdown_table(
            ["Clustering", "QA mode", "char-F1", "ΔQ0", "Win", "Harm", "QA chunks", "Time"],
            [
                [
                    row["clustering"],
                    row["mode"],
                    f"{row['mean_char_f1']:.3f}",
                    f"{row['mean_delta_char_f1']:+.3f}",
                    f"{row['win_rate']:.2f}",
                    f"{row['harm_rate']:.2f}",
                    f"{row['mean_qa_chunks']:.1f}",
                    f"{row['mean_latency_s']:.2f}",
                ]
                for row in summary_rows
            ],
        )
        write_summary(
            output_dir / "summary.md",
            {
                **_summary_dataset(papers, train, dev, test, args, versions, device, encoder),
                "c_plain": __import__("json").loads(chosen_path.read_text())["C_plain"],
                "c_keybert": __import__("json").loads(chosen_path.read_text())["C_keybert"],
                "clustering_table_md": "(see cluster_metrics.csv)",
                "downstream_table_md": table,
                "ablations_md": conclude(metric_map),
                "conclusion_md": conclude(metric_map),
                "ce": "ExtractiveQuestionAnswerer / AnswerFinder",
            },
        )
        print(f"Wrote downstream artifacts to {output_dir}")
    return 0


def _summary_dataset(papers, train, dev, test, args, versions, device, encoder) -> dict[str, object]:
    return {
        "n_documents": len(papers),
        "language": args.language,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "metadata_empty_rate": f"{sum(item.metadata_empty for item in papers) / max(len(papers), 1):.2f}",
        "mean_abstract_words": f"{_mean_words(item.abstract for item in papers):.1f}",
        "keybert_version": versions.get("keybert"),
        "st_version": versions.get("sentence_transformers"),
        "sklearn_version": versions.get("sklearn"),
        "topic_encoder": f"{encoder.model_name} rev={encoder.revision} dim={encoder.embedding_dimension} L2={encoder.normalize}",
        "hardware": _hardware(device),
        "ce": "untie.models.ExtractiveQuestionAnswerer",
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
