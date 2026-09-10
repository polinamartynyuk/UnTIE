"""Isolated KeyBERT + extractive QA integration experiment.

Does not modify production UnTIE behaviour. Writes only under --output-dir.
Does not retrain RoBERTa unless --run-training is passed (still skipped: no trainer).
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrate KeyBERT as a thematic prior on top of extractive RoBERTa QA"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--static-model",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/analysis_results/keyword_tuning_task/en/full_v5/scart_tuned_model.json",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/keybert_topic_clustering/en_full/evidence",
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--include-chunk-keybert",
        action="store_true",
        help="Also extract KeyBERT phrases per chunk for the winning mix (slow).",
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Requested by the prompt's priority-3 experiments. Default off; this repo has no RoBERTa trainer.",
    )
    parser.add_argument(
        "--keybert-weight",
        type=float,
        default=None,
        help="Optional override; otherwise fitted on dev.",
    )
    return parser


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
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
            return f"{device}; {torch.cuda.get_device_name(0)}"
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
        pass
    return 0.0


def _reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def exact_match(prediction: str | None, references: Sequence[str]) -> float:
    from untie.extraction_metrics import normalize_gold, normalize_pred

    predicted = normalize_pred(prediction).casefold()
    if not predicted:
        return 0.0
    return float(
        any(predicted == str(item).strip().casefold() for item in normalize_gold(list(references)))
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from experiments.keybert_qa_integration.attention_online import (
        extract_online_attention,
        keywords_from_cache,
        keywords_to_cache,
        load_attention_cache,
        save_attention_cache,
    )
    from experiments.keybert_qa_integration.config import (
        ExperimentParams,
        ModeSpec,
        ablation_mode_specs,
        all_mode_specs,
        core_mode_specs,
        default_params,
        extra_mode_specs,
        flags_to_dict,
        params_to_dict,
    )
    from experiments.keybert_qa_integration.contract import qa_contract
    from experiments.keybert_qa_integration.features import (
        chunk_question_keybert_scores,
        encode_keywords,
        extract_text_keyphrases,
        global_keyword_frequency,
        text_hash,
    )
    from experiments.keybert_qa_integration.hard_negatives import build_hard_negative_table
    from experiments.keybert_qa_integration.io_util import (
        append_timing,
        assert_isolated_output,
        write_csv,
        write_json,
    )
    from experiments.keybert_qa_integration.predict import DocumentFeatures, predict_document
    from experiments.keybert_qa_integration.report import (
        CORE_ORDER,
        aggregate_mode,
        bootstrap_paired_delta,
        delta_pct,
        markdown_table,
        write_summary,
    )
    from experiments.keybert_topic_clustering.documents import load_papers, split_groups
    from experiments.keybert_topic_clustering.downstream import collect_evidence
    from experiments.keybert_topic_clustering.encoder import CachedSentenceEncoder
    from experiments.keybert_topic_clustering.keyphrases import (
        KeyphraseConfig,
        extract_document_keyphrases,
    )
    from untie.attention import AttentionKeywordExtractor
    from untie.cli import load_static_keywords
    from untie.config import PipelineConfig
    from untie.extraction_metrics import char_f1
    from untie.keyword_evidence import EvidenceStore, ExtractionMetricCache
    from untie.models import ModelFactory, profile_for_language, resolve_device
    from untie.pipelines import DocumentProcessor

    output_dir = assert_isolated_output(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_dir / "timings.csv"
    versions = _versions()
    device = resolve_device(args.device)
    params = default_params(quick=args.quick)
    if args.keybert_weight is not None:
        params = replace(params, keybert_weight=float(args.keybert_weight))

    papers = load_papers(args.dataset, seed=args.seed, quick=args.quick)
    groups = split_groups(papers)
    train, dev, test = groups["train"], groups["dev"], groups["test"]
    question = args.question

    write_json(
        output_dir / "experiment_config.json",
        {
            "dataset": str(args.dataset),
            "output_dir": str(output_dir),
            "static_model": str(args.static_model),
            "evidence_dir": str(args.evidence_dir),
            "language": args.language,
            "seed": args.seed,
            "question": question,
            "quick": args.quick,
            "include_chunk_keybert": args.include_chunk_keybert,
            "run_training": args.run_training,
            "topic_encoder": args.topic_encoder,
            "params": params_to_dict(params),
            "qa_contract": qa_contract(),
            "anti_leakage": {
                "train": "KeyBERT phrases, online attention, global_frequency, hard negatives",
                "dev": "mix weights, topic variant, top-K, filter, cluster_k",
                "test": "frozen evaluation only; no gold for keywords, KeyBERT, or diff",
                "diff": "frozen global static dictionary, not per-document gold contrast",
            },
            "n_train": len(train),
            "n_dev": len(dev),
            "n_test": len(test),
            "versions": versions,
            "hardware": _hardware(device),
        },
    )

    encoder = CachedSentenceEncoder(
        args.topic_encoder,
        device=device,
        cache_dir=output_dir / "cache" / "topic_embeddings",
        normalize=True,
    )
    write_json(output_dir / "encoder_spec.json", encoder.spec())
    kp_config = KeyphraseConfig(
        ngram_low=params.keybert_ngram_low,
        ngram_high=params.keybert_ngram_high,
        top_n=params.keybert_extract_k,
        use_mmr=params.keybert_use_mmr,
        diversity=params.keybert_diversity,
    )

    started = time.perf_counter()
    phrases = extract_document_keyphrases(
        papers,
        encoder,
        kp_config,
        cache_dir=output_dir / "cache" / "keyphrases",
        representation=params.keybert_representation,
        extract_top_n=params.keybert_extract_k,
    )
    append_timing(
        timings_path,
        {
            "stage": "document_keybert",
            "seconds": time.perf_counter() - started,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        },
    )

    dest_store = EvidenceStore(output_dir / "evidence")
    reuse_store = (
        EvidenceStore(args.evidence_dir)
        if args.evidence_dir is not None and Path(args.evidence_dir).is_dir()
        else None
    )
    evidence: dict[str, Any] = {}
    missing = []
    for paper in papers:
        loaded = dest_store.load(paper.doc_id)
        if loaded is None and reuse_store is not None:
            loaded = reuse_store.load(paper.doc_id)
            if loaded is not None:
                dest_store.save(loaded)
        if loaded is None:
            missing.append(paper)
        else:
            evidence[paper.doc_id] = loaded

    models = None
    attention_extractor = None
    if missing:
        profile = profile_for_language(args.language)
        pipeline_config = PipelineConfig(profile=profile, device=args.device)
        models = ModelFactory(pipeline_config)
        processor = DocumentProcessor(
            models.tokenizer, pipeline_config, sentence_encoder=models.sentence_encoder
        )
        attention_extractor = AttentionKeywordExtractor(
            models.qa_model, models.tokenizer, models.device
        )
        started = time.perf_counter()
        extra = collect_evidence(
            missing,
            store=dest_store,
            processor=processor,
            answerer=models.answerer,
            encoder=models.sentence_encoder,
            attention=attention_extractor,
            language=args.language,
            collect_candidates_splits={"train"},
            question=question,
        )
        evidence.update(extra)
        append_timing(
            timings_path,
            {
                "stage": "collect_evidence",
                "seconds": time.perf_counter() - started,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
                "n_missing": len(missing),
            },
        )

    attention_path = output_dir / "cache" / "online_attention.json"
    attention_cache = load_attention_cache(attention_path)
    attention_keywords: dict[str, list] = {}
    attention_missing = [paper for paper in papers if paper.doc_id not in attention_cache]
    if attention_missing:
        if attention_extractor is None:
            profile = profile_for_language(args.language)
            pipeline_config = PipelineConfig(profile=profile, device=args.device)
            models = ModelFactory(pipeline_config)
            attention_extractor = AttentionKeywordExtractor(
                models.qa_model, models.tokenizer, models.device
            )
        started = time.perf_counter()
        for paper in attention_missing:
            keywords, indices = extract_online_attention(
                evidence[paper.doc_id],
                attention_extractor,
                question=question,
                top_chunks=params.attention_top_chunks,
                top_keywords=params.attention_top_keywords,
            )
            attention_cache[paper.doc_id] = keywords_to_cache(keywords, indices)
        save_attention_cache(attention_path, attention_cache)
        append_timing(
            timings_path,
            {
                "stage": "online_attention",
                "seconds": time.perf_counter() - started,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
                "n_docs": len(attention_missing),
            },
        )
    for paper in papers:
        attention_keywords[paper.doc_id] = keywords_from_cache(attention_cache[paper.doc_id])

    static_keywords, static_strategy = load_static_keywords(args.static_model, args.field_id)
    write_json(
        output_dir / "static_strategy.json",
        {"strategy": static_strategy, "n_keywords": len(static_keywords)},
    )
    global_frequency = global_keyword_frequency(
        attention_keywords, [paper.doc_id for paper in train]
    )
    write_json(output_dir / "global_keyword_frequency.json", global_frequency)

    features: dict[str, DocumentFeatures] = {}
    chunk_embeddings: dict[str, np.ndarray] = {}
    phrase_embeddings: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for paper in papers:
        item = evidence[paper.doc_id]
        doc_phrases = phrases.get(paper.doc_id, ())
        phrase_emb = (
            encoder.encode([entry.phrase for entry in doc_phrases])
            if doc_phrases
            else np.zeros((0, encoder.embedding_dimension), dtype=np.float32)
        )
        chunk_emb = (
            encoder.encode([chunk.text for chunk in item.chunks])
            if item.chunks
            else np.zeros((0, encoder.embedding_dimension), dtype=np.float32)
        )
        att = attention_keywords[paper.doc_id]
        att_emb = encode_keywords(att, encoder)
        chunk_embeddings[paper.doc_id] = chunk_emb
        phrase_embeddings[paper.doc_id] = phrase_emb
        features[paper.doc_id] = DocumentFeatures(
            evidence=item,
            phrases=doc_phrases,
            phrase_embeddings=phrase_emb,
            chunk_embeddings=chunk_emb,
            attention_keywords=tuple(att),
            attention_embeddings=att_emb,
        )
    append_timing(
        timings_path,
        {
            "stage": "encode_features",
            "seconds": time.perf_counter() - started,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        },
    )

    metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")

    def score_prediction(prediction: str | None, refs: tuple[str, ...]) -> dict[str, float]:
        metrics = metric_cache.score(
            prediction,
            refs,
            language=args.language,
            include_bertscore=not args.skip_bertscore,
        )
        metrics["exact_match"] = exact_match(prediction, refs)
        nonempty = bool(str(prediction or "").strip())
        metrics["coverage"] = float(nonempty)
        metrics["false_positive"] = float(nonempty and float(metrics.get("char_f1", 0.0)) <= 0.0)
        return metrics

    def mean_char_f1(
        split_papers: Sequence[Any],
        spec: ModeSpec,
        local_params: ExperimentParams,
    ) -> float:
        scores = []
        for paper in split_papers:
            prediction, _meta = predict_document(
                features[paper.doc_id],
                spec.flags,
                local_params,
                static_keywords=static_keywords,
                global_frequency=global_frequency,
                seed=args.seed,
            )
            scores.append(char_f1(prediction, list(paper.references)))
        return float(sum(scores) / max(len(scores), 1))

    att_kb = next(item for item in core_mode_specs() if item.name == "qa_attention_keybert")
    att_diff_kb = next(
        item for item in core_mode_specs() if item.name == "qa_attention_diff_keybert"
    )
    chosen_variant = params.keybert_topic_variant
    if args.keybert_weight is None:
        variant_rows = []
        best_variant_score = -1.0
        for variant in params.topic_variant_grid:
            trial = replace(params, keybert_topic_variant=variant)
            score = mean_char_f1(dev, att_kb, trial)
            variant_rows.append({"variant": variant, "dev_char_f1": score})
            if score > best_variant_score:
                best_variant_score = score
                chosen_variant = variant
        write_csv(output_dir / "dev_topic_variant.csv", variant_rows)
        params = replace(params, keybert_topic_variant=chosen_variant)

        def fit_weight(spec: ModeSpec) -> tuple[float, list[dict[str, float]]]:
            rows = []
            best_w, best_score = params.keybert_weight, -1.0
            for weight in params.keybert_weight_grid:
                trial = replace(params, keybert_weight=weight)
                score = mean_char_f1(dev, spec, trial)
                rows.append({"mode": spec.name, "keybert_weight": weight, "dev_char_f1": score})
                if score > best_score:
                    best_w, best_score = weight, score
            return best_w, rows

        w_att, rows_att = fit_weight(att_kb)
        w_full, rows_full = fit_weight(att_diff_kb)
        write_csv(output_dir / "dev_keybert_weights.csv", rows_att + rows_full)
        params = replace(params, keybert_weight=w_full)
        keybert_weight_att = w_att
        keybert_weight_full = w_full
    else:
        keybert_weight_att = params.keybert_weight
        keybert_weight_full = params.keybert_weight

    winning_mix = att_diff_kb
    winning_weight = keybert_weight_full
    score_att_kb = mean_char_f1(
        dev, att_kb, replace(params, keybert_weight=keybert_weight_att)
    )
    score_full_kb = mean_char_f1(
        dev, att_diff_kb, replace(params, keybert_weight=keybert_weight_full)
    )
    if score_att_kb > score_full_kb:
        winning_mix = att_kb
        winning_weight = keybert_weight_att

    topk_rows = []
    best_k, best_k_score = params.keybert_top_k, -1.0
    for top_k in params.top_k_grid:
        trial = replace(params, keybert_top_k=top_k, keybert_weight=winning_weight)
        score = mean_char_f1(dev, winning_mix, trial)
        topk_rows.append({"top_k": top_k, "dev_char_f1": score, "mode": winning_mix.name})
        if score > best_k_score:
            best_k, best_k_score = top_k, score
    write_csv(output_dir / "dev_topk.csv", topk_rows)
    params = replace(params, keybert_top_k=best_k, keybert_weight=winning_weight)

    filter_specs = {item.name: item for item in extra_mode_specs()}
    filter_rows = []
    best_filter, best_filter_score = "soft", -1.0
    for mode_name in ("qa_attention", "qa_attention_soft_filter", "qa_attention_hard_filter", "qa_attention_semantic_reweight"):
        spec = next(item for item in all_mode_specs() if item.name == mode_name)
        score = mean_char_f1(dev, spec, params)
        filter_rows.append({"mode": mode_name, "dev_char_f1": score})
        if mode_name != "qa_attention" and score > best_filter_score:
            best_filter, best_filter_score = spec.flags.filter_mode, score
    write_csv(output_dir / "dev_filters.csv", filter_rows)

    cluster_rows = []
    best_cluster_k, best_cluster_score = params.cluster_k, -1.0
    cluster_spec = filter_specs["qa_attention_keybert_clusters"]
    for cluster_k in params.cluster_k_grid:
        trial = replace(params, cluster_k=cluster_k)
        score = mean_char_f1(dev, cluster_spec, trial)
        cluster_rows.append({"cluster_k": cluster_k, "dev_char_f1": score})
        if score > best_cluster_score:
            best_cluster_k, best_cluster_score = cluster_k, score
    write_csv(output_dir / "dev_clusters.csv", cluster_rows)
    params = replace(params, cluster_k=best_cluster_k)

    chosen = {
        "keybert_topic_variant": params.keybert_topic_variant,
        "keybert_top_k": params.keybert_top_k,
        "keybert_weight": params.keybert_weight,
        "keybert_weight_att": keybert_weight_att,
        "keybert_weight_full": keybert_weight_full,
        "filter_mode": best_filter,
        "cluster_k": params.cluster_k,
        "winning_mix": winning_mix.name,
        "static_weight_ratio": params.static_weight_ratio,
        "keep_unmatched": params.keep_unmatched,
    }
    write_json(output_dir / "chosen_config.json", chosen)

    def evaluate_split(
        split_name: str,
        split_papers: Sequence[Any],
        specs: Sequence[ModeSpec],
        local_params: ExperimentParams,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in specs:
            mode_params = local_params
            if spec.name == "qa_attention_keybert":
                mode_params = replace(local_params, keybert_weight=keybert_weight_att)
            elif spec.name in {"qa_attention_diff_keybert", "qa_three_way_mix", "ablate_att_diff_keybert"}:
                mode_params = replace(local_params, keybert_weight=keybert_weight_full)
            _reset_gpu_peak()
            peak_rss = _rss_mb()
            for paper in split_papers:
                started_doc = time.perf_counter()
                prediction, meta = predict_document(
                    features[paper.doc_id],
                    spec.flags,
                    mode_params,
                    static_keywords=static_keywords,
                    global_frequency=global_frequency,
                    seed=args.seed,
                )
                latency = time.perf_counter() - started_doc
                metrics = score_prediction(prediction, paper.references)
                n_chunks = max(len(features[paper.doc_id].evidence.chunks), 1)
                peak_rss = max(peak_rss, _rss_mb())
                rows.append(
                    {
                        "document_id": paper.doc_id,
                        "split": split_name,
                        "mode": spec.name,
                        "display_name": spec.display_name,
                        "experiment": spec.experiment,
                        "core_table": spec.core_table,
                        "prediction": prediction or "",
                        **metrics,
                        "fallback": float(bool(meta.get("fallback"))),
                        "keyword_count": meta.get("keyword_count", 0),
                        "latency_s": latency,
                        "latency_per_chunk_s": latency / n_chunks,
                        "qa_chunks": n_chunks,
                        "rss_mb": _rss_mb(),
                        "gpu_mb": _gpu_mb(),
                        **{f"flag_{key}": value for key, value in flags_to_dict(spec.flags).items()},
                    }
                )
        return rows

    test_specs = list(core_mode_specs()) + list(extra_mode_specs()) + list(ablation_mode_specs())
    started = time.perf_counter()
    prediction_rows = evaluate_split("test", test, test_specs, params)
    append_timing(
        timings_path,
        {
            "stage": "evaluate_test",
            "seconds": time.perf_counter() - started,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        },
    )
    write_csv(output_dir / "test_predictions.csv", prediction_rows)
    metric_cache.save()

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_mode[str(row["mode"])].append(row)

    def table_from_modes(names: Sequence[str], baseline_char: float) -> tuple[list[dict[str, Any]], str]:
        aggregated = []
        md_rows = []
        for name in names:
            rows = by_mode.get(name, [])
            summary = aggregate_mode(
                rows, bootstrap_samples=params.bootstrap_samples, seed=args.seed
            )
            if not summary:
                continue
            spec = next((item for item in test_specs if item.name == name), None)
            summary.update(
                {
                    "mode": name,
                    "display_name": spec.display_name if spec else name,
                    "delta_char_f1_vs_baseline": summary["char_f1"] - baseline_char,
                    "delta_char_f1_pct": delta_pct(summary["char_f1"], baseline_char),
                }
            )
            aggregated.append(summary)
            md_rows.append(
                [
                    summary["display_name"],
                    f"{summary['char_f1']:.3f}",
                    f"{summary['token_f1']:.1f}",
                    f"{summary['rouge_l_f1']:.3f}",
                    f"{summary['bertscore_f1']:.3f}" if not args.skip_bertscore else "skipped",
                    f"{summary['coverage']:.3f}",
                    f"{summary['mean_latency_s']:.4f}",
                    f"{_mean_key(rows, 'gpu_mb'):.1f}",
                ]
            )
        header = [
            "Method",
            "Char F1",
            "Token F1",
            "ROUGE-L",
            "BERTScore F1",
            "Coverage",
            "Time",
            "GPU Memory",
        ]
        return aggregated, markdown_table(header, md_rows)

    def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        return float(sum(values) / max(len(values), 1))

    baseline_char = aggregate_mode(by_mode.get("qa_baseline", [])).get("char_f1", 0.0)
    core_agg, core_md = table_from_modes(CORE_ORDER, baseline_char)
    extra_names = [item.name for item in extra_mode_specs()]
    extra_agg, extra_md = table_from_modes(extra_names, baseline_char)
    ablation_names = [item.name for item in ablation_mode_specs()]
    ablation_agg, ablation_md = table_from_modes(ablation_names, baseline_char)
    write_csv(output_dir / "test_core_metrics.csv", core_agg)
    write_csv(output_dir / "test_extra_metrics.csv", extra_agg)
    write_csv(output_dir / "test_ablation_metrics.csv", ablation_agg)

    test_topk_rows = []
    for top_k in params.top_k_grid:
        trial = replace(params, keybert_top_k=top_k, keybert_weight=winning_weight)
        split_rows = evaluate_split("test", test, (winning_mix,), trial)
        for row in split_rows:
            row["top_k"] = top_k
            test_topk_rows.append(row)
    write_csv(output_dir / "test_topk_predictions.csv", test_topk_rows)
    topk_md_rows = []
    for top_k in params.top_k_grid:
        subset = [row for row in test_topk_rows if int(row["top_k"]) == top_k]
        summary = aggregate_mode(subset, bootstrap_samples=params.bootstrap_samples, seed=args.seed)
        topk_md_rows.append(
            [
                str(top_k),
                f"{summary.get('char_f1', 0):.3f}",
                f"{summary.get('mean_latency_s', 0):.4f}",
            ]
        )
    topk_md = markdown_table(["Top-K", "Char F1", "Time"], topk_md_rows)

    scope_md = "Chunk-level KeyBERT skipped (pass --include-chunk-keybert to compare)."
    if args.include_chunk_keybert:
        chunk_texts: dict[str, str] = {}
        chunk_order: dict[str, list[str]] = defaultdict(list)
        for paper in test:
            for chunk in evidence[paper.doc_id].chunks:
                key = f"{paper.doc_id}:{chunk.index}:{text_hash(chunk.text)[:12]}"
                chunk_texts[key] = chunk.text
                chunk_order[paper.doc_id].append(key)
        chunk_cfg = KeyphraseConfig(
            ngram_low=params.keybert_ngram_low,
            ngram_high=params.keybert_ngram_high,
            top_n=min(params.keybert_top_k, 10),
            use_mmr=params.keybert_use_mmr,
            diversity=params.keybert_diversity,
        )
        started = time.perf_counter()
        chunk_phrases, chunk_elapsed = extract_text_keyphrases(
            chunk_texts,
            encoder,
            chunk_cfg,
            cache_path=output_dir / "cache" / "chunk_keyphrases.json",
            extract_top_n=chunk_cfg.top_n,
        )
        append_timing(
            timings_path,
            {
                "stage": "chunk_keybert",
                "seconds": time.perf_counter() - started,
                "keybert_internal_s": chunk_elapsed,
                "rss_mb": _rss_mb(),
                "gpu_mb": _gpu_mb(),
            },
        )
        for paper in test:
            ordered = [
                chunk_phrases.get(key, ())
                for key in chunk_order[paper.doc_id]
            ]
            scores = chunk_question_keybert_scores(ordered, encoder, question)
            current = features[paper.doc_id]
            features[paper.doc_id] = replace(current, chunk_keybert_scores=scores)
        chunk_flags = replace(winning_mix.flags, keybert_scope="chunk")
        chunk_spec = replace(winning_mix, name=f"{winning_mix.name}_chunk", flags=chunk_flags)
        chunk_rows = evaluate_split("test", test, (chunk_spec,), params)
        write_csv(output_dir / "test_chunk_keybert.csv", chunk_rows)
        doc_summary = aggregate_mode(by_mode.get(winning_mix.name, []))
        chunk_summary = aggregate_mode(chunk_rows)
        scope_md = markdown_table(
            ["Scope", "Char F1", "Time"],
            [
                [
                    "document-level",
                    f"{doc_summary.get('char_f1', 0):.3f}",
                    f"{doc_summary.get('mean_latency_s', 0):.4f}",
                ],
                [
                    "chunk-level",
                    f"{chunk_summary.get('char_f1', 0):.3f}",
                    f"{chunk_summary.get('mean_latency_s', 0):.4f}",
                ],
            ],
        )

    paired = bootstrap_paired_delta(
        [float(row["char_f1"]) for row in by_mode.get("qa_attention_diff", [])],
        [float(row["char_f1"]) for row in by_mode.get("qa_attention_diff_keybert", [])],
        seed=args.seed,
        samples=params.bootstrap_samples,
    )
    write_json(output_dir / "bootstrap_att_diff_vs_keybert.json", paired)

    baseline_latency = aggregate_mode(by_mode.get("qa_baseline", [])).get("mean_latency_s", 0.0)
    cost_rows = []
    cost_md = []
    for name in CORE_ORDER:
        summary = next((item for item in core_agg if item["mode"] == name), None)
        if summary is None:
            continue
        time_delta = delta_pct(summary["mean_latency_s"], baseline_latency or 1e-9)
        cost_rows.append(
            {
                "mode": name,
                "mean_latency_s": summary["mean_latency_s"],
                "delta_time_pct": time_delta,
                "mean_gpu_mb": _mean_key(by_mode[name], "gpu_mb"),
                "mean_rss_mb": _mean_key(by_mode[name], "rss_mb"),
            }
        )
        cost_md.append(
            [
                summary["display_name"],
                f"{summary['mean_latency_s']:.4f}",
                f"{time_delta:.1f}%",
                f"{_mean_key(by_mode[name], 'gpu_mb'):.1f}",
                f"{_mean_key(by_mode[name], 'rss_mb'):.1f}",
            ]
        )
    write_csv(output_dir / "test_cost.csv", cost_rows)

    hard_rows = build_hard_negative_table(
        train,
        evidence,
        phrases,
        chunk_embeddings,
        phrase_embeddings,
        params,
    )
    write_csv(output_dir / "train_hard_negatives.csv", hard_rows)
    alignment_rows = []
    for paper in train:
        refs = [item for item in paper.references if str(item).strip()]
        if not refs or phrase_embeddings[paper.doc_id].size == 0:
            continue
        gold_vec = encoder.encode(refs)
        sims = phrase_embeddings[paper.doc_id] @ gold_vec.T
        peak = float(sims.max()) if sims.size else 0.0
        relevant = sims.max(axis=1) >= params.keybert_sim_threshold if sims.size else np.zeros((0,), dtype=bool)
        mean_rel = float(sims.max(axis=1)[relevant].mean()) if np.any(relevant) else 0.0
        alignment_rows.append(
            {
                "document_id": paper.doc_id,
                "max_sim_to_gold": peak,
                "n_relevant_phrases": int(np.sum(relevant)),
                "alignment_loss_proxy": 1.0 - mean_rel if np.any(relevant) else None,
            }
        )
    write_csv(output_dir / "train_alignment_proxy.csv", alignment_rows)
    training_status = (
        "requested via --run-training, but this repository has no RoBERTa QA trainer; "
        "hard-negative CSV and alignment proxy were written instead"
        if args.run_training
        else "skipped (default). Hard-negative CSV and alignment proxy written for later training."
    )
    write_json(
        output_dir / "training_skip.json",
        {
            "run_training": args.run_training,
            "hard_negatives": str(output_dir / "train_hard_negatives.csv"),
            "alignment_proxy": str(output_dir / "train_alignment_proxy.csv"),
            "reason": training_status,
            "alignment_lambdas": list(params.alignment_lambdas),
            "hard_negative_ratios": list(params.hard_negative_ratios),
        },
    )

    ranked = sorted(core_agg, key=lambda item: -float(item["char_f1"]))
    best = ranked[0] if ranked else {"display_name": "n/a", "char_f1": 0.0, "mode": ""}
    att_row = next((item for item in core_agg if item["mode"] == "qa_attention"), {})
    att_kb_row = next((item for item in core_agg if item["mode"] == "qa_attention_keybert"), {})
    full_row = next((item for item in core_agg if item["mode"] == "qa_attention_diff"), {})
    full_kb_row = next((item for item in core_agg if item["mode"] == "qa_attention_diff_keybert"), {})
    kb_only = next((item for item in core_agg if item["mode"] == "qa_keybert"), {})
    helped = float(att_kb_row.get("char_f1", 0)) > float(att_row.get("char_f1", 0)) or float(
        full_kb_row.get("char_f1", 0)
    ) > float(full_row.get("char_f1", 0))
    redundant = []
    if float(kb_only.get("char_f1", 0)) < float(att_row.get("char_f1", 0)):
        redundant.append("KeyBERT-only (control, weaker than attention)")
    if float(full_kb_row.get("char_f1", 0)) <= float(full_row.get("char_f1", 0)):
        redundant.append("KeyBERT on top of attention+diff")
    if float(att_kb_row.get("char_f1", 0)) <= float(att_row.get("char_f1", 0)):
        redundant.append("KeyBERT on top of per-doc attention")
    cheapest_ok = min(
        (item for item in core_agg if item["mode"] != "qa_keybert"),
        key=lambda item: ( -float(item["char_f1"]), float(item["mean_latency_s"])),
        default=best,
    )
    conclusions = {
        "q1": (
            f"Yes: KeyBERT improved char-F1 versus the matching attention system "
            f"(att {att_row.get('char_f1', 0):.3f} → att+KB {att_kb_row.get('char_f1', 0):.3f}; "
            f"att+diff {full_row.get('char_f1', 0):.3f} → att+diff+KB {full_kb_row.get('char_f1', 0):.3f})."
            if helped
            else f"No: KeyBERT did not beat the matching attention system on test char-F1 "
            f"(att {att_row.get('char_f1', 0):.3f} vs att+KB {att_kb_row.get('char_f1', 0):.3f}; "
            f"att+diff {full_row.get('char_f1', 0):.3f} vs att+diff+KB {full_kb_row.get('char_f1', 0):.3f})."
        ),
        "q2": f"{best.get('display_name')} (char-F1 {best.get('char_f1', 0):.3f})",
        "q3": ", ".join(redundant) if redundant else "no component looked strictly redundant on char-F1",
        "q4": f"{cheapest_ok.get('display_name')} (char-F1 {cheapest_ok.get('char_f1', 0):.3f}, {cheapest_ok.get('mean_latency_s', 0):.4f}s/doc)",
        "q5": (
            "Use KeyBERT offline (and at inference as a cheap MiniLM prior) if it won; "
            "do not replace attention. Training-time hard negatives remain unused until a QA trainer exists."
            if helped
            else "Keep KeyBERT offline-only for analysis; do not add it to the inference mix unless a later split reverses this result. Do not replace attention."
        ),
    }
    write_json(output_dir / "conclusions.json", conclusions)
    write_summary(
        output_dir / "summary.md",
        {
            "n_documents": len(papers),
            "n_train": len(train),
            "n_dev": len(dev),
            "n_test": len(test),
            "question": question,
            "output_dir": str(output_dir),
            "evidence_dir": str(args.evidence_dir),
            "static_model": str(args.static_model),
            "keybert_version": versions.get("keybert"),
            "st_version": versions.get("sentence_transformers"),
            "sklearn_version": versions.get("sklearn"),
            "topic_encoder": args.topic_encoder,
            "qa": "untie.models.ExtractiveQuestionAnswerer",
            "hardware": _hardware(device),
            "chosen": chosen,
            "training_status": training_status,
            "core_table_md": core_md,
            "extra_table_md": extra_md,
            "ablation_table_md": ablation_md,
            "topk_table_md": topk_md,
            "scope_table_md": scope_md,
            "cost_table_md": markdown_table(
                ["Method", "Time/doc", "Δtime %", "GPU MB", "RAM MB"], cost_md
            ),
            "cost": {"peak_rss_mb": _rss_mb(), "peak_gpu_mb": _gpu_mb()},
            "paired": paired,
            "conclusions": conclusions,
        },
    )
    print(f"Wrote KeyBERT QA integration artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
