"""Isolated evaluation of question-conditioned answer aggregation.

Does not modify default UnTIE behaviour (merge/filter default off).
Does not write into hierarchical / KeyBERT artifact trees.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge near-duplicate QA spans and filter by similarity to the question"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/keybert_topic_clustering/en_full/evidence",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Which task was solved?")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--min-keep", type=int, default=3)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    from experiments.answer_aggregation.io_util import (
        append_timing,
        assert_isolated_output,
        write_csv,
        write_json,
    )
    from experiments.keybert_topic_clustering.documents import load_papers, split_groups
    from experiments.keybert_topic_clustering.downstream import (
        _answers_from_evidence,
        bootstrap_mean_ci,
    )
    from untie.answer_selection import (
        AggregationSettings,
        query_prototypes,
        select_max_query_similarity,
        select_typical_span,
    )
    from untie.config import PipelineConfig
    from untie.extraction_metrics import char_f1, normalize_pred
    from untie.keyword_evidence import EvidenceStore, ExtractionMetricCache
    from untie.models import ModelFactory, profile_for_language, resolve_device

    output_dir = assert_isolated_output(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    timings_path = output_dir / "timings.csv"
    device = resolve_device(args.device)
    papers = load_papers(args.dataset, seed=args.seed, quick=args.quick)
    groups = split_groups(papers)
    _train, dev, test = groups["train"], groups["dev"], groups["test"]
    question = args.question
    quantile_grid = (0.25, 0.4, 0.5, 0.65) if args.quick else (0.2, 0.35, 0.5, 0.65, 0.8)

    write_json(
        output_dir / "experiment_config.json",
        {
            "dataset": str(args.dataset),
            "output_dir": str(output_dir),
            "evidence_dir": str(args.evidence_dir),
            "language": args.language,
            "question": question,
            "query_prototypes": list(query_prototypes(question, language=args.language)),
            "anti_lexicon": "no field-specific span word list; filter uses the QA question only",
            "quantile_grid": list(quantile_grid),
            "min_keep": args.min_keep,
            "n_train": len(_train),
            "n_dev": len(dev),
            "n_test": len(test),
            "quick": args.quick,
            "device": device,
        },
    )

    dest = EvidenceStore(output_dir / "evidence")
    reuse = (
        EvidenceStore(args.evidence_dir)
        if args.evidence_dir is not None and Path(args.evidence_dir).is_dir()
        else None
    )
    evidence = {}
    missing = []
    for paper in papers:
        loaded = dest.load(paper.doc_id)
        if loaded is None and reuse is not None:
            loaded = reuse.load(paper.doc_id)
            if loaded is not None:
                dest.save(loaded)
        if loaded is None:
            missing.append(paper.doc_id)
        else:
            evidence[paper.doc_id] = loaded
    if missing:
        raise SystemExit(
            "missing cached QA evidence for "
            f"{len(missing)} documents; pass --evidence-dir to a completed run "
            "(hyp2/hyp3). This experiment does not re-run RoBERTa."
        )

    profile = profile_for_language(args.language)
    models = ModelFactory(PipelineConfig(profile=profile, device=args.device))
    encoder = models.sentence_encoder

    def predict(
        paper,
        *,
        merge: bool,
        quantile: float | None,
    ) -> tuple[str | None, dict[str, Any]]:
        item = evidence[paper.doc_id]
        answers, _chunks = _answers_from_evidence(item)
        selected, supporting, stats = select_typical_span(
            answers,
            encoder,
            question=question,
            settings=AggregationSettings(
                merge_near_duplicates=merge,
                query_quantile=quantile,
                min_keep=args.min_keep,
                language=args.language,
            ),
        )
        text = None if selected is None else selected.text
        return text, {
            "n_input": stats.n_input,
            "n_after_merge": stats.n_after_merge,
            "n_after_filter": stats.n_after_filter,
            "query_threshold": stats.query_threshold,
            "fallback": stats.fallback,
            "selected_query_similarity": stats.selected_query_similarity,
            "n_supporting": len(supporting),
            "prototypes": list(stats.prototypes),
        }

    def mean_dev_char_f1(merge: bool, quantile: float | None) -> float:
        scores = []
        for paper in dev:
            pred, _meta = predict(paper, merge=merge, quantile=quantile)
            scores.append(char_f1(pred, list(paper.references)))
        return float(sum(scores) / max(len(scores), 1))

    started = time.perf_counter()
    grid_rows = []
    for merge in (False, True):
        for quantile in (None, *quantile_grid):
            score = mean_dev_char_f1(merge, quantile)
            grid_rows.append(
                {
                    "merge_near_duplicates": merge,
                    "query_quantile": quantile if quantile is not None else "",
                    "dev_char_f1": score,
                }
            )
    write_csv(output_dir / "dev_grid.csv", grid_rows)
    append_timing(
        timings_path,
        {
            "stage": "dev_grid",
            "seconds": time.perf_counter() - started,
            "rss_mb": _rss_mb(),
            "gpu_mb": _gpu_mb(),
        },
    )
    best = max(grid_rows, key=lambda row: float(row["dev_char_f1"]))
    chosen_merge = bool(best["merge_near_duplicates"])
    chosen_quantile = (
        None if best["query_quantile"] == "" else float(best["query_quantile"])
    )
    # Target recipe is (1)+(3): merge AND question filter, even if another cell wins.
    target_cells = [
        row
        for row in grid_rows
        if row["merge_near_duplicates"] and row["query_quantile"] != ""
    ]
    target = max(target_cells, key=lambda row: float(row["dev_char_f1"]))
    target_quantile = float(target["query_quantile"])
    write_json(
        output_dir / "chosen_config.json",
        {
            "best_dev_cell": best,
            "target_merge_and_filter": {
                "merge_near_duplicates": True,
                "query_quantile": target_quantile,
                "dev_char_f1": target["dev_char_f1"],
                "min_keep": args.min_keep,
            },
            "note": "Test reports the target (1)+(3) plus ablations; best_dev_cell is the unrestricted grid winner.",
        },
    )

    metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")

    def exact_match(prediction: str | None, refs: Sequence[str]) -> float:
        predicted = normalize_pred(prediction).casefold()
        if not predicted:
            return 0.0
        return float(any(predicted == str(item).strip().casefold() for item in refs if str(item).strip()))

    def score_pred(prediction: str | None, refs: tuple[str, ...]) -> dict[str, float]:
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

    modes = (
        ("q0_cached", "cached", False, None),
        ("q0_recomputed", "recomputed_legacy", False, None),
        ("merge_only", "merge", True, None),
        ("query_filter_only", "filter", False, target_quantile),
        ("merge_and_query_filter", "merge+filter", True, target_quantile),
        ("query_argmax", "query_argmax", False, None),
    )

    prediction_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for paper in test:
        item = evidence[paper.doc_id]
        answers, _chunks = _answers_from_evidence(item)
        for mode, _label, merge, quantile in modes:
            started_doc = time.perf_counter()
            if mode == "q0_cached":
                pred = item.baseline_answer
                meta = {"n_input": len(item.chunks), "fallback": "cached"}
            elif mode == "query_argmax":
                selected, supporting, stats = select_max_query_similarity(
                    answers,
                    encoder,
                    question=question,
                    settings=AggregationSettings(
                        merge_near_duplicates=True,
                        language=args.language,
                    ),
                )
                pred = None if selected is None else selected.text
                meta = {
                    "n_input": stats.n_input,
                    "n_after_merge": stats.n_after_merge,
                    "n_after_filter": stats.n_after_filter,
                    "selected_query_similarity": stats.selected_query_similarity,
                    "fallback": stats.fallback,
                }
            else:
                pred, meta = predict(paper, merge=merge, quantile=quantile)
            metrics = score_pred(pred, paper.references)
            prediction_rows.append(
                {
                    "document_id": paper.doc_id,
                    "split": "test",
                    "mode": mode,
                    "prediction": pred or "",
                    **metrics,
                    **{key: meta.get(key, "") for key in (
                        "n_input",
                        "n_after_merge",
                        "n_after_filter",
                        "query_threshold",
                        "fallback",
                        "selected_query_similarity",
                    )},
                    "latency_s": time.perf_counter() - started_doc,
                    "matches_cached_q0": float(
                        (pred or "") == (item.baseline_answer or "")
                    ),
                }
            )
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

    def summarize(name: str) -> dict[str, Any]:
        rows = by_mode[name]
        char_values = [float(item["char_f1"]) for item in rows]
        mean, lo, hi = bootstrap_mean_ci(char_values, seed=args.seed)
        baseline = [float(item["char_f1"]) for item in by_mode["q0_cached"]]
        base_by = {item["document_id"]: item for item in by_mode["q0_cached"]}
        win = tie = lose = 0
        for item in rows:
            da = float(base_by[item["document_id"]]["char_f1"])
            db = float(item["char_f1"])
            if db > da + 1e-12:
                win += 1
            elif da > db + 1e-12:
                lose += 1
            else:
                tie += 1
        return {
            "mode": name,
            "n": len(rows),
            "char_f1": float(np.mean(char_values)),
            "token_f1": float(np.mean([float(item["token_f1"]) for item in rows])),
            "rouge_l_f1": float(np.mean([float(item["rouge_l_f1"]) for item in rows])),
            "bertscore_f1": float(np.mean([float(item.get("bertscore_f1") or 0.0) for item in rows])),
            "exact_match": float(np.mean([float(item["exact_match"]) for item in rows])),
            "char_f1_ci95_low": lo,
            "char_f1_ci95_high": hi,
            "delta_vs_q0": float(np.mean(char_values) - np.mean(baseline)),
            "win": win,
            "tie": tie,
            "lose": lose,
            "match_cached_q0_rate": float(np.mean([float(item["matches_cached_q0"]) for item in rows])),
            "mean_n_after_filter": float(
                np.mean([float(item["n_after_filter"] or 0) for item in rows])
            ),
        }

    summaries = [summarize(name) for name, *_rest in modes]
    write_csv(output_dir / "test_metrics.csv", summaries)

    q0 = next(item for item in summaries if item["mode"] == "q0_cached")
    chosen = next(item for item in summaries if item["mode"] == "merge_and_query_filter")
    lines = [
        "# Question-conditioned answer aggregation",
        "",
        f"- test n={len(test)}, question: {question}",
        f"- prototypes from the question: {list(query_prototypes(question, language=args.language))}",
        "- no task/dataset/method span lexicon",
        f"- target (merge + query quantile={target_quantile}, min_keep={args.min_keep})",
        f"- Q0 cached char-F1 {q0['char_f1']:.3f}",
        f"- merge+filter char-F1 {chosen['char_f1']:.3f} (Δ {chosen['delta_vs_q0']:+.3f}, W/T/L {chosen['win']}/{chosen['tie']}/{chosen['lose']})",
        "",
        "| Mode | Char F1 | Δ vs Q0 | W/T/L | EM |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['mode']} | {item['char_f1']:.3f} | {item['delta_vs_q0']:+.3f} | "
            f"{item['win']}/{item['tie']}/{item['lose']} | {item['exact_match']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "test_metrics.json", summaries)
    print(f"Wrote answer-aggregation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
