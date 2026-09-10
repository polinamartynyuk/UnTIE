"""Isolated CE shortlist with topic-conditioned query and AnswerAggregator.

Does not overwrite previous evaluations. Shortlist k is chosen on the dev
split using gold char-F1, then frozen for test. Test ranking never reads gold.
Cluster SFFS dictionaries are not used.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.ce_topic_shortlist import (
    DEFAULT_MAX_TOPIC_TERMS,
    SHORTLIST_K_GRID,
    collect_topic_ce_panel,
    fit_shortlist_k,
    k_label,
    predict_from_shortlist,
    predict_with_aggregator,
    ranked_topic_terms,
)
from untie.config import PipelineConfig
from untie.cross_encoder_ranking import SentenceTransformerCrossEncoder
from untie.keyword_evidence import ExtractionMetricCache
from untie.keyword_tuning import deterministic_document_split
from untie.models import ModelFactory, profile_for_language, resolve_device
from untie.pipelines import DocumentProcessor
from untie.topics import (
    TopicDocument,
    compute_qa_ratios,
    load_topic_artifact,
    route_document_to_nearest_leaf,
)


METRIC_NAMES = (
    "char_f1",
    "token_f1",
    "rouge_l_f1",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "composite_quality",
)
PREDICTION_FIELDS = (
    "doc_id",
    "mode",
    "prediction",
    "char_f1",
    "token_f1",
    "rouge_l_f1",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "composite_quality",
    "qa_chunks",
    "qa_chunk_ratio",
    "qa_token_ratio",
    "fallback",
    "fallback_reason",
    "fallback_stage",
    "topic_keyword_node_id",
    "topic_keyword_ancestor_fallback",
    "latency_s",
)
NEW_MODES = (
    "aggregator-all",
    "ce-title-shortlist",
    "ce-topic-query-shortlist",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CE shortlists with a topic-conditioned query and "
            "baseline AnswerAggregator"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--topic-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-predictions",
        type=Path,
        help="Previous test_predictions_all_metrics.csv; other modes are reused",
    )
    parser.add_argument(
        "--ce-model",
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Which task was solved?")
    parser.add_argument("--chunk-max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument(
        "--max-topic-terms",
        type=int,
        default=DEFAULT_MAX_TOPIC_TERMS,
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files in --output-dir only; never source evaluations",
    )
    return parser


def _load_payload(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("documents")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("dataset must contain a JSON array or JSONL objects")
    return payload


def _references(raw: dict) -> tuple[str, ...]:
    value = raw.get("tasks_cleaned", raw.get("tasks", ()))
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = (value,)
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(
        str(item).replace("_", " ").strip()
        for item in values
        if item is not None and str(item).strip()
    )


def load_split_documents(
    path: Path, seed: int, *, quick: bool = False
) -> tuple[list[TopicDocument], list[TopicDocument], dict[str, tuple[str, ...]]]:
    payload = _load_payload(path)
    ids = [str(item.get("doc_id", item.get("id", ""))) for item in payload]
    if any(not doc_id for doc_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset document ids must be non-empty and unique")
    explicit = [item.get("split") for item in payload]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("split must be present on every record or none")
        split_by_id = {
            doc_id: str(value).casefold() for doc_id, value in zip(ids, explicit)
        }
        invalid = sorted(set(split_by_id.values()).difference({"train", "dev", "test"}))
        if invalid:
            raise ValueError(f"unsupported dataset splits: {invalid}")
    else:
        split = deterministic_document_split(ids, seed=seed)
        split_by_id = {
            doc_id: split_name
            for split_name, split_ids in split.items()
            for doc_id in split_ids
        }

    def convert(raw: dict, split_name: str) -> TopicDocument:
        headings = raw.get("headings", ())
        if isinstance(headings, str):
            headings = (headings,)
        return TopicDocument(
            doc_id=str(raw.get("doc_id", raw.get("id"))),
            text=str(raw.get("original_text", raw.get("text", "")) or ""),
            title=str(raw.get("title", "") or ""),
            abstract=str(raw.get("abstract", "") or ""),
            headings=tuple(str(item) for item in headings),
            split=split_name,
        )

    documents = [
        convert(raw, split_by_id[str(raw.get("doc_id", raw.get("id")))])
        for raw in payload
    ]
    dev = [item for item in documents if item.split == "dev"]
    test = [item for item in documents if item.split == "test"]
    if quick:
        dev = dev[:8]
        test = test[:8]
    references = {
        str(raw.get("doc_id", raw.get("id"))): _references(raw) for raw in payload
    }
    if not test:
        raise ValueError("evaluation requires a non-empty test partition")
    if not dev:
        raise ValueError("shortlist-k fitting requires a non-empty dev partition")
    missing = [
        item.doc_id
        for item in [*dev, *test]
        if not references.get(item.doc_id)
    ]
    if missing:
        raise ValueError(f"documents require task references: {missing[:5]}")
    return dev, test, references


def composite_quality(metrics: dict[str, float]) -> float:
    return (
        float(metrics["char_f1"])
        + float(metrics["token_f1"]) / 100.0
        + float(metrics["rouge_l_f1"])
        + float(metrics.get("bertscore_f1", 0.0))
    ) / 4.0


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protected_evaluation_dirs() -> set[Path]:
    root = PROJECT_ROOT / "artifacts" / "hierarchical_topic_optimized" / "en"
    return {
        root,
        root / "evaluation",
        root / "refit" / "evaluation",
        root / "refit_no_gold_contrast" / "evaluation",
        root / "soft_rerank" / "evaluation",
        root / "cross_encoder" / "evaluation",
        root / "ce_topic_query" / "evaluation",
    }


def load_reused_rows(
    path: Path, test_ids: set[str], *, exclude_modes: set[str]
) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    kept = []
    for row in rows:
        if row.get("doc_id") not in test_ids:
            continue
        copied = dict(row)
        mode = copied.get("mode")
        if mode == "hierarchical":
            copied["mode"] = "hierarchical-tuned"
            mode = copied["mode"]
        if mode in exclude_modes:
            continue
        kept.append(copied)
    return kept


def summarize_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    baseline = {str(row["doc_id"]): row for row in by_mode.get("baseline", ())}
    summaries = []
    for mode, group in by_mode.items():
        count = len(group)
        means = {}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group]
            means[f"mean_{metric}"] = sum(values) / count if count else 0.0
            if mode == "baseline" or not baseline:
                means[f"mean_delta_{metric}"] = 0.0
                continue
            deltas = [
                float(row[metric]) - float(baseline[str(row["doc_id"])][metric])
                for row in group
                if str(row["doc_id"]) in baseline
            ]
            means[f"mean_delta_{metric}"] = sum(deltas) / len(deltas) if deltas else 0.0
        char_deltas = []
        if mode != "baseline" and baseline:
            char_deltas = [
                float(row["char_f1"]) - float(baseline[str(row["doc_id"])]["char_f1"])
                for row in group
                if str(row["doc_id"]) in baseline
            ]
        summaries.append(
            {
                "mode": mode,
                "documents": count,
                **means,
                "win_rate": (
                    sum(delta > 0 for delta in char_deltas) / len(char_deltas)
                    if char_deltas
                    else 0.0
                ),
                "harm_rate": (
                    sum(delta < 0 for delta in char_deltas) / len(char_deltas)
                    if char_deltas
                    else 0.0
                ),
                "fallback_rate": sum(
                    str(row.get("fallback", "")).lower() in {"true", "1"}
                    for row in group
                )
                / count
                if count
                else 0.0,
                "mean_qa_chunk_ratio": sum(float(row["qa_chunk_ratio"]) for row in group)
                / count
                if count
                else 0.0,
                "mean_qa_token_ratio": sum(float(row["qa_token_ratio"]) for row in group)
                / count
                if count
                else 0.0,
                "mean_latency_s": sum(float(row["latency_s"]) for row in group) / count
                if count
                else 0.0,
            }
        )
    order = {
        "baseline": 0,
        "static-keywords": 1,
        "hierarchical-tuned": 2,
        "hierarchical-refit": 3,
        "hierarchical-no-gold-contrast": 4,
        "hierarchical-soft-rerank": 5,
        "cross-encoder": 6,
        "cross-encoder-static-mix": 7,
        "aggregator-all": 8,
        "ce-title-shortlist": 9,
        "ce-topic-query-shortlist": 10,
    }
    summaries.sort(key=lambda row: (order.get(str(row["mode"]), 20), str(row["mode"])))
    return summaries


def _row(
    *,
    document: TopicDocument,
    mode: str,
    prediction: str | None,
    metrics: dict[str, float],
    compute: dict[str, float | int],
    latency: float,
    leaf_id: str = "",
) -> dict[str, object]:
    return {
        "doc_id": document.doc_id,
        "mode": mode,
        "prediction": prediction or "",
        **metrics,
        "composite_quality": composite_quality(metrics),
        "qa_chunks": compute.get("qa_chunks", 0),
        "qa_chunk_ratio": compute.get("qa_chunk_ratio", 1.0),
        "qa_token_ratio": compute.get("qa_token_ratio", 1.0),
        "fallback": False,
        "fallback_reason": "",
        "fallback_stage": "",
        "topic_keyword_node_id": leaf_id,
        "topic_keyword_ancestor_fallback": False,
        "latency_s": latency,
    }


def _leaf_terms(
    document: TopicDocument, encoder, artifact, max_terms: int
) -> tuple[str, tuple[str, ...]]:
    route = route_document_to_nearest_leaf(
        document,
        encoder,
        artifact,
        expected_encoder_revision=artifact.encoder_revision or None,
    )
    if not route.topics:
        return "", ()
    leaf_id = route.topics[0].node_id
    node = artifact.node_map[leaf_id]
    return leaf_id, ranked_topic_terms(
        node.representative_terms, max_terms=max_terms
    )


def _collect_panel(
    document: TopicDocument,
    question: str,
    *,
    processor,
    answerer,
    reranker,
    topic_encoder,
    artifact,
    max_topic_terms: int,
    references: tuple[str, ...] = (),
):
    leaf_id, terms = _leaf_terms(
        document, topic_encoder, artifact, max_topic_terms
    )
    return collect_topic_ce_panel(
        document,
        question,
        processor=processor,
        answerer=answerer,
        reranker=reranker,
        leaf_terms=terms,
        leaf_id=leaf_id,
        references=references,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    if output_dir in protected_evaluation_dirs():
        raise ValueError(
            "topic-CE evaluation refuses to overwrite a previous experiment "
            "directory; choose --output-dir under a new folder"
        )
    existing = [
        output_dir / "test_predictions_all_metrics.csv",
        output_dir / "quality_summary_all_metrics.json",
    ]
    if any(path.exists() for path in existing) and not args.force:
        raise ValueError(
            "evaluation outputs already exist; pass --force to replace files "
            f"in {output_dir} only"
        )
    if args.max_topic_terms < 1:
        raise ValueError("--max-topic-terms must be at least 1")
    dev_documents, test_documents, references = load_split_documents(
        args.dataset, args.seed, quick=args.quick
    )
    test_ids = {item.doc_id for item in test_documents}
    reused = []
    if args.reuse_predictions is not None:
        reused = load_reused_rows(
            args.reuse_predictions,
            test_ids,
            exclude_modes=set(NEW_MODES),
        )
        reused_ids = {
            str(row["doc_id"]) for row in reused if row["mode"] == "baseline"
        }
        if reused_ids != test_ids:
            missing = sorted(test_ids - reused_ids)[:5]
            raise ValueError(
                "reuse predictions do not cover the test split; "
                f"missing ids: {missing}"
            )

    profile = profile_for_language(args.language)
    config = PipelineConfig(
        profile=profile,
        chunk_max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.overlap_tokens,
        device=args.device,
    )
    models = ModelFactory(config)
    processor = DocumentProcessor(
        models.tokenizer,
        config,
        sentence_encoder=models.sentence_encoder,
    )
    artifact = load_topic_artifact(args.topic_model)
    reranker = SentenceTransformerCrossEncoder(
        args.ce_model, resolve_device(args.device)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")
    include_bertscore = not args.skip_bertscore

    def score_char_f1(prediction: str | None, refs: list[str] | tuple[str, ...]) -> float:
        return float(
            metric_cache.score(
                prediction,
                refs,
                language=args.language,
                include_bertscore=False,
            )["char_f1"]
        )

    print(f"Collecting dev panels {len(dev_documents)}", flush=True)
    dev_panels = []
    for index, document in enumerate(dev_documents, start=1):
        dev_panels.append(
            _collect_panel(
                document,
                args.question,
                processor=processor,
                answerer=models.answerer,
                reranker=reranker,
                topic_encoder=models.sentence_encoder,
                artifact=artifact,
                max_topic_terms=args.max_topic_terms,
                references=references[document.doc_id],
            )
        )
        if index % 5 == 0 or index == len(dev_documents):
            print(f"Dev panels {index}/{len(dev_documents)}", flush=True)

    title_selection = fit_shortlist_k(
        dev_panels,
        models.sentence_encoder,
        score_char_f1,
        lambda panel: panel.title_ce_scores,
        grid=SHORTLIST_K_GRID,
    )
    topic_selection = fit_shortlist_k(
        dev_panels,
        models.sentence_encoder,
        score_char_f1,
        lambda panel: panel.topic_ce_scores,
        grid=SHORTLIST_K_GRID,
    )
    print(
        f"Selected title k={title_selection.label} "
        f"dev_char_f1={title_selection.mean_char_f1:.4f}; "
        f"topic k={topic_selection.label} "
        f"dev_char_f1={topic_selection.mean_char_f1:.4f}",
        flush=True,
    )

    new_rows = []
    routed_leaves = 0
    for index, document in enumerate(test_documents, start=1):
        started = time.perf_counter()
        panel = _collect_panel(
            document,
            args.question,
            processor=processor,
            answerer=models.answerer,
            reranker=reranker,
            topic_encoder=models.sentence_encoder,
            artifact=artifact,
            max_topic_terms=args.max_topic_terms,
            references=(),
        )
        if panel.leaf_id:
            routed_leaves += 1
        compute = compute_qa_ratios(panel.chunks, panel.chunks)
        latency = time.perf_counter() - started
        predictions = (
            (
                "aggregator-all",
                predict_with_aggregator(panel.answers, models.sentence_encoder),
            ),
            (
                "ce-title-shortlist",
                predict_from_shortlist(
                    panel,
                    panel.title_ce_scores,
                    models.sentence_encoder,
                    title_selection.k,
                ),
            ),
            (
                "ce-topic-query-shortlist",
                predict_from_shortlist(
                    panel,
                    panel.topic_ce_scores,
                    models.sentence_encoder,
                    topic_selection.k,
                ),
            ),
        )
        for mode, prediction in predictions:
            metrics = metric_cache.score(
                prediction,
                references[document.doc_id],
                language=args.language,
                include_bertscore=include_bertscore,
            )
            if not include_bertscore:
                metrics.setdefault("bertscore_p", 0.0)
                metrics.setdefault("bertscore_r", 0.0)
                metrics.setdefault("bertscore_f1", 0.0)
            new_rows.append(
                _row(
                    document=document,
                    mode=mode,
                    prediction=prediction,
                    metrics=metrics,
                    compute=compute,
                    latency=latency,
                    leaf_id=panel.leaf_id,
                )
            )
        if index % 5 == 0 or index == len(test_documents):
            print(f"Evaluated {index}/{len(test_documents)}", flush=True)
            metric_cache.save()

    all_rows = [*reused, *new_rows]
    summaries = summarize_metric_rows(all_rows)
    _write_csv(output_dir / "test_predictions_all_metrics.csv", all_rows, PREDICTION_FIELDS)
    (output_dir / "quality_summary_all_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "output_dir": str(output_dir),
        "ce_model": args.ce_model,
        "topic_model": str(args.topic_model),
        "max_topic_terms": args.max_topic_terms,
        "ranking_query_title": "identity+question (body fallback if title/abstract empty)",
        "ranking_query_topic": "identity+question+leaf representative_terms",
        "ranking_query_body_fallback": True,
        "selector": "AnswerAggregator",
        "uses_cluster_sffs_keywords": False,
        "keep_all_chunks_for_qa": True,
        "documents_dev": len(dev_documents),
        "documents_test": len(test_documents),
        "test_routed_leaves": routed_leaves,
        "title_shortlist_k": k_label(title_selection.k),
        "title_dev_mean_char_f1": title_selection.mean_char_f1,
        "title_k_grid": [
            {"k": trial.label, "mean_char_f1": trial.mean_char_f1}
            for trial in title_selection.trials
        ],
        "topic_shortlist_k": k_label(topic_selection.k),
        "topic_dev_mean_char_f1": topic_selection.mean_char_f1,
        "topic_k_grid": [
            {"k": trial.label, "mean_char_f1": trial.mean_char_f1}
            for trial in topic_selection.trials
        ],
        "reuse_predictions": (
            str(args.reuse_predictions) if args.reuse_predictions else None
        ),
        "modes": [item["mode"] for item in summaries],
        "gold_used_on_test_ranking": False,
    }
    (output_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metric_cache.save()
    return {"summaries": summaries, **manifest}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
