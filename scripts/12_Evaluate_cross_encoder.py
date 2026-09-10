"""Isolated cross-encoder chunk ranking on SciREx test, with optional static mix.

Does not overwrite previous hierarchical/static evaluations. Mix weights are
chosen on the dev split using gold char-F1, then frozen for test. Test ranking
never reads gold.
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

from untie.cli import load_static_keywords
from untie.config import PipelineConfig
from untie.cross_encoder_ranking import (
    MIX_WEIGHT_GRID,
    MixWeights,
    SentenceTransformerCrossEncoder,
    collect_ranked_panel,
    fit_mix_weights,
    predict_from_panel,
)
from untie.keyword_evidence import ExtractionMetricCache
from untie.keyword_training import SCORING_STRATEGIES
from untie.keyword_tuning import deterministic_document_split
from untie.models import ModelFactory, profile_for_language, resolve_device
from untie.pipelines import DocumentProcessor
from untie.topics import TopicDocument, compute_qa_ratios


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cross-encoder chunk ranking, optionally mixed with "
            "frozen static keywords"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--static-model", type=Path, required=True)
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
    parser.add_argument("--field-id", type=int, default=1)
    parser.add_argument("--chunk-max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=50)
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
        raise ValueError("mix-weight fitting requires a non-empty dev partition")
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
        root / "ce_topic_query" / "evaluation",
        root / "ce_topic_query_body" / "evaluation",
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
    }
    summaries.sort(key=lambda row: (order.get(str(row["mode"]), 10), str(row["mode"])))
    return summaries


def _row(
    *,
    document: TopicDocument,
    mode: str,
    prediction: str | None,
    metrics: dict[str, float],
    compute: dict[str, float | int],
    latency: float,
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
        "topic_keyword_node_id": "",
        "topic_keyword_ancestor_fallback": False,
        "latency_s": latency,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    if output_dir in protected_evaluation_dirs():
        raise ValueError(
            "cross-encoder evaluation refuses to overwrite a previous experiment "
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
    dev_documents, test_documents, references = load_split_documents(
        args.dataset, args.seed, quick=args.quick
    )
    test_ids = {item.doc_id for item in test_documents}
    reused = []
    if args.reuse_predictions is not None:
        reused = load_reused_rows(
            args.reuse_predictions,
            test_ids,
            exclude_modes={"cross-encoder", "cross-encoder-static-mix"},
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
    keywords, strategy = load_static_keywords(args.static_model, args.field_id)
    keyword_weight_ratio = SCORING_STRATEGIES.get(
        str(strategy.get("score_chunk_strategy", "")), 0.0
    )
    cluster_strategy = "weighted_score"
    answer_strategy = "combined_score"
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
            collect_ranked_panel(
                document,
                args.question,
                processor=processor,
                answerer=models.answerer,
                reranker=reranker,
                keywords=keywords,
                keyword_weight_ratio=keyword_weight_ratio,
                references=references[document.doc_id],
            )
        )
        if index % 5 == 0 or index == len(dev_documents):
            print(f"Dev panels {index}/{len(dev_documents)}", flush=True)

    selection = fit_mix_weights(
        dev_panels,
        models.sentence_encoder,
        score_char_f1,
        grid=MIX_WEIGHT_GRID,
        cluster_strategy=cluster_strategy,
        answer_strategy=answer_strategy,
    )
    ce_only = MixWeights(1.0, 0.0)
    print(
        f"Selected mix {selection.weights.name} "
        f"dev_char_f1={selection.mean_char_f1:.4f}",
        flush=True,
    )

    new_rows = []
    for index, document in enumerate(test_documents, start=1):
        started = time.perf_counter()
        panel = collect_ranked_panel(
            document,
            args.question,
            processor=processor,
            answerer=models.answerer,
            reranker=reranker,
            keywords=keywords,
            keyword_weight_ratio=keyword_weight_ratio,
            references=(),
        )
        compute = compute_qa_ratios(panel.chunks, panel.chunks)
        latency = time.perf_counter() - started
        for mode, weights in (
            ("cross-encoder", ce_only),
            ("cross-encoder-static-mix", selection.weights),
        ):
            prediction = predict_from_panel(
                panel,
                weights,
                models.sentence_encoder,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
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
        "static_model": str(args.static_model),
        "keyword_weight_ratio": keyword_weight_ratio,
        "score_chunk_strategy": strategy.get("score_chunk_strategy"),
        "cluster_strategy": cluster_strategy,
        "answer_strategy": answer_strategy,
        "ranking_query": "title+question",
        "keep_unmatched": True,
        "documents_dev": len(dev_documents),
        "documents_test": len(test_documents),
        "mix_weights": {
            "cross_encoder": selection.weights.cross_encoder,
            "static_keywords": selection.weights.static_keywords,
        },
        "dev_mean_char_f1": selection.mean_char_f1,
        "weight_grid": [
            {
                "cross_encoder": trial.weights.cross_encoder,
                "static_keywords": trial.weights.static_keywords,
                "mean_char_f1": trial.mean_char_f1,
            }
            for trial in selection.trials
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
