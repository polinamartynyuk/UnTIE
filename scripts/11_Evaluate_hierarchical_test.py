"""Test-only evaluation of a frozen hierarchical topic model.

Writes into a dedicated output directory and never replaces the source tuned
evaluation. Baseline and static-keyword rows can be reused from a previous
all-metrics CSV so only the new hierarchical model is executed.
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
from untie.keyword_evidence import ExtractionMetricCache
from untie.keyword_tuning import deterministic_document_split
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import DocumentProcessor, HierarchicalTopicRerankingPipeline
from untie.topics import TopicDocument, load_topic_artifact


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
        description="Evaluate a hierarchical topic model on held-out test documents"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--topic-model", type=Path, required=True)
    parser.add_argument("--static-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-predictions",
        type=Path,
        help="Previous test_predictions_all_metrics.csv; other modes are reused",
    )
    parser.add_argument("--mode-name", default="hierarchical-refit")
    parser.add_argument(
        "--soft-rerank",
        action="store_true",
        help=(
            "QA every chunk and use keywords/title/topic only to rank answers. "
            "Does not overwrite previous hard-filter evaluations."
        ),
    )
    parser.add_argument("--title-prior-weight", type=float, default=0.5)
    parser.add_argument("--topic-prior-weight", type=float, default=0.5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files in --output-dir only; never source evaluations",
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


def load_test_documents(
    path: Path, seed: int, *, quick: bool = False
) -> tuple[list[TopicDocument], dict[str, tuple[str, ...]]]:
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
        if split_by_id.get(str(raw.get("doc_id", raw.get("id")))) == "test"
    ]
    if quick:
        documents = documents[:8]
    references = {
        str(raw.get("doc_id", raw.get("id"))): _references(raw) for raw in payload
    }
    if not documents:
        raise ValueError("evaluation requires a non-empty test partition")
    missing = [item.doc_id for item in documents if not references.get(item.doc_id)]
    if missing:
        raise ValueError(f"test documents require task references: {missing[:5]}")
    return documents, references


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
        root / "ce_topic_query_body" / "evaluation",
    }


def load_reused_rows(
    path: Path, test_ids: set[str], *, exclude_modes: set[str] | None = None
) -> list[dict[str, object]]:
    excluded = exclude_modes or set()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    kept = []
    for row in rows:
        if row.get("doc_id") not in test_ids:
            continue
        mode = row.get("mode")
        copied = dict(row)
        if mode == "hierarchical":
            copied["mode"] = "hierarchical-tuned"
            mode = copied["mode"]
        if mode in excluded:
            continue
        kept.append(copied)
    return kept


def summarize_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    baseline = {
        str(row["doc_id"]): row for row in by_mode.get("baseline", ())
    }
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
                "mean_topic_keyword_ancestor_fallback": sum(
                    str(row.get("topic_keyword_ancestor_fallback", "")).lower()
                    in {"true", "1"}
                    for row in group
                )
                / count
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
    }
    summaries.sort(key=lambda row: (order.get(str(row["mode"]), 10), str(row["mode"])))
    return summaries


def _row_from_result(
    *,
    document: TopicDocument,
    mode: str,
    result,
    metrics: dict[str, float],
    latency: float,
    nodes,
) -> dict[str, object]:
    metadata = result.metadata or {}
    compute = metadata.get("compute") or {}
    route = metadata.get("topic_route") or ()
    routed_leaf = route[0]["node_id"] if route else None
    keyword_node = metadata.get("topic_keyword_node_id")
    parent_ids = {node.node_id for node in nodes if node.children}
    ancestor = bool(keyword_node) and keyword_node in parent_ids
    prediction = result.final_answer.text if result.final_answer else ""
    fallback = bool(metadata.get("keyword_fallback")) or bool(
        metadata.get("fallback_reason")
    )
    return {
        "doc_id": document.doc_id,
        "mode": mode,
        "prediction": prediction,
        **metrics,
        "composite_quality": composite_quality(metrics),
        "qa_chunks": compute.get("qa_chunks", len(result.used_chunks)),
        "qa_chunk_ratio": compute.get("qa_chunk_ratio", 1.0),
        "qa_token_ratio": compute.get("qa_token_ratio", 1.0),
        "fallback": fallback,
        "fallback_reason": metadata.get("fallback_reason") or "",
        "fallback_stage": metadata.get("fallback_stage") or "",
        "topic_keyword_node_id": keyword_node or "",
        "topic_keyword_ancestor_fallback": ancestor,
        "latency_s": latency,
        "routed_leaf": routed_leaf or "",
    }


def win_harm_by_cluster(
    rows: list[dict[str, object]], mode: str
) -> list[dict[str, object]]:
    baseline = {
        str(row["doc_id"]): float(row["char_f1"])
        for row in rows
        if row["mode"] == "baseline"
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["mode"] != mode:
            continue
        node = str(row.get("topic_keyword_node_id") or "")
        if str(row.get("fallback", "")).lower() in {"true", "1"} and not node:
            node = "UNASSIGNED"
        grouped.setdefault(node or "UNASSIGNED", []).append(row)
    summary = []
    for node_id, group in sorted(grouped.items()):
        deltas = [
            float(row["char_f1"]) - baseline[str(row["doc_id"])]
            for row in group
            if str(row["doc_id"]) in baseline
        ]
        fallbacks = sum(
            str(row.get("fallback", "")).lower() in {"true", "1"} for row in group
        )
        summary.append(
            {
                "node_id": node_id,
                "documents": len(group),
                "wins": sum(delta > 0 for delta in deltas),
                "harms": sum(delta < 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "mean_delta_char_f1": sum(deltas) / len(deltas) if deltas else 0.0,
                "fallbacks": fallbacks,
                "active_rerank": len(group) - fallbacks,
                "win_rate": sum(delta > 0 for delta in deltas) / len(deltas)
                if deltas
                else 0.0,
                "harm_rate": sum(delta < 0 for delta in deltas) / len(deltas)
                if deltas
                else 0.0,
                "fallback_rate": fallbacks / len(group) if group else 0.0,
            }
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    protected = protected_evaluation_dirs()
    if output_dir in protected:
        raise ValueError(
            "evaluation refuses to overwrite a previous experiment directory; "
            "choose --output-dir under a new experiment folder"
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
    documents, references = load_test_documents(
        args.dataset, args.seed, quick=args.quick
    )
    test_ids = {item.doc_id for item in documents}
    mode_name = args.mode_name
    if args.soft_rerank and mode_name == "hierarchical-refit":
        mode_name = "hierarchical-soft-rerank"
    reused = []
    if args.reuse_predictions is not None:
        reused = load_reused_rows(
            args.reuse_predictions, test_ids, exclude_modes={mode_name}
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
    artifact = load_topic_artifact(args.topic_model)
    pipeline = HierarchicalTopicRerankingPipeline(
        processor,
        models.answerer,
        models.sentence_encoder,
        config,
        artifact,
        keywords,
        static_weight_ratio={
            "only_score_diff": 0.0,
            "only_weight": 1.0,
            "equal_weight_score_diff": 0.5,
        }.get(str(strategy.get("score_chunk_strategy", "")), 0.5),
        expected_encoder_revision=artifact.encoder_revision or None,
        soft_rerank=bool(args.soft_rerank),
        title_prior_weight=float(args.title_prior_weight),
        topic_prior_weight=float(args.topic_prior_weight),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_cache = ExtractionMetricCache(output_dir / "extraction_metrics_cache.json")
    include_bertscore = not args.skip_bertscore
    new_rows = []
    for index, document in enumerate(documents, start=1):
        started = time.perf_counter()
        result = pipeline.run(
            document.text,
            args.question,
            document=document,
            cluster_strategy=str(
                "weighted_score"
                if args.soft_rerank
                else strategy.get("choose_cluster_strategy", "weighted_score")
            ),
            answer_strategy=str(
                "combined_score"
                if args.soft_rerank
                else strategy.get("choose_answer_strategy", "highest_similarity")
            ),
        )
        latency = time.perf_counter() - started
        prediction = result.final_answer.text if result.final_answer else None
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
            _row_from_result(
                document=document,
                mode=mode_name,
                result=result,
                metrics=metrics,
                latency=latency,
                nodes=artifact.nodes,
            )
        )
        if index % 5 == 0 or index == len(documents):
            print(f"Evaluated {index}/{len(documents)}", flush=True)
            metric_cache.save()

    all_rows = [*reused, *new_rows]
    summaries = summarize_metric_rows(all_rows)
    _write_csv(output_dir / "test_predictions_all_metrics.csv", all_rows, PREDICTION_FIELDS)
    (output_dir / "quality_summary_all_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cluster_rows = win_harm_by_cluster(all_rows, mode_name)
    _write_csv(
        output_dir / "win_harm_by_cluster.csv",
        cluster_rows,
        (
            "node_id",
            "documents",
            "wins",
            "harms",
            "ties",
            "mean_delta_char_f1",
            "fallbacks",
            "active_rerank",
            "win_rate",
            "harm_rate",
            "fallback_rate",
        ),
    )
    metric_cache.save()
    manifest = {
        "output_dir": str(output_dir),
        "topic_model": str(args.topic_model),
        "documents": len(documents),
        "mode_name": mode_name,
        "soft_rerank": bool(args.soft_rerank),
        "title_prior_weight": float(args.title_prior_weight),
        "topic_prior_weight": float(args.topic_prior_weight),
        "reuse_predictions": (
            str(args.reuse_predictions) if args.reuse_predictions else None
        ),
        "cluster_strategy": (
            "weighted_score"
            if args.soft_rerank
            else str(strategy.get("choose_cluster_strategy", "weighted_score"))
        ),
        "answer_strategy": (
            "combined_score"
            if args.soft_rerank
            else str(strategy.get("choose_answer_strategy", "highest_similarity"))
        ),
        "summaries": summaries,
    }
    (output_dir / "eval_manifest.json").write_text(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "summaries"}
            | {"modes": [item["mode"] for item in summaries]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


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
