"""Tune one static-style SFFS dictionary per train-derived topic cluster."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.config import PipelineConfig
from untie.keyword_evidence import (
    EvidenceStore,
    ExtractionMetricCache,
    build_evidence_fingerprint,
)
from untie.keyword_training import MetricWeights, StrategyConfig, TrainingConfig
from untie.keyword_tuning import ObjectiveConfig, deterministic_document_split
from untie.models import ModelFactory, profile_for_language
from untie.topic_keyword_training import tune_topic_keyword_profiles
from untie.topics import (
    TopicDocument,
    load_topic_artifact,
    save_topic_artifact_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune cluster-specific keyword dictionaries for a topic artifact"
    )
    parser.add_argument("--topic-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-id", type=int, default=1)
    parser.add_argument("--question", default="Which task was solved?")
    parser.add_argument("--attention-top-k", type=int, default=100)
    parser.add_argument("--chunk-max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--min-train-documents", type=int, default=2)
    parser.add_argument("--min-dev-documents", type=int, default=1)
    parser.add_argument("--min-document-support", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-keywords", type=int, default=8)
    parser.add_argument("--evaluation-budget", type=int, default=100)
    parser.add_argument("--stability-runs", type=int, default=3)
    parser.add_argument("--stability-threshold", type=float, default=0.4)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument(
        "--score-strategy",
        choices=("only_score_diff", "only_weight", "equal_weight_score_diff"),
        default="only_score_diff",
    )
    parser.add_argument(
        "--cluster-strategy",
        choices=("highest_avg_score", "weighted_score", "highest_cohesion"),
        default="weighted_score",
    )
    parser.add_argument(
        "--answer-strategy",
        choices=("highest_chunk_score", "highest_similarity", "combined_score"),
        default="highest_similarity",
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
    if not isinstance(payload, list):
        raise ValueError("dataset must contain a JSON array or JSONL records")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("each dataset record must be an object")
    return payload


def load_topic_documents(
    path: Path,
    seed: int,
) -> tuple[list[TopicDocument], list[TopicDocument]]:
    payload = _load_payload(path)
    ids = [str(item.get("doc_id", item.get("id", ""))) for item in payload]
    if any(not doc_id for doc_id in ids):
        raise ValueError("each dataset record must have doc_id or id")
    explicit_splits = [item.get("split") for item in payload]
    if any(value is not None for value in explicit_splits):
        if not all(value is not None for value in explicit_splits):
            raise ValueError("dataset split must be present on every record or none")
        normalized = [str(value).casefold() for value in explicit_splits]
        invalid = sorted(set(normalized).difference({"train", "dev", "test"}))
        if invalid:
            raise ValueError(f"unsupported dataset splits: {invalid}")
        train_ids = {
            doc_id
            for doc_id, split_name in zip(ids, normalized)
            if split_name == "train"
        }
        dev_ids = {
            doc_id
            for doc_id, split_name in zip(ids, normalized)
            if split_name == "dev"
        }
    else:
        split = deterministic_document_split(ids, seed=seed)
        train_ids = set(split["train"])
        dev_ids = set(split["dev"])

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

    train = [
        convert(raw, "train")
        for raw in payload
        if str(raw.get("doc_id", raw.get("id"))) in train_ids
    ]
    dev = [
        convert(raw, "dev")
        for raw in payload
        if str(raw.get("doc_id", raw.get("id"))) in dev_ids
    ]
    return train, dev


def load_reference_map(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for raw in _load_payload(path):
        doc_id = str(raw.get("doc_id", raw.get("id", "")))
        value = raw.get("tasks_cleaned", raw.get("tasks", ()))
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
            value = parsed
        values = value if isinstance(value, (list, tuple)) else (value,)
        result[doc_id] = tuple(
            str(item).replace("_", " ").strip()
            for item in values
            if item is not None and str(item).strip()
        )
    return result


def run(args: argparse.Namespace):
    artifact = load_topic_artifact(args.topic_model)
    train_documents, dev_documents = load_topic_documents(args.dataset, args.seed)
    references = load_reference_map(args.dataset)
    profile = profile_for_language(args.language)
    config = PipelineConfig(
        profile=profile,
        chunk_max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.overlap_tokens,
        device=args.device,
    )
    evidence_store = EvidenceStore(args.evidence_dir)
    evidence = []
    missing = []
    for document in (*train_documents, *dev_documents):
        expected = build_evidence_fingerprint(
            doc_id=document.doc_id,
            text=document.text,
            aspect_id=str(args.field_id),
            question=args.question,
            references=references.get(document.doc_id, ()),
            collect_candidates=document.split == "train",
            fingerprint_payload={
                "language": args.language,
                "qa_model": profile.qa_model,
                "sentence_model": profile.sentence_model,
                "chunk_max_tokens": args.chunk_max_tokens,
                "overlap_tokens": args.overlap_tokens,
                "attention_top_k": args.attention_top_k,
                "pipeline_version": "static-keywords-v1",
            },
        )
        cached = evidence_store.load(
            document.doc_id,
            expected_fingerprint=expected,
        )
        if cached is None:
            missing.append(document.doc_id)
        else:
            evidence.append(cached)
    if missing:
        raise ValueError(
            "keyword evidence is missing or incompatible; run "
            "scripts/05_Tune_model_keywords.py with the same dataset, split, "
            f"question, models and chunk settings first. Invalid ids: {missing[:5]}"
        )

    encoder = ModelFactory(config).sentence_encoder
    metric_cache_path = args.metric_cache or (
        args.output.parent / f"{args.output.stem}.metrics.json"
    )
    result = tune_topic_keyword_profiles(
        artifact,
        train_documents,
        dev_documents,
        evidence,
        encoder,
        config=TrainingConfig(
            language=args.language,
            min_document_support=args.min_document_support,
            max_candidates=args.max_candidates,
            max_keywords=args.max_keywords,
            evaluation_budget=args.evaluation_budget,
            stability_runs=args.stability_runs,
            stability_threshold=args.stability_threshold,
            seed=args.seed,
            screen_top_k=min(40, args.max_candidates),
            strategies=(
                StrategyConfig(
                    args.score_strategy,
                    args.cluster_strategy,
                    args.answer_strategy,
                ),
            ),
        ),
        metric_cache=ExtractionMetricCache(metric_cache_path),
        metric_weights=MetricWeights.exact_match(),
        objective_config=ObjectiveConfig(
            downside_penalty=0.75,
            harm_penalty=0.5,
            fallback_penalty=0.1,
            harm_threshold=0.01,
            confidence_weight=0.25,
            bootstrap_seed=args.seed,
        ),
        checkpoint_dir=args.checkpoint_dir
        or args.output.parent / f"{args.output.stem}.checkpoints",
        min_train_documents=args.min_train_documents,
        min_dev_documents=args.min_dev_documents,
        expected_encoder_name=profile.sentence_model,
    )
    save_topic_artifact_atomic(result.artifact, args.output)
    summary_path = args.output.with_suffix(".topic-keywords.json")
    summary_path.write_text(
        json.dumps(
            {
                "trained_nodes": {
                    node_id: {
                        "keywords": list(outcome.keywords),
                        "objective": outcome.objective,
                        "train_documents": len(outcome.train_doc_ids),
                        "dev_documents": len(outcome.dev_doc_ids),
                    }
                    for node_id, outcome in sorted(result.outcomes.items())
                },
                "skipped_nodes": dict(sorted(result.skipped_nodes.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result, summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, summary_path = run(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": str(summary_path),
                "trained_nodes": len(result.outcomes),
                "skipped_nodes": len(result.skipped_nodes),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
