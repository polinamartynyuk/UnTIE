"""Build a versioned hierarchical topic artifact from train documents only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language
from untie.topics import (
    TopicBuildConfig,
    TopicDocument,
    TopicMixingConfig,
    TopicRoutingConfig,
    build_topic_artifact,
    save_topic_artifact_atomic,
    save_topic_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a train-only hierarchical topic model"
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="Defaults to <output-stem>.diagnostics beside the artifact",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument(
        "--assume-train",
        action="store_true",
        help="Allow records without an explicit split field to be treated as train",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encoder-revision",
        default="",
        help="Optional immutable model revision or local weights fingerprint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leaf-clusters", type=int, default=4)
    parser.add_argument("--ngram-max", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--min-topic-document-support", type=float, default=0.2)
    parser.add_argument("--topic-term-top-k", type=int, default=30)
    parser.add_argument("--routing-top-k", type=int, default=3)
    parser.add_argument("--routing-beam-width", type=int, default=2)
    parser.add_argument("--routing-temperature", type=float, default=0.1)
    parser.add_argument("--routing-similarity-threshold", type=float, default=0.0)
    parser.add_argument("--routing-min-margin", type=float, default=0.0)
    parser.add_argument("--aspect-threshold", type=float, default=0.0)
    parser.add_argument("--theta-aspect", type=float, default=1.0)
    parser.add_argument("--theta-topic", type=float, default=0.5)
    parser.add_argument("--theta-interaction", type=float, default=0.5)
    parser.add_argument(
        "--max-qa-chunks",
        type=int,
        default=0,
        help="0 keeps every chunk (safe reranking-only default)",
    )
    return parser


def load_train_documents(
    path: Path, *, assume_train: bool = False
) -> list[TopicDocument]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("documents")
    if not isinstance(payload, list):
        raise ValueError("train dataset must be a JSON array or JSONL records")
    documents = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"document {index} must be an object")
        if "split" not in raw and not assume_train:
            raise ValueError(
                f"document {index}.split is required; use --assume-train explicitly"
            )
        headings = raw.get("headings", [])
        if isinstance(headings, str):
            headings = [headings]
        if not isinstance(headings, list):
            raise ValueError(f"document {index}.headings must be an array")
        documents.append(
            TopicDocument(
                doc_id=str(raw.get("doc_id", raw.get("id", index))),
                text=str(raw.get("text", raw.get("original_text", ""))),
                title=str(raw.get("title", "")),
                abstract=str(raw.get("abstract", "")),
                headings=tuple(str(item) for item in headings),
                split=str(raw.get("split", "train" if assume_train else "")),
            )
        )
    return documents


def run(args: argparse.Namespace):
    documents = load_train_documents(
        args.train_dataset, assume_train=args.assume_train
    )
    profile = profile_for_language(args.language)
    config = PipelineConfig(profile=profile, device=args.device)
    encoder = ModelFactory(config).sentence_encoder
    artifact = build_topic_artifact(
        documents,
        encoder,
        encoder_name=profile.sentence_model,
        encoder_revision=args.encoder_revision,
        config=TopicBuildConfig(
            leaf_clusters=args.leaf_clusters,
            ngram_max=args.ngram_max,
            min_topic_document_support=args.min_topic_document_support,
            topic_term_top_k=args.topic_term_top_k,
            seed=args.seed,
        ),
        routing=TopicRoutingConfig(
            top_k=args.routing_top_k,
            beam_width=args.routing_beam_width,
            temperature=args.routing_temperature,
            similarity_threshold=args.routing_similarity_threshold,
            min_margin=args.routing_min_margin,
        ),
        mixing=TopicMixingConfig(
            aspect_threshold=args.aspect_threshold,
            theta_aspect=args.theta_aspect,
            theta_topic=args.theta_topic,
            theta_interaction=args.theta_interaction,
            max_qa_chunks=args.max_qa_chunks,
        ),
    )
    save_topic_artifact_atomic(artifact, args.output)
    diagnostics_dir = args.diagnostics_dir or (
        args.output.parent / f"{args.output.stem}.diagnostics"
    )
    save_topic_diagnostics(artifact, diagnostics_dir)
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": artifact.schema_version,
                "nodes": len(artifact.nodes),
                "train_documents": artifact.node_map[
                    artifact.root_id
                ].document_count,
                "seed": artifact.build.seed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
