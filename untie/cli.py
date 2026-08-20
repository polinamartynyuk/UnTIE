from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attention import AttentionKeywordExtractor
from .config import PipelineConfig
from .models import ModelFactory, profile_for_language
from .model_params import load_model_params
from .pipelines import (
    AnswerPipeline,
    AttentionRerankingPipeline,
    DocumentProcessor,
    HierarchicalTopicRerankingPipeline,
    StaticKeywordRerankingPipeline,
)
from .ranking import WeightedKeyword
from .topics import TopicArtifactError, load_topic_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a short answer from a scientific text")
    parser.add_argument("input", type=Path, help="UTF-8 text file")
    parser.add_argument("--question", required=True)
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument(
        "--mode",
        choices=(
            "baseline",
            "attention",
            "static-keywords",
            "hierarchical-static-keywords",
        ),
        default="baseline",
    )
    parser.add_argument("--reference-answer", help="Required by attention mode")
    parser.add_argument(
        "--model-params",
        type=Path,
        help="Tuned static-keyword model JSON",
    )
    parser.add_argument(
        "--topic-model",
        type=Path,
        help="Versioned topic artifact required by hierarchical-static-keywords mode",
    )
    parser.add_argument(
        "--topic-encoder-revision",
        help="Optional expected encoder revision/fingerprint for compatibility checks",
    )
    parser.add_argument("--field-id", type=int)
    parser.add_argument("--device", default="auto")
    return parser


def load_static_keywords(
    path: Path, field_id: int | None = None
) -> tuple[list[WeightedKeyword], dict[str, object]]:
    model_params = load_model_params(path)
    fields = (
        [field for field in model_params.fields if field.field_id == field_id]
        if field_id is not None
        else model_params.fields[:1]
    )
    if not fields:
        raise ValueError(f"Field not found in model params: {field_id}")
    field = fields[0]
    metadata = {item.word: item for item in field.keyword_metadata}
    keywords = []
    for word in field.keywords:
        item = metadata.get(word)
        keywords.append(
            WeightedKeyword(
                word=word,
                lemma=(item.lemma if item and item.lemma else word.lower()),
                stem=(item.stem if item and item.stem else word.lower()),
                attention_weight=item.attention_weight if item else 1.0,
                score_difference=item.score_difference if item else 1.0,
            )
        )
    strategy = field.tuning_metadata.get("strategy", {})
    return keywords, dict(strategy) if isinstance(strategy, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "attention" and not args.reference_answer:
        raise SystemExit("--reference-answer is required in attention mode")
    if args.mode == "static-keywords" and not args.model_params:
        raise SystemExit("--model-params is required in static-keywords mode")
    if args.mode == "hierarchical-static-keywords" and not args.topic_model:
        raise SystemExit(
            "--topic-model is required in hierarchical-static-keywords mode"
        )

    config = PipelineConfig(
        profile=profile_for_language(args.language),
        device=args.device,
    )
    if args.mode == "attention" and not config.profile.attention_supported:
        raise SystemExit(f"Attention mode is not supported by the {args.language} profile")

    keywords: list[WeightedKeyword] = []
    strategy: dict[str, object] = {}
    topic_artifact = None
    try:
        if args.mode in {"static-keywords", "hierarchical-static-keywords"}:
            if args.model_params:
                keywords, strategy = load_static_keywords(
                    args.model_params, args.field_id
                )
        if args.mode == "hierarchical-static-keywords":
            topic_artifact = load_topic_artifact(args.topic_model)
    except (ValueError, TopicArtifactError) as error:
        raise SystemExit(str(error)) from error

    models = ModelFactory(config)
    processor = DocumentProcessor(
        tokenizer=models.tokenizer,
        config=config,
        sentence_encoder=models.sentence_encoder,
    )
    text = args.input.read_text(encoding="utf-8")
    if args.mode == "baseline":
        result = AnswerPipeline(
            processor, models.answerer, models.sentence_encoder, config
        ).run(text, [args.question])
    elif args.mode == "attention":
        attention = AttentionKeywordExtractor(
            models.qa_model, models.tokenizer, models.device
        )
        result = AttentionRerankingPipeline(
            processor,
            models.answerer,
            models.sentence_encoder,
            config,
            attention,
        ).run(text, args.question, args.reference_answer)
    elif args.mode == "static-keywords":
        result = StaticKeywordRerankingPipeline(
            processor,
            models.answerer,
            models.sentence_encoder,
            config,
            keywords,
        ).run(
            text,
            args.question,
            weight_ratio={
                "only_score_diff": 0.0,
                "only_weight": 1.0,
                "equal_weight_score_diff": 0.5,
            }.get(str(strategy.get("score_chunk_strategy", "")), 0.5),
            cluster_strategy=str(
                strategy.get("choose_cluster_strategy", "weighted_score")
            ),
            answer_strategy=str(
                strategy.get("choose_answer_strategy", "combined_score")
            ),
        )
    else:
        if topic_artifact is None:  # Defensive: parser validation requires it.
            raise SystemExit("topic artifact was not loaded")
        try:
            result = HierarchicalTopicRerankingPipeline(
                processor,
                models.answerer,
                models.sentence_encoder,
                config,
                topic_artifact,
                keywords,
                static_weight_ratio={
                    "only_score_diff": 0.0,
                    "only_weight": 1.0,
                    "equal_weight_score_diff": 0.5,
                }.get(str(strategy.get("score_chunk_strategy", "")), 0.5),
                expected_encoder_revision=args.topic_encoder_revision,
            ).run(
                text,
                args.question,
                cluster_strategy=str(
                    strategy.get("choose_cluster_strategy", "weighted_score")
                ),
                answer_strategy=str(
                    strategy.get("choose_answer_strategy", "combined_score")
                ),
            )
        except TopicArtifactError as error:
            raise SystemExit(str(error)) from error

    payload = {
        "answer": result.final_answer.text if result.final_answer else None,
        "confidence": result.final_answer.confidence if result.final_answer else None,
        "chunks_used": len(result.used_chunks),
        "metadata": result.metadata,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
