from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attention import AttentionKeywordExtractor
from .config import PipelineConfig
from .models import ModelFactory, profile_for_language
from .pipelines import AnswerPipeline, AttentionRerankingPipeline, DocumentProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a short answer from a scientific text")
    parser.add_argument("input", type=Path, help="UTF-8 text file")
    parser.add_argument("--question", required=True)
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--mode", choices=("baseline", "attention"), default="baseline")
    parser.add_argument("--reference-answer", help="Required by attention mode")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "attention" and not args.reference_answer:
        raise SystemExit("--reference-answer is required in attention mode")

    config = PipelineConfig(
        profile=profile_for_language(args.language),
        device=args.device,
    )
    if args.mode == "attention" and not config.profile.attention_supported:
        raise SystemExit(f"Attention mode is not supported by the {args.language} profile")

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
    else:
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
