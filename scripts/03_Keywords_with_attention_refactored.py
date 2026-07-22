"""Пакетный эксперимент с attention-ключами на новом API UnTIE.

Это поддерживаемый аналог ``03_Keywords_with_attention.py``. Он сохраняет исходные
3 режима оценки × 3 стратегии выбора кластера × 3 стратегии выбора ответа, но
устраняет жёсткую привязку к рабочему каталогу и повторные проходы QA.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.attention import AttentionKeywordExtractor
from untie.config import PipelineConfig
from untie.data import load_json_dataframe, replace_underscores, save_json_dataframe
from untie.keywords import KeywordExtractor, score_keyword_contrast
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import AnswerPipeline, DocumentProcessor
from untie.qa import AnswerConsensus, ScoredAnswerFinder
from untie.ranking import WeightedKeyword, score_chunks
from untie.domain import Question


LOGGER = logging.getLogger("untie.attention_batch")

SCORING_MODES = {
    "only_score_diff": 0.0,
    "only_weight": 1.0,
    "equal_weight_score_diff": 0.5,
}
CLUSTER_STRATEGIES = (
    "highest_avg_score",
    "weighted_score",
    "highest_cohesion",
)
ANSWER_STRATEGIES = (
    "highest_chunk_score",
    "highest_similarity",
    "combined_score",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the refactored attention-keyword strategy experiment"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scirex_structured.json",
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        default=PROJECT_ROOT / "model_params" / "scart_init_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "results_keys_refactored.json",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attention-top-k", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def load_field(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload.get("fields", [])
    if not fields:
        raise ValueError(f"No fields found in model parameters: {path}")
    field = fields[0]
    if not field.get("questions"):
        raise ValueError(f"No questions found in model parameters: {path}")
    return field


def first_reference_answer(record: pd.Series) -> str:
    references = record.get("tasks_cleaned", record.get("tasks"))
    if isinstance(references, list) and references:
        return str(references[0])
    if isinstance(references, str) and references:
        return references
    raise ValueError("Record has no reference answer in tasks_cleaned/tasks")


def merge_attention_keywords(
    raw_batches: list[list[dict[str, float | str]]],
    filtered_chunks: list[str],
) -> list[dict[str, float | str]]:
    merged: dict[str, dict[str, float | str]] = {}
    for batch, filtered_text in zip(raw_batches, filtered_chunks):
        allowed = set(re.findall(r"\b[\w-]+\b", filtered_text.lower()))
        for keyword in batch:
            word = str(keyword["word"]).lower()
            if word not in allowed or word in ENGLISH_STOP_WORDS or len(word) <= 1:
                continue
            if word not in merged or float(keyword.get("weight", 0)) > float(
                merged[word].get("weight", 0)
            ):
                merged[word] = {**keyword, "word": word}
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("weight", 0)),
        reverse=True,
    )


def to_weighted_keywords(
    keywords: list[dict[str, float | str]],
) -> list[WeightedKeyword]:
    try:
        from nltk.stem import SnowballStemmer

        stem = SnowballStemmer("english").stem
    except ImportError:
        stem = lambda word: word
    return [
        WeightedKeyword(
            word=str(keyword["word"]),
            lemma=str(keyword["word"]).lower(),
            stem=stem(str(keyword["word"]).lower()),
            attention_weight=float(keyword.get("weight", 1.0)),
            score_difference=float(keyword.get("score_diff", 1.0)),
        )
        for keyword in keywords
    ]


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def text_metrics(encoder: Any, reference: str, answer: str) -> dict[str, float | int]:
    embeddings = np.asarray(encoder.encode([reference, answer]))
    return {
        "cosine_sim": float(cosine_similarity(embeddings[:1], embeddings[1:])[0, 0]),
        "lev_dist": levenshtein(reference, answer),
    }


def no_valid_answer_row(
    record: pd.Series,
    reference_answers: Any,
    first_answer: str | None,
    base_metrics: dict[str, float | int] | None,
) -> dict[str, Any]:
    filler = "--No valid answers--"
    return {
        "doc_id": record.get("doc_id"),
        "original_text": record.get("original_text"),
        "tasks_cleaned": reference_answers,
        "first_answer": first_answer or "--None--",
        "score_chunk_strategy": filler,
        "choose_cluster_strategy": filler,
        "choose_answer_strategy": filler,
        "final_answer": filler,
        "base_metrics": base_metrics,
        "corrected_metrics": filler,
    }


def process_record(
    record: pd.Series,
    *,
    field: dict[str, Any],
    config: PipelineConfig,
    models: ModelFactory,
    processor: DocumentProcessor,
    attention: AttentionKeywordExtractor,
) -> list[dict[str, Any]]:
    text = str(record["original_text"])
    reference = first_reference_answer(record)
    questions = [str(question) for question in field["questions"]]
    aspect_name = str(field["field_name"])

    baseline = AnswerPipeline(
        processor, models.answerer, models.sentence_encoder, config
    )
    initial = baseline.run(text, questions)
    first_answer = initial.final_answer.text if initial.final_answer else None
    base_metrics = (
        text_metrics(models.sentence_encoder, reference, first_answer)
        if first_answer
        else None
    )
    valid_answers = baseline.validate(initial, reference)
    reference_answers = record.get("tasks_cleaned", record.get("tasks"))
    if not valid_answers:
        return [
            no_valid_answer_row(
                record, reference_answers, first_answer, base_metrics
            )
        ]

    valid_chunks = [answer.chunk.text for answer in valid_answers]
    raw_batches = [attention(questions[0], chunk) for chunk in valid_chunks]
    keyword_extractor = KeywordExtractor(
        models.sentence_encoder, lemmatizer=str.lower
    )
    filtered_chunks, _ = keyword_extractor.dynamic_idf_filter(
        valid_chunks,
        initial_threshold=config.keyword_idf_threshold,
    )
    merged = merge_attention_keywords(raw_batches, filtered_chunks)
    contrasted = score_keyword_contrast(
        merged,
        positive_reference=aspect_name,
        negative_reference=reference,
        encoder=models.sentence_encoder,
    )
    keywords = to_weighted_keywords(contrasted)
    if not keywords:
        return [
            no_valid_answer_row(
                record, reference_answers, first_answer, base_metrics
            )
        ]

    all_chunks = processor.process(text)
    consensus = AnswerConsensus(models.sentence_encoder)
    rows: list[dict[str, Any]] = []
    for scoring_name, weight_ratio in SCORING_MODES.items():
        scored_chunks = score_chunks(
            all_chunks,
            keywords,
            weight_ratio=weight_ratio,
        )
        answered = ScoredAnswerFinder(models.answerer).find(
            Question(questions[0]), scored_chunks
        )
        for cluster_strategy in CLUSTER_STRATEGIES:
            for answer_strategy in ANSWER_STRATEGIES:
                selected = consensus.select_clustered(
                    answered.answers,
                    cluster_strategy=cluster_strategy,
                    answer_strategy=answer_strategy,
                )
                rows.append(
                    {
                        "doc_id": record.get("doc_id"),
                        "original_text": text,
                        "tasks_cleaned": reference_answers,
                        "first_answer": first_answer or "--None--",
                        "score_chunk_strategy": scoring_name,
                        "choose_cluster_strategy": cluster_strategy,
                        "choose_answer_strategy": answer_strategy,
                        "final_answer": selected.text if selected else "--None--",
                        "base_metrics": base_metrics,
                        "corrected_metrics": (
                            text_metrics(
                                models.sentence_encoder, reference, selected.text
                            )
                            if selected
                            else "--None--"
                        ),
                    }
                )
    return rows


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.language != "en":
        raise ValueError(
            "The attention workflow is currently validated only for the English profile"
        )
    dataframe = load_json_dataframe(args.dataset)
    dataframe = replace_underscores(
        dataframe, "tasks", destination_column="tasks_cleaned"
    )
    field = load_field(args.model_params)
    config = PipelineConfig(
        profile=profile_for_language(args.language),
        device=args.device,
    )
    models = ModelFactory(config)
    processor = DocumentProcessor(
        models.tokenizer,
        config,
        sentence_encoder=models.sentence_encoder,
    )
    attention = AttentionKeywordExtractor(
        models.qa_model,
        models.tokenizer,
        models.device,
        limit=args.attention_top_k,
    )

    stop = len(dataframe)
    if args.limit is not None:
        stop = min(stop, args.start_index + args.limit)
    output_rows: list[dict[str, Any]] = []
    for index in range(args.start_index, stop):
        try:
            output_rows.extend(
                process_record(
                    dataframe.iloc[index],
                    field=field,
                    config=config,
                    models=models,
                    processor=processor,
                    attention=attention,
                )
            )
            save_json_dataframe(pd.DataFrame(output_rows), args.output)
            LOGGER.info("Processed document %s/%s", index + 1, stop)
        except Exception:
            LOGGER.exception("Failed to process document at index %s", index)
            if not args.continue_on_error:
                raise
    return pd.DataFrame(output_rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run(args)
    LOGGER.info("Saved %s strategy rows to %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
