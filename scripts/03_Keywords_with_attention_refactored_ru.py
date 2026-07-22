"""Обработка русского датасета RusErrC через новое ядро UnTIE.

Сценарий повторяет английский attention-эксперимент: выполняет первичный QA,
выделяет ключевые слова по attention, оценивает чанки тремя способами и сохраняет
результаты для 27 комбинаций стратегий. В качестве входа используется
предобработанный ``ruserrc_structured.csv``.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.attention import AttentionKeywordExtractor
from untie.config import PipelineConfig
from untie.data import save_json_dataframe
from untie.domain import Question
from untie.keywords import KeywordExtractor, score_keyword_contrast
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import AnswerPipeline, DocumentProcessor
from untie.qa import AnswerConsensus, ScoredAnswerFinder
from untie.ranking import WeightedKeyword, score_chunks
from untie.text import RussianSentenceSplitter


LOGGER = logging.getLogger("untie.attention_batch_ru")

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

# Частотные русские служебные слова удаляются до оценки чанков. Набор хранится
# локально, поэтому для запуска не требуется скачивать корпус стоп-слов NLTK.
RUSSIAN_STOP_WORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
    "в", "вам", "вас", "весь", "во", "вот", "все", "всего", "всех",
    "вы", "где", "да", "даже", "для", "до", "его", "ее", "если", "есть",
    "еще", "же", "за", "здесь", "и", "из", "или", "им", "их", "к", "как",
    "ко", "когда", "который", "ли", "либо", "мне", "может", "мы", "на",
    "над", "надо", "наш", "не", "него", "нее", "нет", "ни", "них", "но",
    "ну", "о", "об", "однако", "он", "она", "они", "оно", "от", "очень",
    "по", "под", "при", "про", "с", "сам", "себя", "со", "так", "также",
    "такой", "там", "те", "тем", "то", "того", "тоже", "той", "только",
    "том", "тот", "у", "уже", "чем", "что", "чтобы", "эта", "эти", "это",
    "этого", "этой", "этот", "я",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обработать RusErrC с attention-переранжированием"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ruserrc_structured.csv",
    )
    parser.add_argument(
        "--model-params",
        type=Path,
        default=PROJECT_ROOT / "model_params" / "ruserrc_init_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "results_keys_rus_refactored.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attention-top-k", type=int, default=100)
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=128,
        help="Максимальный размер RU-чанка в токенах",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=24,
        help="Перекрытие соседних RU-чанков в токенах",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def load_field(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload.get("fields", [])
    if not fields or not fields[0].get("questions"):
        raise ValueError(f"В параметрах модели отсутствует поле с вопросами: {path}")
    return fields[0]


def parse_string_list(value: Any) -> list[str]:
    """Преобразует CSV-представление списка аспектов в список строк."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def load_ruserrc(path: Path) -> pd.DataFrame:
    """Загружает предобработанный RusErrC и приводит его к контракту UnTIE."""
    dataframe = pd.read_csv(path, sep=";", encoding="utf-8")
    required = {"id", "text_clean", "Task_aspects"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"В RusErrC отсутствуют колонки: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "doc_id": dataframe["id"].astype(str),
            "original_text": dataframe["text_clean"].fillna(dataframe["text"]),
            "tasks_cleaned": dataframe["Task_aspects"].apply(parse_string_list),
        }
    )
    without_reference = normalized["tasks_cleaned"].map(len) == 0
    skipped = int(without_reference.sum())
    if skipped:
        LOGGER.info(
            "Пропущено записей без Task_aspects: %s из %s",
            skipped,
            len(normalized),
        )
    return normalized.loc[~without_reference].reset_index(drop=True)


def build_russian_normalizers() -> tuple[Callable[[str], str], Callable[[str], str]]:
    """Создаёт русские лемматизатор и стеммер без загрузки моделей при импорте."""
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as error:
        raise RuntimeError(
            "Для русского сценария установите pymorphy3"
        ) from error

    morph = MorphAnalyzer()
    try:
        from nltk.stem import SnowballStemmer

        stemmer = SnowballStemmer("russian")
        stem = stemmer.stem
    except ImportError:
        stem = lambda word: word.lower()

    def lemmatize(word: str) -> str:
        return morph.parse(word.lower())[0].normal_form

    return lemmatize, stem


def merge_attention_keywords(
    raw_batches: list[list[dict[str, float | str]]],
    filtered_chunks: list[str],
) -> list[dict[str, float | str]]:
    """Объединяет attention-ключи и оставляет слова из очищенных чанков."""
    merged: dict[str, dict[str, float | str]] = {}
    for batch, filtered_text in zip(raw_batches, filtered_chunks):
        allowed = set(re.findall(r"\b[а-яёa-z][а-яёa-z-]*\b", filtered_text.lower()))
        for keyword in batch:
            word = str(keyword["word"]).lower()
            if word not in allowed or word in RUSSIAN_STOP_WORDS or len(word) <= 1:
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
    *,
    lemmatize: Callable[[str], str],
    stem: Callable[[str], str],
) -> list[WeightedKeyword]:
    """Добавляет к русским ключам леммы, стемы и два исходных веса."""
    return [
        WeightedKeyword(
            word=str(keyword["word"]),
            lemma=lemmatize(str(keyword["word"])),
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
                    previous[right_index - 1] + (left_character != right_character),
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


def strategy_diagnostics(
    *,
    total_chunk_count: int,
    scored_chunk_count: int,
    answers: list[Any],
    valid_answer_count: int,
    keyword_count: int,
    chunk_max_tokens: int,
) -> dict[str, int | bool]:
    """Описывает, был ли реальный выбор между несколькими RU-ответами."""
    distinct_answer_count = len(
        {
            str(answer.text).strip()
            for answer in answers
            if str(answer.text).strip()
        }
    )
    return {
        "total_chunk_count": total_chunk_count,
        "scored_chunk_count": scored_chunk_count,
        "answer_candidate_count": len(answers),
        "distinct_answer_count": distinct_answer_count,
        "strategy_applicable": distinct_answer_count > 1,
        "valid_answer_count": valid_answer_count,
        "keyword_count": keyword_count,
        "chunk_max_tokens": chunk_max_tokens,
    }


def no_valid_answer_row(
    record: pd.Series,
    first_answer: str | None,
    base_metrics: dict[str, float | int] | None,
    reason: str = "--No valid answers--",
    *,
    total_chunk_count: int = 0,
    valid_answer_count: int = 0,
    keyword_count: int = 0,
    chunk_max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": record["doc_id"],
        "original_text": record["original_text"],
        "tasks_cleaned": record["tasks_cleaned"],
        "first_answer": first_answer or "--None--",
        "score_chunk_strategy": reason,
        "choose_cluster_strategy": reason,
        "choose_answer_strategy": reason,
        "final_answer": reason,
        "base_metrics": base_metrics,
        "corrected_metrics": reason,
        "total_chunk_count": total_chunk_count,
        "scored_chunk_count": 0,
        "answer_candidate_count": 0,
        "distinct_answer_count": 0,
        "strategy_applicable": False,
        "valid_answer_count": valid_answer_count,
        "keyword_count": keyword_count,
        "chunk_max_tokens": chunk_max_tokens,
    }


def process_record(
    record: pd.Series,
    *,
    field: dict[str, Any],
    config: PipelineConfig,
    models: ModelFactory,
    processor: DocumentProcessor,
    attention: AttentionKeywordExtractor,
    lemmatize: Callable[[str], str],
    stem: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Выполняет полный attention-конвейер для одной русской статьи."""
    text = str(record["original_text"])
    reference = str(record["tasks_cleaned"][0])
    questions = [str(question) for question in field["questions"]]
    aspect_name = str(field["field_name"])

    baseline = AnswerPipeline(
        processor, models.answerer, models.sentence_encoder, config
    )
    initial = baseline.run(text, questions)
    total_chunk_count = len(initial.used_chunks)
    first_answer = initial.final_answer.text if initial.final_answer else None
    base_metrics = (
        text_metrics(models.sentence_encoder, reference, first_answer)
        if first_answer
        else None
    )
    valid_answers = baseline.validate(initial, reference)
    if not valid_answers:
        row = no_valid_answer_row(
            record,
            first_answer,
            base_metrics,
            total_chunk_count=total_chunk_count,
            chunk_max_tokens=config.chunk_max_tokens,
        )
        return [row]

    valid_chunks = [answer.chunk.text for answer in valid_answers]
    raw_batches = [attention(questions[0], chunk) for chunk in valid_chunks]
    keyword_extractor = KeywordExtractor(
        models.sentence_encoder,
        lemmatizer=lemmatize,
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
    keywords = to_weighted_keywords(
        contrasted,
        lemmatize=lemmatize,
        stem=stem,
    )
    if not keywords:
        return [
            no_valid_answer_row(
                record,
                first_answer,
                base_metrics,
                reason="--No keywords--",
                total_chunk_count=total_chunk_count,
                valid_answer_count=len(valid_answers),
                chunk_max_tokens=config.chunk_max_tokens,
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
            Question(questions[0]),
            scored_chunks,
        )
        diagnostics = strategy_diagnostics(
            total_chunk_count=len(all_chunks),
            scored_chunk_count=len(scored_chunks),
            answers=answered.answers,
            valid_answer_count=len(valid_answers),
            keyword_count=len(keywords),
            chunk_max_tokens=config.chunk_max_tokens,
        )
        for cluster_strategy in CLUSTER_STRATEGIES:
            for answer_strategy in ANSWER_STRATEGIES:
                selected = consensus.select_clustered(
                    answered.answers,
                    similarity_threshold=config.answer_cluster_threshold,
                    cluster_strategy=cluster_strategy,
                    answer_strategy=answer_strategy,
                )
                rows.append(
                    {
                        "doc_id": record["doc_id"],
                        "original_text": text,
                        "tasks_cleaned": record["tasks_cleaned"],
                        "first_answer": first_answer or "--None--",
                        "score_chunk_strategy": scoring_name,
                        "choose_cluster_strategy": cluster_strategy,
                        "choose_answer_strategy": answer_strategy,
                        "final_answer": selected.text if selected else "--None--",
                        "base_metrics": base_metrics,
                        "corrected_metrics": (
                            text_metrics(
                                models.sentence_encoder,
                                reference,
                                selected.text,
                            )
                            if selected
                            else "--None--"
                        ),
                        **diagnostics,
                    }
                )
    return rows


def run(args: argparse.Namespace) -> pd.DataFrame:
    dataframe = load_ruserrc(args.dataset)
    field = load_field(args.model_params)
    # Attention включается явно: локальная русская QA-модель умеет возвращать
    # attention, но базовый RU-профиль не включает его для обычного CLI.
    profile = replace(profile_for_language("ru"), attention_supported=True)
    config = PipelineConfig(
        profile=profile,
        device=args.device,
        chunk_max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    models = ModelFactory(config)
    processor = DocumentProcessor(
        models.tokenizer,
        config,
        splitter=RussianSentenceSplitter(),
        sentence_encoder=models.sentence_encoder,
    )
    attention = AttentionKeywordExtractor(
        models.qa_model,
        models.tokenizer,
        models.device,
        limit=args.attention_top_k,
    )
    lemmatize, stem = build_russian_normalizers()

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
                    lemmatize=lemmatize,
                    stem=stem,
                )
            )
            save_json_dataframe(pd.DataFrame(output_rows), args.output)
            LOGGER.info("Обработан документ %s/%s", index + 1, stop)
        except Exception:
            LOGGER.exception("Ошибка обработки документа с индексом %s", index)
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
    LOGGER.info("Сохранено строк: %s; файл: %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
