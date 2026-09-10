"""Topic-conditioned CE shortlist with baseline AnswerAggregator.

Cluster SFFS dictionaries are not used. The frozen topic route only expands
the CE query with leaf representative terms. Extractive QA still runs on
every chunk; CE only chooses which answers the aggregator sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .cross_encoder_ranking import (
    PairwiseReranker,
    cross_encoder_chunk_scores,
    ranking_query,
)
from .domain import Answer, Question, TextChunk
from .qa import AnswerAggregator, AnswerFinder
from .topics import TopicDocument, TopicTerm


SHORTLIST_K_GRID: tuple[int | None, ...] = (8, 12, 16, None)
DEFAULT_MAX_TOPIC_TERMS = 8


@dataclass(frozen=True)
class ShortlistKTrial:
    k: int | None
    mean_char_f1: float

    @property
    def label(self) -> str:
        return "all" if self.k is None else str(self.k)


@dataclass(frozen=True)
class ShortlistKSelection:
    k: int | None
    mean_char_f1: float
    trials: tuple[ShortlistKTrial, ...]

    @property
    def label(self) -> str:
        return "all" if self.k is None else str(self.k)


@dataclass(frozen=True)
class TopicCEPanel:
    """Per-document QA answers plus CE scores for title and topic queries."""

    doc_id: str
    split: str
    question: str
    references: tuple[str, ...]
    chunks: tuple[TextChunk, ...]
    answers: tuple[Answer, ...]
    title_query: str
    topic_query: str
    title_ce_scores: tuple[float, ...]
    topic_ce_scores: tuple[float, ...]
    leaf_id: str
    topic_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        n_chunks = len(self.chunks)
        if len(self.answers) != n_chunks:
            raise ValueError("answers must align with chunks")
        if len(self.title_ce_scores) != n_chunks:
            raise ValueError("title_ce_scores must align with chunks")
        if len(self.topic_ce_scores) != n_chunks:
            raise ValueError("topic_ce_scores must align with chunks")


def k_label(k: int | None) -> str:
    return "all" if k is None else str(int(k))


def ranked_topic_terms(
    terms: Sequence[TopicTerm], *, max_terms: int = DEFAULT_MAX_TOPIC_TERMS
) -> tuple[str, ...]:
    if max_terms < 1:
        raise ValueError("max_terms must be at least 1")
    ordered = sorted(
        terms, key=lambda item: (-float(item.weight), str(item.term).casefold())
    )
    seen: set[str] = set()
    selected: list[str] = []
    for item in ordered:
        term = str(item.term).strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        selected.append(term)
        if len(selected) >= max_terms:
            break
    return tuple(selected)


def topic_ranking_query(
    document: TopicDocument,
    question: str,
    terms: Sequence[str],
) -> str:
    base = ranking_query(document, question)
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = str(term).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    if not cleaned:
        return base
    return f"{base} Topic: {', '.join(cleaned)}."


def shortlist_answers(
    answers: Sequence[Answer],
    scores: Sequence[float],
    k: int | None,
) -> list[Answer]:
    if len(answers) != len(scores):
        raise ValueError("answers and scores must align")
    if not answers:
        return []
    if k is not None:
        if isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer or None")
        if k >= len(answers):
            return list(answers)
    else:
        return list(answers)
    ranked = sorted(
        range(len(answers)),
        key=lambda index: (-float(scores[index]), index),
    )
    return [answers[index] for index in ranked[:k]]


def predict_with_aggregator(
    answers: Sequence[Answer], encoder: object
) -> str | None:
    selected = AnswerAggregator(encoder).aggregate(list(answers))  # type: ignore[arg-type]
    return None if selected is None else selected.text


def predict_from_shortlist(
    panel: TopicCEPanel,
    scores: Sequence[float],
    encoder: object,
    k: int | None,
) -> str | None:
    return predict_with_aggregator(
        shortlist_answers(panel.answers, scores, k), encoder
    )


def collect_topic_ce_panel(
    document: TopicDocument,
    question: str,
    *,
    processor: object,
    answerer: object,
    reranker: PairwiseReranker,
    leaf_terms: Sequence[str] = (),
    leaf_id: str = "",
    references: Sequence[str] = (),
) -> TopicCEPanel:
    chunks = tuple(processor.process(document.text))  # type: ignore[attr-defined]
    title_query = ranking_query(document, question)
    terms = tuple(str(item).strip() for item in leaf_terms if str(item).strip())
    topic_query = topic_ranking_query(document, question, terms)
    title_scores = tuple(
        cross_encoder_chunk_scores(reranker, title_query, chunks)
    )
    if topic_query == title_query:
        topic_scores = title_scores
    else:
        topic_scores = tuple(
            cross_encoder_chunk_scores(reranker, topic_query, chunks)
        )
    found = AnswerFinder(answerer).find(Question(question), chunks)
    by_chunk = {id(answer.chunk): answer for answer in found.answers}
    try:
        answers = tuple(by_chunk[id(chunk)] for chunk in chunks)
    except KeyError as error:
        raise RuntimeError("QA must return one answer per chunk") from error
    return TopicCEPanel(
        doc_id=document.doc_id,
        split=document.split,
        question=question,
        references=tuple(str(item) for item in references if str(item).strip()),
        chunks=chunks,
        answers=answers,
        title_query=title_query,
        topic_query=topic_query,
        title_ce_scores=title_scores,
        topic_ce_scores=topic_scores,
        leaf_id=str(leaf_id or ""),
        topic_terms=terms,
    )


def fit_shortlist_k(
    panels: Sequence[TopicCEPanel],
    encoder: object,
    score_char_f1: Callable[[str | None, Sequence[str]], float],
    scores_of: Callable[[TopicCEPanel], Sequence[float]],
    *,
    grid: Sequence[int | None] = SHORTLIST_K_GRID,
) -> ShortlistKSelection:
    if not panels:
        raise ValueError("shortlist-k fitting requires a non-empty fit split")
    invalid = [item.doc_id for item in panels if item.split.casefold() == "test"]
    if invalid:
        raise ValueError(
            f"shortlist-k fitting refuses test documents: {invalid[:5]}"
        )
    trials: list[ShortlistKTrial] = []
    for k in grid:
        scores = [
            float(
                score_char_f1(
                    predict_from_shortlist(panel, scores_of(panel), encoder, k),
                    panel.references,
                )
            )
            for panel in panels
        ]
        trials.append(
            ShortlistKTrial(k=k, mean_char_f1=sum(scores) / len(scores))
        )
    best = max(
        trials,
        key=lambda item: (
            item.mean_char_f1,
            1 if item.k is None else 0,
            item.k or 0,
        ),
    )
    return ShortlistKSelection(
        k=best.k, mean_char_f1=best.mean_char_f1, trials=tuple(trials)
    )
