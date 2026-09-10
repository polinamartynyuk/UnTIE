"""Document-level chunk ranking with a cross-encoder and optional static keywords.

Gold is used only to select mix weights on train/dev. Test inference applies
frozen weights and never reads references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence

from .domain import Answer, Question, ScoredChunk, TextChunk
from .qa import AnswerConsensus, ScoredAnswerFinder
from .ranking import WeightedKeyword, mix_score_channels, score_chunks
from .topics import TopicDocument, document_topic_representation


MIX_WEIGHT_GRID: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
)


class PairwiseReranker(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Return one score per (query, passage) pair, same order as ``pairs``."""


@dataclass(frozen=True)
class MixWeights:
    cross_encoder: float
    static_keywords: float

    @property
    def name(self) -> str:
        return f"ce={self.cross_encoder:.2f}+kw={self.static_keywords:.2f}"


@dataclass(frozen=True)
class MixWeightTrial:
    weights: MixWeights
    mean_char_f1: float


@dataclass(frozen=True)
class MixWeightSelection:
    weights: MixWeights
    mean_char_f1: float
    trials: tuple[MixWeightTrial, ...]


@dataclass(frozen=True)
class RankedDocumentPanel:
    """Per-document QA answers plus both ranking channels, aligned by chunk."""

    doc_id: str
    split: str
    question: str
    query: str
    references: tuple[str, ...]
    chunks: tuple[TextChunk, ...]
    answers: tuple[Answer, ...]
    cross_encoder_scores: tuple[float, ...]
    keyword_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        n_chunks = len(self.chunks)
        if len(self.answers) != n_chunks:
            raise ValueError("answers must align with chunks")
        if len(self.cross_encoder_scores) != n_chunks:
            raise ValueError("cross_encoder_scores must align with chunks")
        if len(self.keyword_scores) != n_chunks:
            raise ValueError("keyword_scores must align with chunks")


_BODY_DOC_PREFIX = re.compile(r"^(?:document|title)\s*:\s*", re.IGNORECASE)
_BODY_SECTION_CUT = re.compile(
    r"\b(?:section\s*:?\s*abstract|abstract\s*:)\b",
    re.IGNORECASE,
)


def _clip_words(text: str, max_words: int) -> str:
    words = [item for item in str(text).split() if item]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return parts[0].strip()


def _compact_body_identity(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = _BODY_DOC_PREFIX.sub("", cleaned)
    cut = _BODY_SECTION_CUT.search(cleaned)
    if cut is not None and cut.start() >= 8:
        heading = cleaned[: cut.start()].strip(" .:;-")
        if heading:
            return heading
    return _first_sentence(cleaned)


def document_identity_for_query(
    document: TopicDocument, *, max_words: int = 40
) -> str:
    """Paper identity for a CE query, aligned with topic-routing fallbacks.

    Explicit title/abstract/headings win. If those fields are empty, the same
    body representation as routing is compacted to a heading or first sentences
    so MiniLM is not dominated by leaf terms alone.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    title = str(document.title or "").strip()
    if title:
        return _clip_words(title, max_words)
    abstract = str(document.abstract or "").strip()
    if abstract:
        return _clip_words(_first_sentence(abstract), max_words)
    heading = next(
        (str(item).strip() for item in (document.headings or ()) if str(item).strip()),
        "",
    )
    if heading:
        return _clip_words(heading, max_words)
    source = document_topic_representation(document)
    return _clip_words(_compact_body_identity(source), max_words)


def ranking_query(
    document: TopicDocument, question: str, *, max_identity_words: int = 40
) -> str:
    """Build the CE query from paper identity plus the QA question."""
    asked = str(question or "").strip()
    identity = document_identity_for_query(
        document, max_words=max_identity_words
    )
    if identity and asked:
        return f"{identity.rstrip(' .')}. {asked}"
    return identity or asked


def cross_encoder_chunk_scores(
    reranker: PairwiseReranker,
    query: str,
    chunks: Sequence[TextChunk],
) -> list[float]:
    if not chunks:
        return []
    prompt = str(query or "").strip() or "Which task was solved?"
    pairs = [(prompt, chunk.text) for chunk in chunks]
    raw = list(reranker.predict(pairs))
    if len(raw) != len(chunks):
        raise ValueError("cross-encoder must return one score per chunk")
    return [float(value) for value in raw]


def keyword_chunk_scores(
    chunks: Sequence[TextChunk],
    keywords: Sequence[WeightedKeyword],
    *,
    weight_ratio: float = 0.0,
) -> list[float]:
    """Static keyword scores without dropping unmatched chunks."""
    if not chunks:
        return []
    scored = score_chunks(
        list(chunks),
        list(keywords),
        weight_ratio=weight_ratio,
        keep_unmatched=True,
        unmatched_floor=0.0,
    )
    by_id = {id(item.chunk): float(item.score) for item in scored}
    missing = [chunk for chunk in chunks if id(chunk) not in by_id]
    if missing:
        raise RuntimeError("keyword scoring lost chunks under keep_unmatched")
    return [by_id[id(chunk)] for chunk in chunks]


def mixed_chunk_scores(
    cross_encoder_scores: Sequence[float],
    keyword_scores: Sequence[float],
    weights: MixWeights,
) -> list[float]:
    return mix_score_channels(
        (cross_encoder_scores, keyword_scores),
        (weights.cross_encoder, weights.static_keywords),
    )


def collect_ranked_panel(
    document: TopicDocument,
    question: str,
    *,
    processor: object,
    answerer: object,
    reranker: PairwiseReranker,
    keywords: Sequence[WeightedKeyword],
    keyword_weight_ratio: float,
    references: Sequence[str] = (),
) -> RankedDocumentPanel:
    chunks = tuple(processor.process(document.text))  # type: ignore[attr-defined]
    query = ranking_query(document, question)
    ce_scores = tuple(cross_encoder_chunk_scores(reranker, query, chunks))
    kw_scores = tuple(
        keyword_chunk_scores(
            chunks, keywords, weight_ratio=keyword_weight_ratio
        )
    )
    scored = [ScoredChunk(chunk=chunk, score=0.0) for chunk in chunks]
    found = ScoredAnswerFinder(answerer).find(Question(question), scored)
    by_chunk = {id(answer.chunk): answer for answer in found.answers}
    try:
        answers = tuple(by_chunk[id(chunk)] for chunk in chunks)
    except KeyError as error:
        raise RuntimeError("QA must return one answer per chunk") from error
    return RankedDocumentPanel(
        doc_id=document.doc_id,
        split=document.split,
        question=question,
        query=query,
        references=tuple(str(item) for item in references if str(item).strip()),
        chunks=chunks,
        answers=answers,
        cross_encoder_scores=ce_scores,
        keyword_scores=kw_scores,
    )


def answers_with_mixed_scores(
    panel: RankedDocumentPanel, weights: MixWeights
) -> list[Answer]:
    mixed = mixed_chunk_scores(
        panel.cross_encoder_scores, panel.keyword_scores, weights
    )
    updated: list[Answer] = []
    for answer, score, ce_score, kw_score in zip(
        panel.answers,
        mixed,
        panel.cross_encoder_scores,
        panel.keyword_scores,
    ):
        metadata = dict(answer.metadata)
        metadata["chunk_score"] = float(score)
        metadata["ce_score"] = float(ce_score)
        metadata["kw_score"] = float(kw_score)
        updated.append(replace(answer, metadata=metadata))
    return updated


def predict_from_panel(
    panel: RankedDocumentPanel,
    weights: MixWeights,
    encoder: object,
    *,
    similarity_threshold: float = 0.75,
    cluster_strategy: str = "weighted_score",
    answer_strategy: str = "combined_score",
) -> str | None:
    answers = answers_with_mixed_scores(panel, weights)
    selected = AnswerConsensus(encoder).select_clustered(  # type: ignore[arg-type]
        answers,
        similarity_threshold=similarity_threshold,
        cluster_strategy=cluster_strategy,  # type: ignore[arg-type]
        answer_strategy=answer_strategy,  # type: ignore[arg-type]
    )
    return None if selected is None else selected.text


def fit_mix_weights(
    panels: Sequence[RankedDocumentPanel],
    encoder: object,
    score_char_f1: Callable[[str | None, Sequence[str]], float],
    *,
    grid: Sequence[tuple[float, float]] = MIX_WEIGHT_GRID,
    similarity_threshold: float = 0.75,
    cluster_strategy: str = "weighted_score",
    answer_strategy: str = "combined_score",
) -> MixWeightSelection:
    """Pick mix weights on a gold split. ``panels`` must not be the test set."""
    if not panels:
        raise ValueError("mix-weight fitting requires a non-empty fit split")
    invalid = [item.doc_id for item in panels if item.split.casefold() == "test"]
    if invalid:
        raise ValueError(
            "mix-weight fitting refuses test documents: "
            f"{invalid[:5]}"
        )
    trials: list[MixWeightTrial] = []
    for cross_encoder, static_keywords in grid:
        weights = MixWeights(cross_encoder, static_keywords)
        scores = []
        for panel in panels:
            prediction = predict_from_panel(
                panel,
                weights,
                encoder,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            scores.append(float(score_char_f1(prediction, panel.references)))
        trials.append(
            MixWeightTrial(weights=weights, mean_char_f1=sum(scores) / len(scores))
        )
    best = max(
        trials,
        key=lambda item: (
            item.mean_char_f1,
            item.weights.cross_encoder,
            -item.weights.static_keywords,
        ),
    )
    return MixWeightSelection(
        weights=best.weights,
        mean_char_f1=best.mean_char_f1,
        trials=tuple(trials),
    )


class SentenceTransformerCrossEncoder:
    """Lazy wrapper around ``sentence_transformers.CrossEncoder``."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers to use the cross-encoder experiment"
            ) from error
        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        import numpy as np

        scores = self._model.predict(list(pairs), show_progress_bar=False)
        return [float(value) for value in np.asarray(scores).reshape(-1)]
