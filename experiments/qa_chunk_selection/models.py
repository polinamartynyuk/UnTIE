"""Model wrappers. Production UnTIE loaders are reused; smoke uses lexical stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from untie.config import PipelineConfig
from untie.cross_encoder_ranking import SentenceTransformerCrossEncoder
from untie.models import ExtractiveQuestionAnswerer, ModelFactory, resolve_device

from .ranking import lexical_overlap_scores
from .schema import ChunkRecord


EncodeFn = Callable[[Sequence[str]], np.ndarray]
AnswerFn = Callable[[str, str], dict[str, Any]]


@dataclass
class LexicalReranker:
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, passage in pairs:
            fake = [
                ChunkRecord(
                    chunk_id=0,
                    text=passage,
                    label="easy_negative",
                    primary_class="easy_negative",
                    position=0,
                    section_id="",
                    section_title="",
                    char_start=0,
                    char_end=len(passage),
                    token_start=0,
                    token_end=0,
                    gold_supporting=False,
                    previous_chunk_id=None,
                    next_chunk_id=None,
                )
            ]
            scores.extend(lexical_overlap_scores(query, fake))
        return scores


def hash_encode(texts: Sequence[str], dim: int = 32) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=float)
    for row, text in enumerate(texts):
        for token in str(text).casefold().split():
            matrix[row, hash(token) % dim] += 1.0
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix


def lexical_answer(question: str, context: str) -> dict[str, Any]:
    q_tokens = [token for token in question.casefold().split() if len(token) > 3]
    best = ""
    best_score = -1.0
    words = context.split()
    for width in range(1, min(6, len(words) + 1)):
        for start in range(0, len(words) - width + 1):
            span = " ".join(words[start : start + width])
            overlap = sum(token in span.casefold() for token in q_tokens)
            score = overlap / float(width)
            if score > best_score:
                best_score = score
                best = span
    start = context.find(best) if best else 0
    return {
        "answer": best,
        "score": max(0.0, float(best_score)),
        "start": max(0, start),
        "end": max(0, start) + len(best),
        "start_logit": float(best_score),
        "end_logit": float(best_score),
    }


@dataclass
class ModelBundle:
    device: str
    encode: EncodeFn
    answer: AnswerFn
    relevance_reranker: Any
    qa_answerer: ExtractiveQuestionAnswerer | None
    smoke: bool
    qa_model_name: str
    encoder_name: str
    ce_model_name: str


def load_frozen_qa(*, device: str, smoke: bool = False) -> ModelBundle:
    """Load only the pipeline's frozen extractive QA. No cross-encoder / HF download."""
    if smoke:
        return load_models(device=device, cross_encoder_model="unused", smoke=True)
    resolved = resolve_device(device)
    factory = ModelFactory(PipelineConfig(device=resolved))
    answerer = factory.answerer

    def encode(texts: Sequence[str]) -> np.ndarray:
        return hash_encode(texts)

    def answer(question: str, context: str) -> dict[str, Any]:
        return dict(answerer(question=question, context=context))

    return ModelBundle(
        device=factory.device,
        encode=encode,
        answer=answer,
        relevance_reranker=None,
        qa_answerer=answerer,
        smoke=False,
        qa_model_name=str(factory.config.profile.qa_model),
        encoder_name="not-loaded",
        ce_model_name="not-loaded",
    )


def load_models(
    *,
    device: str,
    cross_encoder_model: str,
    smoke: bool,
) -> ModelBundle:
    resolved = resolve_device(device)
    if smoke:
        reranker = LexicalReranker()

        def encode(texts: Sequence[str]) -> np.ndarray:
            return hash_encode(texts)

        def answer(question: str, context: str) -> dict[str, Any]:
            return lexical_answer(question, context)

        return ModelBundle(
            device="cpu",
            encode=encode,
            answer=answer,
            relevance_reranker=reranker,
            qa_answerer=None,
            smoke=True,
            qa_model_name="lexical-smoke",
            encoder_name="hash-bow",
            ce_model_name="lexical-overlap",
        )

    factory = ModelFactory(PipelineConfig(device=resolved))
    answerer = factory.answerer
    encoder = factory.sentence_encoder
    reranker = SentenceTransformerCrossEncoder(cross_encoder_model, factory.device)

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        matrix = np.asarray(encoder.encode(list(texts), show_progress_bar=False), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    def answer(question: str, context: str) -> dict[str, Any]:
        return dict(answerer(question=question, context=context))

    return ModelBundle(
        device=factory.device,
        encode=encode,
        answer=answer,
        relevance_reranker=reranker,
        qa_answerer=answerer,
        smoke=False,
        qa_model_name=str(factory.config.profile.qa_model),
        encoder_name=str(factory.config.profile.sentence_model),
        ce_model_name=cross_encoder_model,
    )
