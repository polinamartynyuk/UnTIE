from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Sentence:
    text: str
    number: int
    tokens: tuple[str, ...] = ()
    token_ids: tuple[int, ...] = ()

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class TextChunk:
    sentences: tuple[Sentence, ...]
    text: str
    token_count: int
    embedding: NDArray[np.floating[Any]] | None = field(default=None, compare=False)

    @property
    def start_sentence(self) -> int:
        return self.sentences[0].number

    @property
    def end_sentence(self) -> int:
        return self.sentences[-1].number


@dataclass
class Answer:
    text: str
    chunk: TextChunk
    confidence: float
    start_pos: int | None = None
    end_pos: int | None = None
    similarity_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    text: str
    answers: list[Answer] = field(default_factory=list)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: TextChunk
    score: float
    matched_keywords: tuple[str, ...] = ()
    keyword_scores: dict[str, float] = field(default_factory=dict)
    original_weights: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    confidence: float
    supporting_answers: tuple[Answer, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    final_answer: FinalAnswer | None
    questions: tuple[Question, ...]
    used_chunks: tuple[TextChunk, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
