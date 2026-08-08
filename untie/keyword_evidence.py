"""Сбор и переиспользование evidence для настройки глобальных ключевых слов.

Модуль отделяет дорогие вызовы QA/attention от дешёвого перебора подмножеств:
для каждого документа QA выполняется один раз на всех чанках, после чего
статические словари можно оценивать только пересчётом chunk score и consensus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .domain import Answer, Question, Sentence, TextChunk
from .keywords import KeywordExtractor
from .qa import AnswerAggregator, AnswerFinder, AnswerValidator
from .ranking import WeightedKeyword, score_chunks


_EN_STOP_WORDS: frozenset[str]
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    _EN_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
except ImportError:  # pragma: no cover - sklearn is a core dependency
    _EN_STOP_WORDS = frozenset()

_RU_STOP_WORDS = frozenset(
    {
        "а",
        "без",
        "более",
        "бы",
        "был",
        "была",
        "были",
        "было",
        "быть",
        "в",
        "во",
        "все",
        "для",
        "до",
        "его",
        "ее",
        "если",
        "есть",
        "же",
        "за",
        "и",
        "из",
        "или",
        "их",
        "к",
        "как",
        "на",
        "не",
        "но",
        "о",
        "об",
        "от",
        "по",
        "под",
        "при",
        "с",
        "со",
        "так",
        "то",
        "у",
        "что",
        "чтобы",
        "это",
    }
)


def stable_fingerprint(payload: Any) -> str:
    """Возвращает стабильный SHA-256 для JSON-совместимого payload."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_evidence_fingerprint(
    *,
    doc_id: str,
    text: str,
    aspect_id: str,
    question: str,
    references: Sequence[str],
    collect_candidates: bool,
    fingerprint_payload: Mapping[str, Any] | None = None,
) -> str:
    return stable_fingerprint(
        {
            "doc_id": doc_id,
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "aspect_id": aspect_id,
            "question": question,
            "references": tuple(
                str(reference).strip()
                for reference in references
                if str(reference).strip()
            ),
            "collect_candidates": collect_candidates,
            **dict(fingerprint_payload or {}),
        }
    )


def normalize_term(term: str) -> str:
    """Нормализует поверхностную форму, не выполняя морфологический анализ."""
    return re.sub(r"\s+", " ", str(term).strip().lower())


def language_stop_words(language: str) -> frozenset[str]:
    if language == "en":
        return _EN_STOP_WORDS
    if language == "ru":
        return _RU_STOP_WORDS
    raise ValueError("language must be 'en' or 'ru'")


@dataclass(frozen=True)
class CandidateObservation:
    word: str
    lemma: str
    stem: str
    attention_weight: float
    score_difference: float
    matched_chunk_indices: tuple[int, ...]


@dataclass(frozen=True)
class CachedChunkAnswer:
    index: int
    text: str
    token_count: int
    answer: str
    confidence: float
    start_pos: int | None = None
    end_pos: int | None = None


@dataclass(frozen=True)
class CachedDocumentEvidence:
    doc_id: str
    aspect_id: str
    question: str
    references: tuple[str, ...]
    chunks: tuple[CachedChunkAnswer, ...]
    answer_similarity: tuple[tuple[float, ...], ...]
    baseline_answer: str | None
    baseline_confidence: float | None
    candidates: tuple[CandidateObservation, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CachedDocumentEvidence":
        chunks = tuple(
            CachedChunkAnswer(
                index=int(item["index"]),
                text=str(item["text"]),
                token_count=int(item["token_count"]),
                answer=str(item["answer"]),
                confidence=float(item["confidence"]),
                start_pos=item.get("start_pos"),
                end_pos=item.get("end_pos"),
            )
            for item in payload.get("chunks", [])
        )
        candidates = tuple(
            CandidateObservation(
                word=str(item["word"]),
                lemma=str(item["lemma"]),
                stem=str(item["stem"]),
                attention_weight=float(item["attention_weight"]),
                score_difference=float(item["score_difference"]),
                matched_chunk_indices=tuple(
                    int(index) for index in item.get("matched_chunk_indices", [])
                ),
            )
            for item in payload.get("candidates", [])
        )
        return cls(
            doc_id=str(payload["doc_id"]),
            aspect_id=str(payload["aspect_id"]),
            question=str(payload["question"]),
            references=tuple(str(item) for item in payload.get("references", [])),
            chunks=chunks,
            answer_similarity=tuple(
                tuple(float(value) for value in row)
                for row in payload.get("answer_similarity", [])
            ),
            baseline_answer=payload.get("baseline_answer"),
            baseline_confidence=(
                float(payload["baseline_confidence"])
                if payload.get("baseline_confidence") is not None
                else None
            ),
            candidates=candidates,
            fingerprint=str(payload["fingerprint"]),
        )


class EvidenceStore:
    """Файловый JSON-кэш с проверкой fingerprint и атомарной записью."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, doc_id: str) -> Path:
        safe_name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_.-]+", "_", doc_id)
        suffix = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:10]
        return self.root / f"{safe_name[:80]}-{suffix}.json"

    def load(
        self, doc_id: str, *, expected_fingerprint: str | None = None
    ) -> CachedDocumentEvidence | None:
        path = self.path_for(doc_id)
        if not path.is_file():
            return None
        evidence = CachedDocumentEvidence.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if (
            expected_fingerprint is not None
            and evidence.fingerprint != expected_fingerprint
        ):
            return None
        return evidence

    def save(self, evidence: CachedDocumentEvidence) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(evidence.doc_id)
        handle, temporary_name = tempfile.mkstemp(
            dir=self.root, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    evidence.to_dict(),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination


class ExtractionMetricCache:
    """Персистентный кэш метрик для конечного набора QA-ответов."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._values: dict[str, dict[str, float]] = {
                str(key): {
                    str(name): float(value)
                    for name, value in metrics.items()
                }
                for key, metrics in raw.items()
            }
        else:
            self._values = {}

    @staticmethod
    def key(
        prediction: str | None,
        references: Sequence[str],
        *,
        language: str,
        include_bertscore: bool,
        metric_version: str = "extraction-v1",
    ) -> str:
        return stable_fingerprint(
            {
                "prediction": prediction or "",
                "references": list(references),
                "language": language,
                "include_bertscore": include_bertscore,
                "metric_version": metric_version,
            }
        )

    def score(
        self,
        prediction: str | None,
        references: Sequence[str],
        *,
        language: str,
        include_bertscore: bool = False,
    ) -> dict[str, float]:
        key = self.key(
            prediction,
            references,
            language=language,
            include_bertscore=include_bertscore,
        )
        if key not in self._values:
            from .extraction_metrics import compute_row_metrics

            self._values[key] = compute_row_metrics(
                prediction,
                list(references),
                lang=language,
                include_bertscore=include_bertscore,
            )
        return dict(self._values[key])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(
                    self._values,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def _allowed_terms(text: str, language: str) -> set[str]:
    pattern = r"\b[а-яёa-z][а-яёa-z-]*\b" if language == "ru" else r"\b[\w-]+\b"
    return set(re.findall(pattern, text.lower()))


def _merge_attention_batches(
    batches: Sequence[Sequence[dict[str, float | str]]],
    filtered_chunks: Sequence[str],
    *,
    language: str,
) -> list[dict[str, float | str]]:
    stop_words = language_stop_words(language)
    merged: dict[str, dict[str, float | str]] = {}
    for batch, filtered_text in zip(batches, filtered_chunks):
        allowed = _allowed_terms(filtered_text, language)
        for item in batch:
            word = normalize_term(str(item["word"]))
            if len(word) <= 1 or word in stop_words or word not in allowed:
                continue
            if word not in merged or float(item.get("weight", 0.0)) > float(
                merged[word].get("weight", 0.0)
            ):
                merged[word] = {**item, "word": word}
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("weight", 0.0)),
        reverse=True,
    )


def _contrast_against_references(
    candidates: Sequence[dict[str, float | str]],
    *,
    aspect_reference: str,
    references: Sequence[str],
    encoder: Any,
) -> list[dict[str, float | str]]:
    """Оставляет слова, более близкие аспекту, чем любому gold-ответу.

    Функция вызывается только при сборе train evidence. На validation/test
    статический словарь уже фиксирован и reference не используется.
    """
    if not candidates:
        return []
    words = [str(item["word"]) for item in candidates]
    word_embeddings = np.asarray(encoder.encode(words))
    aspect_embedding = np.asarray(encoder.encode([aspect_reference]))
    positive = cosine_similarity(word_embeddings, aspect_embedding).reshape(-1)
    if references:
        reference_embeddings = np.asarray(encoder.encode(list(references)))
        negative = cosine_similarity(word_embeddings, reference_embeddings).max(axis=1)
    else:
        negative = np.zeros(len(words), dtype=float)
    result = [
        {**item, "score_diff": float(positive[index] - negative[index])}
        for index, item in enumerate(candidates)
        if positive[index] > negative[index]
    ]
    return sorted(
        result, key=lambda item: float(item["score_diff"]), reverse=True
    )


def _unique_valid_answers(
    answers: list[Answer],
    references: Sequence[str],
    validator: AnswerValidator,
) -> list[Answer]:
    selected: dict[tuple[str, str], Answer] = {}
    for reference in references:
        for answer in validator.validate(answers, reference, top_k=3):
            key = (answer.text, answer.chunk.text)
            selected[key] = answer
    return list(selected.values())


def collect_document_evidence(
    *,
    doc_id: str,
    text: str,
    aspect_id: str,
    aspect_reference: str,
    question: str,
    references: Sequence[str],
    processor: Any,
    answerer: Any,
    encoder: Any,
    attention_extractor: Callable[
        [str, str], list[dict[str, float | str]]
    ],
    language: str,
    lemmatize: Callable[[str], str] = str.lower,
    stem: Callable[[str], str] = str.lower,
    strict_answer_threshold: float = 0.9,
    min_answer_threshold: float = 0.7,
    keyword_idf_threshold: float = 1.5,
    collect_candidates: bool = True,
    fingerprint_payload: dict[str, Any] | None = None,
) -> CachedDocumentEvidence:
    """Выполняет дорогой train-only сбор evidence для одного документа."""
    clean_references = tuple(
        str(reference).strip() for reference in references if str(reference).strip()
    )
    chunks = processor.process(text)
    question_result = AnswerFinder(answerer).find(Question(question), chunks)
    answers = question_result.answers
    baseline = AnswerAggregator(encoder).aggregate(answers)

    contrasted: list[dict[str, float | str]] = []
    if collect_candidates:
        validator = AnswerValidator(
            encoder,
            strict_threshold=strict_answer_threshold,
            min_threshold=min_answer_threshold,
        )
        valid_answers = _unique_valid_answers(answers, clean_references, validator)
        valid_chunks = [answer.chunk.text for answer in valid_answers]
        raw_batches = [
            attention_extractor(question, answer.chunk.text) for answer in valid_answers
        ]
        filtered_chunks, _ = KeywordExtractor(
            encoder, lemmatizer=lemmatize
        ).dynamic_idf_filter(
            valid_chunks,
            initial_threshold=keyword_idf_threshold,
        )
        merged = _merge_attention_batches(
            raw_batches, filtered_chunks, language=language
        )
        contrasted = _contrast_against_references(
            merged,
            aspect_reference=aspect_reference,
            references=clean_references,
            encoder=encoder,
        )

    chunk_index_by_text: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        chunk_index_by_text.setdefault(chunk.text, []).append(index)
    observations = tuple(
        CandidateObservation(
            word=str(item["word"]),
            lemma=lemmatize(str(item["word"])),
            stem=stem(str(item["word"])),
            attention_weight=float(item.get("weight", 1.0)),
            score_difference=float(item.get("score_diff", 0.0)),
            matched_chunk_indices=tuple(
                index
                for chunk_text, indices in chunk_index_by_text.items()
                if re.search(
                    rf"\b{re.escape(str(item['word']).lower())}\b",
                    chunk_text.lower(),
                )
                for index in indices
            ),
        )
        for item in contrasted
    )

    answers_by_chunk = {answer.chunk.text: answer for answer in answers}
    cached_chunks = tuple(
        CachedChunkAnswer(
            index=index,
            text=chunk.text,
            token_count=chunk.token_count,
            answer=answers_by_chunk[chunk.text].text,
            confidence=answers_by_chunk[chunk.text].confidence,
            start_pos=answers_by_chunk[chunk.text].start_pos,
            end_pos=answers_by_chunk[chunk.text].end_pos,
        )
        for index, chunk in enumerate(chunks)
    )
    if answers:
        answer_embeddings = np.asarray(
            encoder.encode([cached.answer for cached in cached_chunks])
        )
        similarity = cosine_similarity(answer_embeddings)
    else:
        similarity = np.empty((0, 0), dtype=float)

    fingerprint = build_evidence_fingerprint(
        doc_id=doc_id,
        text=text,
        aspect_id=aspect_id,
        question=question,
        references=clean_references,
        collect_candidates=collect_candidates,
        fingerprint_payload=fingerprint_payload,
    )
    return CachedDocumentEvidence(
        doc_id=doc_id,
        aspect_id=aspect_id,
        question=question,
        references=clean_references,
        chunks=cached_chunks,
        answer_similarity=tuple(
            tuple(float(value) for value in row) for row in similarity
        ),
        baseline_answer=baseline.text if baseline else None,
        baseline_confidence=baseline.confidence if baseline else None,
        candidates=observations,
        fingerprint=fingerprint,
    )


def _rebuild_chunks(evidence: CachedDocumentEvidence) -> list[TextChunk]:
    return [
        TextChunk(
            sentences=(
                Sentence(
                    text=item.text,
                    number=item.index,
                    tokens=tuple(item.text.split()),
                ),
            ),
            text=item.text,
            token_count=item.token_count,
        )
        for item in evidence.chunks
    ]


def _cluster_indices(
    similarity: np.ndarray, threshold: float
) -> list[list[int]]:
    clusters: list[list[int]] = []
    visited: set[int] = set()
    for index in range(len(similarity)):
        if index in visited:
            continue
        cluster = [index]
        visited.add(index)
        for other in range(index + 1, len(similarity)):
            if other not in visited and similarity[index, other] >= threshold:
                cluster.append(other)
                visited.add(other)
        clusters.append(cluster)
    return clusters


def select_cached_answer(
    answers: list[Answer],
    similarity: np.ndarray,
    *,
    similarity_threshold: float,
    cluster_strategy: str,
    answer_strategy: str,
) -> Answer | None:
    """Эквивалент ``AnswerConsensus.select_clustered`` без encoder-вызова."""
    if not answers:
        return None
    if len(answers) == 1:
        return answers[0]
    clusters = _cluster_indices(similarity, similarity_threshold)
    metrics: list[dict[str, Any]] = []
    for indices in clusters:
        chunk_scores = [
            float(answers[index].metadata.get("chunk_score", 0.0))
            for index in indices
        ]
        pairs = [
            float(similarity[left, right])
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ]
        average = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        metrics.append(
            {
                "indices": indices,
                "average_chunk_score": average,
                "weighted_score": len(indices) * average,
                "cohesion": float(np.mean(pairs)) if pairs else 0.0,
            }
        )
    selectors = {
        "highest_avg_score": lambda item: item["average_chunk_score"],
        "weighted_score": lambda item: item["weighted_score"],
        "highest_cohesion": lambda item: item["cohesion"],
    }
    if cluster_strategy not in selectors:
        raise ValueError(f"Unknown cluster strategy: {cluster_strategy}")
    selected = max(metrics, key=selectors[cluster_strategy])
    indices = selected["indices"]

    def answer_score(index: int) -> float:
        chunk_score = float(answers[index].metadata.get("chunk_score", 0.0))
        peers = [other for other in indices if other != index]
        similarity_score = (
            float(np.mean([similarity[index, other] for other in peers]))
            if peers
            else 0.0
        )
        if answer_strategy == "highest_chunk_score":
            return chunk_score
        if answer_strategy == "highest_similarity":
            return similarity_score
        if answer_strategy == "combined_score":
            return 0.5 * chunk_score + 0.5 * similarity_score
        raise ValueError(f"Unknown answer strategy: {answer_strategy}")

    best_index = max(indices, key=answer_score)
    return answers[best_index]


def rerank_cached_document(
    evidence: CachedDocumentEvidence,
    keywords: Sequence[WeightedKeyword],
    *,
    weight_ratio: float = 0.5,
    similarity_threshold: float = 0.75,
    cluster_strategy: str = "weighted_score",
    answer_strategy: str = "combined_score",
) -> tuple[str | None, dict[str, Any]]:
    """Дешёво оценивает статический словарь поверх кэшированных QA-ответов."""
    chunks = _rebuild_chunks(evidence)
    scored = score_chunks(chunks, list(keywords), weight_ratio=weight_ratio)
    if not scored:
        return evidence.baseline_answer, {
            "fallback": True,
            "matched_chunks": 0,
            "matched_keywords": [],
        }

    cached_by_index = {item.index: item for item in evidence.chunks}
    answers: list[Answer] = []
    original_indices: list[int] = []
    for scored_chunk in scored:
        index = scored_chunk.chunk.start_sentence
        cached = cached_by_index[index]
        answers.append(
            Answer(
                text=cached.answer,
                chunk=scored_chunk.chunk,
                confidence=cached.confidence,
                start_pos=cached.start_pos,
                end_pos=cached.end_pos,
                metadata={
                    "chunk_score": scored_chunk.score,
                    "matched_keywords": scored_chunk.matched_keywords,
                    "keyword_scores": scored_chunk.keyword_scores,
                },
            )
        )
        original_indices.append(index)
    full_similarity = np.asarray(evidence.answer_similarity, dtype=float)
    subset_similarity = full_similarity[np.ix_(original_indices, original_indices)]
    selected = select_cached_answer(
        answers,
        subset_similarity,
        similarity_threshold=similarity_threshold,
        cluster_strategy=cluster_strategy,
        answer_strategy=answer_strategy,
    )
    return (
        selected.text if selected else evidence.baseline_answer,
        {
            "fallback": selected is None,
            "matched_chunks": len(scored),
            "matched_keywords": sorted(
                {
                    keyword
                    for item in scored
                    for keyword in item.matched_keywords
                }
            ),
        },
    )
