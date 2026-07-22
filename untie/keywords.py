from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .protocols import Lemmatizer, SentenceEncoder


def contains_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


def clean_digit_candidates(candidates: Iterable[str]) -> list[str]:
    result: set[str] = set()
    for candidate in candidates:
        words = candidate.split()
        if not contains_digit(candidate):
            result.add(candidate)
        elif len(words) in {3, 4} and not contains_digit(words[0]) and not contains_digit(words[-1]):
            result.add(candidate)
    return sorted(result)


@dataclass
class KeywordExtractor:
    encoder: SentenceEncoder
    lemmatizer: Lemmatizer = lambda word: word.lower()

    def extract(
        self,
        texts: list[str],
        *,
        reference_texts: list[str] | None = None,
        ngram_size: int = 1,
        limit: int = 20,
    ) -> list[str]:
        if not texts:
            return []
        document = " ".join(texts)
        vectorizer = CountVectorizer(
            ngram_range=(ngram_size, ngram_size),
            stop_words="english",
        )
        candidates = clean_digit_candidates(vectorizer.fit([document]).get_feature_names_out())
        if not candidates:
            return []
        reference = " ".join(reference_texts or texts)
        reference_embedding = np.asarray(self.encoder.encode([reference]))
        candidate_embeddings = np.asarray(self.encoder.encode(candidates))
        similarities = cosine_similarity(reference_embedding, candidate_embeddings)[0]
        order = np.argsort(similarities)[::-1][:limit]
        return [candidates[int(index)] for index in order]

    def dynamic_idf_filter(
        self,
        texts: list[str],
        *,
        initial_threshold: float = 1.5,
        min_threshold: float = 0.5,
        target_words_per_document: int = 3,
    ) -> tuple[list[str], float]:
        if not texts:
            return [], 0.0
        normalized = [
            " ".join(self.lemmatizer(word) for word in text.split())
            for text in texts
        ]
        vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=False)
        try:
            vectorizer.fit(normalized)
        except ValueError:
            return normalized, 0.0

        idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        best = normalized
        best_threshold = min_threshold
        best_coverage = -1.0
        threshold = initial_threshold
        step = 0.2
        while threshold >= min_threshold:
            filtered = [
                " ".join(word for word in text.split() if idf.get(word, 0) >= threshold)
                for text in normalized
            ]
            coverage = sum(
                len(text.split()) >= target_words_per_document for text in filtered
            ) / len(filtered)
            if coverage > best_coverage:
                best, best_threshold, best_coverage = filtered, threshold, coverage
            if coverage >= 0.8:
                break
            threshold -= step
            step = max(step * 0.8, 0.05)
        return best, best_threshold


def add_lemma_and_stem(
    keywords: list[dict[str, float | str]],
    *,
    lemmatizer: Lemmatizer = lambda word: word.lower(),
    stemmer: Lemmatizer = lambda word: word.lower(),
) -> list[dict[str, float | str]]:
    return [
        {
            **keyword,
            "lemma": lemmatizer(str(keyword["word"])),
            "stem": stemmer(str(keyword["word"])),
        }
        for keyword in keywords
    ]


def score_keyword_contrast(
    keywords: list[dict[str, float | str]],
    *,
    positive_reference: str,
    negative_reference: str,
    encoder: SentenceEncoder,
) -> list[dict[str, float | str]]:
    if not keywords:
        return []
    words = [str(keyword["word"]) for keyword in keywords]
    word_embeddings = np.asarray(encoder.encode(words))
    positive = np.asarray(encoder.encode([positive_reference]))
    negative = np.asarray(encoder.encode([negative_reference]))
    positive_scores = cosine_similarity(word_embeddings, positive).reshape(-1)
    negative_scores = cosine_similarity(word_embeddings, negative).reshape(-1)
    result = [
        {**keyword, "score_diff": float(positive_scores[index] - negative_scores[index])}
        for index, keyword in enumerate(keywords)
        if positive_scores[index] > negative_scores[index]
    ]
    return sorted(result, key=lambda item: float(item["score_diff"]), reverse=True)
