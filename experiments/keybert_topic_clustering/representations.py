"""E0 TF-IDF+SVD, E1 document embedding, E2 KeyBERT-only, E3 hybrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .documents import PaperDocument
from .encoder import CachedSentenceEncoder, l2_normalize
from .keyphrases import Keyphrase, keyphrase_matrix


@dataclass(frozen=True)
class RepresentationSpec:
    family: str
    text_field: str = "R2"
    alpha: float | None = None
    keyphrase_config: str = ""

    @property
    def name(self) -> str:
        parts = [self.family, self.text_field]
        if self.alpha is not None:
            parts.append(f"a{self.alpha:.2f}")
        if self.keyphrase_config:
            parts.append(self.keyphrase_config)
        return "+".join(parts)

    @property
    def uses_keybert(self) -> bool:
        return self.family in {"E2", "E3"}


class TfidfSvdFit:
    def __init__(self, vectorizer: TfidfVectorizer, svd: TruncatedSVD) -> None:
        self.vectorizer = vectorizer
        self.svd = svd

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(list(texts))
        dense = self.svd.transform(sparse)
        return l2_normalize(np.asarray(dense, dtype=np.float32))


def fit_e0(train_texts: Sequence[str], *, max_features: int = 8000, components: int = 100) -> TfidfSvdFit:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    sparse = vectorizer.fit_transform(list(train_texts))
    n_components = min(components, max(2, min(sparse.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(sparse)
    return TfidfSvdFit(vectorizer, svd)


def document_matrix(
    papers: Sequence[PaperDocument],
    encoder: CachedSentenceEncoder,
    *,
    text_field: str,
) -> np.ndarray:
    texts = [paper.representation(text_field) for paper in papers]
    return l2_normalize(encoder.encode(texts))


def hybrid_matrix(document_vectors: np.ndarray, keyphrase_vectors: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    mixed = alpha * document_vectors + (1.0 - alpha) * keyphrase_vectors
    return l2_normalize(mixed)


def build_matrix(
    papers: Sequence[PaperDocument],
    spec: RepresentationSpec,
    *,
    encoder: CachedSentenceEncoder | None = None,
    e0: TfidfSvdFit | None = None,
    phrases: dict[str, tuple[Keyphrase, ...]] | None = None,
) -> np.ndarray:
    texts = [paper.representation(spec.text_field) for paper in papers]
    if spec.family == "E0":
        if e0 is None:
            raise ValueError("E0 requires a train-fit TF-IDF/SVD transformer")
        return e0.transform(texts)
    if encoder is None:
        raise ValueError(f"{spec.family} requires an encoder")
    document_vectors = document_matrix(papers, encoder, text_field=spec.text_field)
    if spec.family == "E1":
        return document_vectors
    if phrases is None:
        raise ValueError(f"{spec.family} requires KeyBERT phrases")
    phrase_vectors = keyphrase_matrix(papers, phrases, encoder)
    if spec.family == "E2":
        return phrase_vectors
    if spec.family == "E3":
        if spec.alpha is None:
            raise ValueError("E3 requires alpha")
        return hybrid_matrix(document_vectors, phrase_vectors, spec.alpha)
    raise ValueError(f"unknown family {spec.family}")
