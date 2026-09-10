"""KeyBERT keyphrase extraction with on-disk cache."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .documents import PaperDocument
from .encoder import CachedSentenceEncoder


@dataclass(frozen=True)
class Keyphrase:
    phrase: str
    weight: float


@dataclass(frozen=True)
class KeyphraseConfig:
    ngram_low: int = 1
    ngram_high: int = 3
    top_n: int = 10
    use_mmr: bool = False
    diversity: float = 0.5

    @property
    def key(self) -> str:
        mmr = f"mmr{self.diversity:.2f}" if self.use_mmr else "nommr"
        return f"ng{self.ngram_low}-{self.ngram_high}_top{self.top_n}_{mmr}"


def _cache_path(cache_dir: Path, config: KeyphraseConfig) -> Path:
    return cache_dir / f"document_keyphrases_{config.key}.json"


def extract_document_keyphrases(
    papers: Sequence[PaperDocument],
    encoder: CachedSentenceEncoder,
    config: KeyphraseConfig,
    *,
    cache_dir: Path,
    representation: str = "R2",
    extract_top_n: int | None = None,
) -> dict[str, tuple[Keyphrase, ...]]:
    """Extract KeyBERT phrases. ``extract_top_n`` may exceed config.top_n for reuse."""
    from keybert import KeyBERT

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, config)
    cached: dict[str, list[list[object]]] = {}
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
    needed = [paper for paper in papers if paper.doc_id not in cached]
    timings = {"keybert_candidate_time": 0.0, "keybert_ranking_time": 0.0}
    if needed:
        backend = KeyBERT(model=encoder.model)
        top_n = int(extract_top_n or max(config.top_n, 20))
        started = time.perf_counter()
        texts = [paper.representation(representation) for paper in needed]
        doc_embeddings = encoder.encode(texts)
        timings["keybert_candidate_time"] = time.perf_counter() - started
        started = time.perf_counter()
        for paper, text, embedding in zip(needed, texts, doc_embeddings):
            raw = backend.extract_keywords(
                [text],
                keyphrase_ngram_range=(config.ngram_low, config.ngram_high),
                stop_words="english",
                top_n=top_n,
                use_mmr=config.use_mmr,
                diversity=config.diversity,
                doc_embeddings=embedding.reshape(1, -1),
            )
            items = raw[0] if raw and isinstance(raw[0], list) else raw
            cached[paper.doc_id] = [
                [str(phrase), float(weight)] for phrase, weight in items
            ]
        timings["keybert_ranking_time"] = time.perf_counter() - started
        path.write_text(json.dumps(cached, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sliced: dict[str, tuple[Keyphrase, ...]] = {}
    for paper in papers:
        items = cached.get(paper.doc_id, [])[: config.top_n]
        sliced[paper.doc_id] = tuple(
            Keyphrase(str(phrase), float(weight)) for phrase, weight in items if str(phrase).strip()
        )
    (cache_dir / f"keybert_timings_{config.key}.json").write_text(
        json.dumps(timings, indent=2) + "\n", encoding="utf-8"
    )
    return sliced


def keyphrase_matrix(
    papers: Sequence[PaperDocument],
    phrases: dict[str, tuple[Keyphrase, ...]],
    encoder: CachedSentenceEncoder,
) -> np.ndarray:
    """Weighted mean of phrase embeddings, L2-normalized. Empty docs → zero then skip later."""
    from .encoder import l2_normalize

    rows = []
    for paper in papers:
        items = phrases.get(paper.doc_id, ())
        if not items:
            rows.append(np.zeros(encoder.embedding_dimension, dtype=np.float32))
            continue
        vectors = encoder.encode([item.phrase for item in items])
        weights = np.asarray([max(item.weight, 0.0) for item in items], dtype=np.float32)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        mixed = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
        rows.append(mixed)
    matrix = np.vstack(rows).astype(np.float32)
    nonempty = np.linalg.norm(matrix, axis=1) > 0
    if nonempty.any():
        matrix[nonempty] = l2_normalize(matrix[nonempty])
    return matrix
