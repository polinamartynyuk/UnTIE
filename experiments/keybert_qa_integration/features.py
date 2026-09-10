"""Train-only global frequencies and KeyBERT phrase helpers."""

from __future__ import annotations

import json
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.encoder import CachedSentenceEncoder
from experiments.keybert_topic_clustering.keyphrases import Keyphrase, KeyphraseConfig
from untie.ranking import WeightedKeyword


def global_keyword_frequency(
    keywords_by_doc: Mapping[str, Sequence[WeightedKeyword]],
    train_ids: Sequence[str],
) -> dict[str, float]:
    """Document frequency on train only, scaled to [0, 1] by n_train."""
    counts: Counter[str] = Counter()
    n_train = 0
    for doc_id in train_ids:
        items = keywords_by_doc.get(doc_id)
        if items is None:
            continue
        n_train += 1
        seen = {
            (item.lemma or item.word).casefold()
            for item in items
            if (item.lemma or item.word).strip()
        }
        counts.update(seen)
    if n_train <= 0:
        return {}
    return {word: count / n_train for word, count in counts.items()}


def encode_keywords(
    keywords: Sequence[WeightedKeyword],
    encoder: CachedSentenceEncoder,
) -> np.ndarray:
    if not keywords:
        return np.zeros((0, encoder.embedding_dimension), dtype=np.float32)
    return encoder.encode([item.word for item in keywords])


def extract_text_keyphrases(
    texts: Mapping[str, str],
    encoder: CachedSentenceEncoder,
    config: KeyphraseConfig,
    *,
    cache_path: Path,
    extract_top_n: int | None = None,
) -> tuple[dict[str, tuple[Keyphrase, ...]], float]:
    """KeyBERT over arbitrary texts, cached by id. Used for chunk-level phrases."""
    from keybert import KeyBERT

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, list[list[object]]] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    needed = [key for key in texts if key not in cached]
    started = time.perf_counter()
    if needed:
        backend = KeyBERT(model=encoder.model)
        top_n = int(extract_top_n or config.top_n)
        payload_texts = [texts[key] for key in needed]
        embeddings = encoder.encode(payload_texts)
        for key, text, embedding in zip(needed, payload_texts, embeddings):
            if not str(text).strip():
                cached[key] = []
                continue
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
            cached[key] = [[str(phrase), float(weight)] for phrase, weight in items]
        cache_path.write_text(
            json.dumps(cached, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elapsed = time.perf_counter() - started
    sliced: dict[str, tuple[Keyphrase, ...]] = {}
    for key in texts:
        items = cached.get(key, [])[: config.top_n]
        sliced[key] = tuple(
            Keyphrase(str(phrase), float(weight))
            for phrase, weight in items
            if str(phrase).strip()
        )
    return sliced, elapsed


def chunk_question_keybert_scores(
    chunk_phrases: Sequence[Sequence[Keyphrase]],
    encoder: CachedSentenceEncoder,
    question: str,
) -> np.ndarray:
    """Chunk-level prior: max cosine of the chunk's own KeyBERT phrases to the question."""
    question_vector = encoder.encode([question])[0]
    scores = np.zeros((len(chunk_phrases),), dtype=np.float32)
    for index, phrases in enumerate(chunk_phrases):
        if not phrases:
            continue
        embeddings = encoder.encode([item.phrase for item in phrases])
        scores[index] = float(np.clip((embeddings @ question_vector).max(), 0.0, 1.0))
    return scores


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
