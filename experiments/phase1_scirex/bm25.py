"""Okapi BM25 over the eligible chunks of one document."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).casefold())


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    corpus = [tokenize(item) for item in documents]
    query_tokens = tokenize(query)
    if not query_tokens or not corpus:
        return [0.0] * len(documents)
    n_docs = len(corpus)
    avgdl = sum(len(item) for item in corpus) / n_docs
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))
    scores: list[float] = []
    query_tf = Counter(query_tokens)
    for doc in corpus:
        dl = len(doc) or 1
        tf = Counter(doc)
        score = 0.0
        for term, qtf in query_tf.items():
            n_q = df.get(term, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            freq = tf.get(term, 0)
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            if denom <= 0:
                continue
            score += idf * freq * (k1 + 1.0) * qtf / denom
        scores.append(float(score))
    return scores
