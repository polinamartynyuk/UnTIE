"""Uniform candidate check: H_m(k), document support, shrinkage.

H_m(k) = log( (P(k|C+_m)+eps) / (P(k|C_hard-_m)+eps) )
applies to every source the same way. Topic keywords are not auto-QA weights.
"""

from __future__ import annotations

import math
import csv
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .config import EPS, ExperimentParams, SHRINKAGE_N0


def normalize_keyword(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    return value


def keyword_in_text(keyword: str, text: str) -> bool:
    key = normalize_keyword(keyword)
    if not key:
        return False
    pattern = r"\b" + re.escape(key).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, str(text or "").casefold()) is not None


def presence_rate(
    keyword: str,
    texts: Sequence[str],
    *,
    eps: float = EPS,
) -> float:
    if not texts:
        return 0.0
    hits = sum(1 for text in texts if keyword_in_text(keyword, text))
    return (hits + 0.0) / float(len(texts))


def discriminative_score(
    positive_rate: float,
    negative_rate: float,
    *,
    eps: float = EPS,
) -> float:
    return math.log((float(positive_rate) + eps) / (float(negative_rate) + eps))


def support_threshold(n_docs: int, *, docs: int, fraction: float) -> int:
    n = max(0, int(n_docs))
    return max(int(docs), int(math.ceil(float(fraction) * n)), 1 if n else 0)


def shrinkage_lambda(n_docs: int, *, n0: float = SHRINKAGE_N0) -> float:
    n = max(0, int(n_docs))
    return float(n0) / float(n0 + n) if (n0 + n) else 1.0


def blend(cluster_weight: float, global_weight: float, lam: float) -> float:
    return float(lam) * float(global_weight) + (1.0 - float(lam)) * float(cluster_weight)


@dataclass
class KeywordProfile:
    cluster_id: int
    keyword: str
    source_attention: int = 0
    source_diff: int = 0
    source_question_keybert: int = 0
    source_topic: int = 0
    attention_score: float = 0.0
    diff_score: float = 0.0
    question_score: float = 0.0
    topic_score: float = 0.0
    positive_frequency: float = 0.0
    negative_frequency: float = 0.0
    hard_negative_frequency: float = 0.0
    document_support: float = 0.0
    document_support_count: int = 0
    discriminative_score: float = 0.0
    global_discriminative_score: float = 0.0
    final_weight: float = 0.0
    ngram: int = 1

    def to_row(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "keyword": self.keyword,
            "source_attention": self.source_attention,
            "source_diff": self.source_diff,
            "source_question_keybert": self.source_question_keybert,
            "source_topic": self.source_topic,
            "attention_score": self.attention_score,
            "diff_score": self.diff_score,
            "question_score": self.question_score,
            "topic_score": self.topic_score,
            "positive_frequency": self.positive_frequency,
            "negative_frequency": self.negative_frequency,
            "hard_negative_frequency": self.hard_negative_frequency,
            "document_support": self.document_support,
            "document_support_count": self.document_support_count,
            "discriminative_score": self.discriminative_score,
            "global_discriminative_score": self.global_discriminative_score,
            "final_weight": self.final_weight,
            "ngram": self.ngram,
        }


def profile_from_row(row: Mapping[str, object]) -> KeywordProfile:
    def _int(name: str) -> int:
        value = row.get(name, 0)
        return int(float(value)) if str(value).strip() != "" else 0

    def _float(name: str) -> float:
        value = row.get(name, 0.0)
        return float(value) if str(value).strip() != "" else 0.0

    return KeywordProfile(
        cluster_id=_int("cluster_id"),
        keyword=str(row.get("keyword") or ""),
        source_attention=_int("source_attention"),
        source_diff=_int("source_diff"),
        source_question_keybert=_int("source_question_keybert"),
        source_topic=_int("source_topic"),
        attention_score=_float("attention_score"),
        diff_score=_float("diff_score"),
        question_score=_float("question_score"),
        topic_score=_float("topic_score"),
        positive_frequency=_float("positive_frequency"),
        negative_frequency=_float("negative_frequency"),
        hard_negative_frequency=_float("hard_negative_frequency"),
        document_support=_float("document_support"),
        document_support_count=_int("document_support_count"),
        discriminative_score=_float("discriminative_score"),
        global_discriminative_score=_float("global_discriminative_score"),
        final_weight=_float("final_weight"),
        ngram=_int("ngram") or 1,
    )


def load_profiles_by_pool(path: Path) -> dict[str, dict[int, list[KeywordProfile]]]:
    grouped: dict[str, dict[int, list[KeywordProfile]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            profile = profile_from_row(row)
            pool = str(row.get("pool") or "K1")
            grouped[pool][int(profile.cluster_id)].append(profile)
    return {pool: dict(clusters) for pool, clusters in grouped.items()}


def score_profiles(
    profiles: Sequence[KeywordProfile],
    positive_texts: Sequence[str],
    hard_texts: Sequence[str],
    easy_texts: Sequence[str],
    doc_texts: Sequence[str],
    *,
    params: ExperimentParams | None = None,
    negative: str = "hard",
) -> list[KeywordProfile]:
    params = params or ExperimentParams()
    neg = hard_texts if negative == "hard" else easy_texts
    n_docs = len(doc_texts)
    floor = support_threshold(
        n_docs, docs=params.support_docs, fraction=params.support_fraction
    )
    scored = []
    for profile in profiles:
        pos_rate = presence_rate(profile.keyword, positive_texts, eps=0.0)
        hard_rate = presence_rate(profile.keyword, hard_texts, eps=0.0)
        easy_rate = presence_rate(profile.keyword, easy_texts, eps=0.0)
        neg_rate = presence_rate(profile.keyword, neg, eps=0.0)
        support_count = sum(1 for text in doc_texts if keyword_in_text(profile.keyword, text))
        profile.positive_frequency = pos_rate
        profile.hard_negative_frequency = hard_rate
        profile.negative_frequency = easy_rate
        profile.document_support_count = int(support_count)
        profile.document_support = (support_count / n_docs) if n_docs else 0.0
        profile.discriminative_score = discriminative_score(
            pos_rate, neg_rate, eps=params.eps
        )
        profile.ngram = len(profile.keyword.split())
        if support_count < floor:
            continue
        scored.append(profile)
    return scored
