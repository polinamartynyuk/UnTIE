"""Frozen clustering spec and experiment knobs. No field-specific task lexicon."""

from __future__ import annotations

from dataclasses import dataclass

FROZEN_CLUSTERING = {
    "name": "C_keybert",
    "config_id": "E2+R2+ng1-2_top20_mmr0.50|KMeans|k4|s42",
    "representation": "E2",
    "text_field": "R2",
    "algorithm": "KMeans",
    "k": 4,
    "seed": 42,
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "keyphrase_ngram": (1, 2),
    "keyphrase_top_n": 20,
    "mmr_diversity": 0.5,
}

QUESTION_DEFAULT = "Which task was solved?"
FIELD_ID_DEFAULT = 1
GOLD_SUBSTRING_MIN_CHARS = 4
GOLD_ANSWER_F1 = 0.5
EPS = 1e-6
HN_TOP_N = 3
SHRINKAGE_N0 = 30
DEFAULT_SUPPORT_FRACTION = 0.05
DEFAULT_SUPPORT_DOCS = 2


POOL_SPECS = (
    ("K1", ("attention",)),
    ("K2", ("attention", "diff")),
    ("K3", ("question_keybert",)),
    ("K4", ("topic",)),
    ("K5", ("attention", "question_keybert")),
    ("K6", ("attention", "topic")),
    ("K7", ("question_keybert", "topic")),
    ("K8", ("attention", "question_keybert", "topic")),
    ("K9", ("attention", "diff", "question_keybert", "topic")),
)


@dataclass(frozen=True)
class ExperimentParams:
    gold_substring_min_chars: int = GOLD_SUBSTRING_MIN_CHARS
    gold_answer_f1: float = GOLD_ANSWER_F1
    eps: float = EPS
    hn_top_n: int = HN_TOP_N
    shrinkage_n0: float = SHRINKAGE_N0
    support_fraction: float = DEFAULT_SUPPORT_FRACTION
    support_docs: int = DEFAULT_SUPPORT_DOCS
    question_chunk_alpha: float = 1.0
    keep_negative_h: bool = True
    use_modifiers: bool = True
    soft_routing: bool = False
    max_keywords_per_cluster: int = 80
