"""Candidate keyword pools K1–K9. Sources are unioned; weights are scored later."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.routing import RoutingResult
from untie.keyword_evidence import CachedDocumentEvidence

from .config import POOL_SPECS
from .discrimination import KeywordProfile, normalize_keyword


def load_assignments(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["doc_id"])] = {
                "doc_id": row["doc_id"],
                "split": row["split"],
                "cluster_id": int(row["cluster_id"]) if str(row.get("cluster_id", "")).strip() != "" else None,
            }
    return rows


def load_routing(path: Path) -> dict[str, RoutingResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = {}
    for item in payload:
        routes[str(item["doc_id"])] = RoutingResult(
            doc_id=str(item["doc_id"]),
            split=str(item["split"]),
            top1=int(item["top1"]),
            top1_similarity=float(item["top1_similarity"]),
            top_ids=tuple(int(value) for value in item.get("top_ids", ())),
            similarities=tuple(float(value) for value in item.get("similarities", ())),
            weights=tuple(float(value) for value in item.get("weights", ())),
        )
    return routes


def cluster_of(
    doc_id: str,
    split: str,
    assignments: Mapping[str, Mapping[str, object]],
    routes: Mapping[str, RoutingResult],
) -> int:
    if split == "train":
        item = assignments[doc_id]
        cluster = item.get("cluster_id")
        if cluster is None:
            raise KeyError(doc_id)
        return int(cluster)
    return int(routes[doc_id].top1)


def load_document_keyphrases(path: Path) -> dict[str, list[tuple[str, float]]]:
    by_doc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phrase = normalize_keyword(row.get("phrase", ""))
            if not phrase:
                continue
            by_doc[str(row["document_id"])].append((phrase, float(row.get("weight") or 0.0)))
    return dict(by_doc)


def load_topic_terms(chosen_configs: Mapping[str, object]) -> dict[int, list[tuple[str, float]]]:
    details = chosen_configs.get("details", {}).get("C_keybert", {})
    raw = details.get("representative_terms", {})
    out: dict[int, list[tuple[str, float]]] = {}
    for cluster_id, terms in raw.items():
        out[int(cluster_id)] = [
            (normalize_keyword(item["term"]), float(item.get("weight") or 0.0))
            for item in terms
            if normalize_keyword(item.get("term", ""))
        ]
    return out


def _blank(cluster_id: int, keyword: str) -> KeywordProfile:
    return KeywordProfile(cluster_id=cluster_id, keyword=keyword)


def attention_pool(
    cluster_id: int,
    train_ids: Sequence[str],
    evidence: Mapping[str, CachedDocumentEvidence],
    *,
    require_diff: bool = False,
) -> dict[str, KeywordProfile]:
    acc: dict[str, KeywordProfile] = {}
    att_sum: dict[str, list[float]] = defaultdict(list)
    diff_sum: dict[str, list[float]] = defaultdict(list)
    for doc_id in train_ids:
        item = evidence.get(doc_id)
        if item is None:
            continue
        for cand in item.candidates:
            key = normalize_keyword(cand.lemma or cand.word)
            if not key:
                continue
            if require_diff and float(cand.score_difference) <= 0:
                continue
            profile = acc.setdefault(key, _blank(cluster_id, key))
            profile.source_attention = 1
            att_sum[key].append(float(cand.attention_weight))
            if float(cand.score_difference) > 0:
                profile.source_diff = 1
                diff_sum[key].append(float(cand.score_difference))
    for key, profile in acc.items():
        profile.attention_score = float(sum(att_sum[key]) / len(att_sum[key]))
        if diff_sum[key]:
            profile.diff_score = float(sum(diff_sum[key]) / len(diff_sum[key]))
    return acc


def question_keybert_pool(
    cluster_id: int,
    train_ids: Sequence[str],
    phrases: Mapping[str, Sequence[tuple[str, float]]],
    phrase_question_sim: Mapping[str, float],
    *,
    min_ngram: int = 1,
    max_ngram: int = 2,
) -> dict[str, KeywordProfile]:
    acc: dict[str, KeywordProfile] = {}
    for doc_id in train_ids:
        for phrase, _weight in phrases.get(doc_id, ()):
            ngram = len(phrase.split())
            if ngram < min_ngram or ngram > max_ngram:
                continue
            profile = acc.setdefault(phrase, _blank(cluster_id, phrase))
            profile.source_question_keybert = 1
            profile.question_score = max(
                profile.question_score, float(phrase_question_sim.get(phrase, 0.0))
            )
    return acc


def topic_pool(
    cluster_id: int,
    terms: Sequence[tuple[str, float]],
) -> dict[str, KeywordProfile]:
    acc: dict[str, KeywordProfile] = {}
    for phrase, weight in terms:
        profile = acc.setdefault(phrase, _blank(cluster_id, phrase))
        profile.source_topic = 1
        profile.topic_score = max(profile.topic_score, float(weight))
    return acc


def merge_pools(cluster_id: int, parts: Sequence[Mapping[str, KeywordProfile]]) -> dict[str, KeywordProfile]:
    acc: dict[str, KeywordProfile] = {}
    for part in parts:
        for key, src in part.items():
            dst = acc.setdefault(key, _blank(cluster_id, key))
            dst.source_attention = max(dst.source_attention, src.source_attention)
            dst.source_diff = max(dst.source_diff, src.source_diff)
            dst.source_question_keybert = max(
                dst.source_question_keybert, src.source_question_keybert
            )
            dst.source_topic = max(dst.source_topic, src.source_topic)
            dst.attention_score = max(dst.attention_score, src.attention_score)
            dst.diff_score = max(dst.diff_score, src.diff_score)
            dst.question_score = max(dst.question_score, src.question_score)
            dst.topic_score = max(dst.topic_score, src.topic_score)
    return acc


def importance_weight(profile: KeywordProfile) -> float:
    values = []
    if profile.source_attention:
        values.append(profile.attention_score)
    if profile.source_diff:
        values.append(profile.diff_score)
    if profile.source_question_keybert:
        values.append(profile.question_score)
    if profile.source_topic:
        values.append(profile.topic_score)
    return max(values) if values else 0.0


def build_named_pools(
    cluster_id: int,
    *,
    attention: Mapping[str, KeywordProfile],
    attention_diff: Mapping[str, KeywordProfile],
    question: Mapping[str, KeywordProfile],
    topic: Mapping[str, KeywordProfile],
) -> dict[str, dict[str, KeywordProfile]]:
    named = {
        "attention": attention,
        "diff": attention_diff,
        "question_keybert": question,
        "topic": topic,
    }
    out = {}
    for pool_name, sources in POOL_SPECS:
        parts = [named[source] for source in sources if named[source]]
        out[pool_name] = merge_pools(cluster_id, parts)
    return out
