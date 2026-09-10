"""Unguided (no-gold) attention keywords from top-confidence QA chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from untie.keyword_evidence import CachedDocumentEvidence
from untie.ranking import WeightedKeyword


def merge_attention_batches(
    batches: Sequence[Sequence[Mapping[str, Any]]],
    *,
    top_keywords: int,
) -> list[WeightedKeyword]:
    merged: dict[str, float] = {}
    for batch in batches:
        for item in batch:
            word = str(item.get("word", "")).strip()
            if not word:
                continue
            key = word.casefold()
            weight = float(item.get("weight", 0.0))
            merged[key] = max(merged.get(key, 0.0), weight)
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[: max(1, int(top_keywords))]
    return [
        WeightedKeyword(
            word=word,
            lemma=word,
            stem=word,
            attention_weight=float(weight),
            score_difference=0.0,
        )
        for word, weight in ranked
    ]


def extract_online_attention(
    evidence: CachedDocumentEvidence,
    attention_extractor: Callable[[str, str], list[dict[str, float | str]]],
    *,
    question: str,
    top_chunks: int,
    top_keywords: int,
) -> tuple[list[WeightedKeyword], tuple[int, ...]]:
    """Attention from the highest-QA-confidence chunks. No gold span is used."""
    ranked = sorted(evidence.chunks, key=lambda item: (-float(item.confidence), item.index))
    chosen = ranked[: max(1, int(top_chunks))]
    batches = [attention_extractor(question, item.text) for item in chosen]
    keywords = merge_attention_batches(batches, top_keywords=top_keywords)
    return keywords, tuple(item.index for item in chosen)


def load_attention_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_attention_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def keywords_from_cache(raw: Mapping[str, Any]) -> list[WeightedKeyword]:
    items = raw.get("keywords") or []
    return [
        WeightedKeyword(
            word=str(item["word"]),
            lemma=str(item.get("lemma", item["word"])).casefold(),
            stem=str(item.get("stem", item["word"])).casefold(),
            attention_weight=float(item.get("attention_weight", item.get("weight", 1.0))),
            score_difference=float(item.get("score_difference", 0.0)),
        )
        for item in items
        if str(item.get("word", "")).strip()
    ]


def keywords_to_cache(keywords: Sequence[WeightedKeyword], chunk_indices: Sequence[int]) -> dict[str, Any]:
    return {
        "chunk_indices": list(chunk_indices),
        "keywords": [
            {
                "word": item.word,
                "lemma": item.lemma,
                "stem": item.stem,
                "attention_weight": item.attention_weight,
                "score_difference": item.score_difference,
            }
            for item in keywords
        ],
    }
