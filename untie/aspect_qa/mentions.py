"""Deterministic lexical mention search with raw character offsets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .normalize import matching_key, normalize_entity


_DASHES = frozenset("-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d")


@dataclass(frozen=True)
class GoldMention:
    mention_id: str
    canonical: str
    raw_entity: str
    matched_text: str
    char_start: int
    char_end: int
    in_bibliography: bool = False


def _fold_char(char: str) -> str:
    if char in _DASHES or char == "_":
        folded = " " if char == "_" else "-"
    else:
        folded = unicodedata.normalize("NFKC", char).casefold()
    return folded


def build_search_index(text: str) -> tuple[str, list[int]]:
    """Build a casefolded search string and a map from search index → raw index.

    Allowed technical transforms only: NFKC, casefold, underscore→space,
    dash unification, whitespace collapse, spaces around dashes.
    """
    chars: list[str] = []
    raw_map: list[int] = []
    for index, char in enumerate(text):
        folded = _fold_char(char)
        for piece in folded:
            chars.append(piece)
            raw_map.append(index)

    collapsed: list[str] = []
    collapsed_map: list[int] = []
    prev_space = False
    for char, raw_index in zip(chars, raw_map):
        if char.isspace():
            if prev_space:
                continue
            collapsed.append(" ")
            collapsed_map.append(raw_index)
            prev_space = True
            continue
        collapsed.append(char)
        collapsed_map.append(raw_index)
        prev_space = False

    # Collapse optional spaces around dashes: "v - net" → "v-net".
    dashed: list[str] = []
    dashed_map: list[int] = []
    i = 0
    while i < len(collapsed):
        if collapsed[i] == "-" :
            if dashed and dashed[-1] == " ":
                dashed.pop()
                dashed_map.pop()
            dashed.append("-")
            dashed_map.append(collapsed_map[i])
            i += 1
            if i < len(collapsed) and collapsed[i] == " ":
                i += 1
            continue
        dashed.append(collapsed[i])
        dashed_map.append(collapsed_map[i])
        i += 1
    return "".join(dashed), dashed_map


def _needles_for(raw_variants: tuple[str, ...], normalized_variants: tuple[str, ...]) -> list[str]:
    needles: list[str] = []
    for value in (*raw_variants, *normalized_variants):
        key = matching_key(value)
        if key and key not in needles:
            needles.append(key)
    needles.sort(key=len, reverse=True)
    return needles


def find_lexical_mentions(
    text: str,
    *,
    canonical: str,
    raw_variants: tuple[str, ...],
    normalized_variants: tuple[str, ...] | None = None,
    mention_prefix: str,
) -> list[GoldMention]:
    haystack, index_map = build_search_index(text)
    if not haystack:
        return []
    variants = normalized_variants or tuple(normalize_entity(item) for item in raw_variants)
    mentions: list[GoldMention] = []
    occupied = [False] * len(haystack)
    occurrence = 0
    for needle in _needles_for(raw_variants, variants):
        pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
        for match in pattern.finditer(haystack):
            start, end = match.start(), match.end()
            if any(occupied[start:end]):
                continue
            raw_start = index_map[start]
            raw_end = index_map[end - 1] + 1
            matched = text[raw_start:raw_end]
            if not matched:
                continue
            for pos in range(start, end):
                occupied[pos] = True
            mentions.append(
                GoldMention(
                    mention_id=f"{mention_prefix}:{occurrence}",
                    canonical=canonical,
                    raw_entity=raw_variants[0],
                    matched_text=matched,
                    char_start=raw_start,
                    char_end=raw_end,
                )
            )
            occurrence += 1
    mentions.sort(key=lambda item: (item.char_start, item.char_end))
    remapped: list[GoldMention] = []
    for index, mention in enumerate(mentions):
        remapped.append(
            GoldMention(
                mention_id=f"{mention_prefix}:{index}",
                canonical=mention.canonical,
                raw_entity=mention.raw_entity,
                matched_text=mention.matched_text,
                char_start=mention.char_start,
                char_end=mention.char_end,
            )
        )
    return remapped
