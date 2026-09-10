"""Deterministic source-only normalization for SciREX gold entities.

Normalization changes surface form only. It does not add synonyms, lemmas,
translations, abbreviations, or external canonical names.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)
_UNDERSCORES = re.compile(r"_+")
_WHITESPACE = re.compile(r"\s+")
_SPACED_DASH = re.compile(r"\s*-\s*")


def normalize_entity(value: str) -> str:
    """NFKC, underscores → space, dash unification, whitespace collapse."""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(_DASH_TRANSLATION)
    text = _UNDERSCORES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def matching_key(value: str) -> str:
    """Casefolded normalized form used only for identity and lexical search."""
    text = normalize_entity(value)
    text = _SPACED_DASH.sub("-", text)
    return text.casefold()


def identity_key(value: str) -> str:
    return matching_key(value)


def _case_rank(value: str) -> tuple[int, int, int]:
    letters = [char for char in value if char.isalpha()]
    upper = sum(char.isupper() for char in letters)
    lower = sum(char.islower() for char in letters)
    return (upper, lower, len(value))


@dataclass(frozen=True)
class GoldEntity:
    canonical: str
    raw_variants: tuple[str, ...]
    normalized_variants: tuple[str, ...]
    identity_key: str


def deduplicate_entities(raw_values: list[str]) -> list[GoldEntity]:
    """Group case/format variants by identity_key; keep every raw variant."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        text = str(raw)
        if not text.strip():
            continue
        key = identity_key(text)
        if not key:
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        if text not in groups[key]:
            groups[key].append(text)

    entities: list[GoldEntity] = []
    for key in order:
        variants = tuple(groups[key])
        chosen = max(variants, key=_case_rank)
        normalized = tuple(dict.fromkeys(normalize_entity(item) for item in variants))
        entities.append(
            GoldEntity(
                canonical=normalize_entity(chosen),
                raw_variants=variants,
                normalized_variants=normalized,
                identity_key=key,
            )
        )
    return entities
