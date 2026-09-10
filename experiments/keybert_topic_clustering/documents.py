"""Load SciREx JSON/JSONL and recover title/abstract/headings from body text."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from untie.keyword_tuning import deterministic_document_split


_DOC_PREFIX = re.compile(r"^(?:document|title)\s*:\s*", re.IGNORECASE)
_ABSTRACT_CUT = re.compile(
    r"\b(?:section\s*:?\s*abstract|abstract\s*:)\b",
    re.IGNORECASE,
)
_SECTION_SPLIT = re.compile(r"\bsection\s*:\s*", re.IGNORECASE)
_SINGLE_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "experiments",
        "experiment",
        "conclusion",
        "conclusions",
        "methods",
        "method",
        "results",
        "discussion",
        "references",
        "approach",
        "model",
        "dataset",
        "related",
    }
)


def _heading_name(block: str) -> str:
    words = [item.strip(".:;,") for item in str(block).split() if item.strip(".:;,")]
    if not words:
        return ""
    first = words[0]
    if first.casefold() in _SINGLE_HEADINGS:
        if first.casefold() == "related" and len(words) > 1 and words[1].casefold() == "work":
            return "Related Work"
        return first
    taken: list[str] = []
    for word in words[:5]:
        if taken and (word[:1].islower() or word.casefold() in {"we", "the", "this", "in", "our"}):
            break
        taken.append(word)
    return " ".join(taken)


def parse_headings(text: str) -> tuple[str, ...]:
    parts = _SECTION_SPLIT.split(str(text or ""))
    headings = []
    for block in parts[1:]:
        name = _heading_name(block)
        if name and name.casefold() != "abstract":
            headings.append(name)
    return tuple(headings)


@dataclass(frozen=True)
class PaperDocument:
    doc_id: str
    split: str
    text: str
    title: str
    abstract: str
    headings: tuple[str, ...]
    references: tuple[str, ...]
    metadata_empty: bool

    @property
    def r1_abstract(self) -> str:
        return self.abstract.strip()

    @property
    def r2_title_abstract(self) -> str:
        parts = [item for item in (self.title, self.abstract) if item.strip()]
        return "\n".join(parts)

    @property
    def r3_title_abstract_headings(self) -> str:
        parts = [item for item in (self.title, self.abstract, *self.headings) if item.strip()]
        return "\n".join(parts)

    def representation(self, name: str) -> str:
        mapping = {
            "R1": self.r1_abstract,
            "R2": self.r2_title_abstract,
            "R3": self.r3_title_abstract_headings,
        }
        if name not in mapping:
            raise ValueError(f"unknown representation {name}")
        text = mapping[name].strip()
        if text:
            return text
        # Last-resort compact opening, not the full paper.
        return compact_opening(self.text, max_words=80)


def load_payload(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("documents")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("dataset must contain a JSON array or JSONL objects")
    return payload


def parse_references(raw: Mapping[str, Any]) -> tuple[str, ...]:
    value = raw.get("tasks_cleaned", raw.get("tasks", ()))
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = (value,)
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(
        str(item).replace("_", " ").strip()
        for item in values
        if item is not None and str(item).strip()
    )


def _clip_words(text: str, max_words: int) -> str:
    words = [item for item in str(text).split() if item]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return parts[0].strip()


def compact_opening(text: str, *, max_words: int = 80) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = _DOC_PREFIX.sub("", cleaned)
    cut = _ABSTRACT_CUT.search(cleaned)
    if cut is not None and cut.start() >= 8:
        heading = cleaned[: cut.start()].strip(" .:;-")
        rest = cleaned[cut.end() :].strip()
        abstract = _first_sentence(rest)
        return _clip_words(" ".join(item for item in (heading, abstract) if item), max_words)
    return _clip_words(_first_sentence(cleaned) or cleaned, max_words)


def parse_title_abstract_headings(text: str) -> tuple[str, str, tuple[str, ...]]:
    """Recover paper identity from SciREx-style body text."""
    raw = str(text or "").strip()
    headings = parse_headings(raw)
    cleaned = re.sub(r"\s+", " ", raw)
    cleaned = _DOC_PREFIX.sub("", cleaned)
    cut = _ABSTRACT_CUT.search(cleaned)
    if cut is not None and cut.start() >= 8:
        title = cleaned[: cut.start()].strip(" .:;-")
        remainder = cleaned[cut.end() :].strip()
        next_section = re.search(r"\bsection\s*:", remainder, re.IGNORECASE)
        abstract_body = remainder[: next_section.start()] if next_section else remainder
        abstract = _clip_words(abstract_body, 120)
        return title, abstract, headings
    title = _clip_words(_first_sentence(cleaned), 40)
    rest = cleaned[len(title) :].strip(" .")
    abstract = _clip_words(_first_sentence(rest) if rest else "", 80)
    return title, abstract, headings


def load_papers(
    path: Path,
    *,
    seed: int = 42,
    quick: bool = False,
    quick_train: int = 24,
    quick_dev: int = 8,
    quick_test: int = 8,
) -> list[PaperDocument]:
    payload = load_payload(path)
    ids = [str(item.get("doc_id", item.get("id", ""))) for item in payload]
    if any(not doc_id for doc_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset document ids must be non-empty and unique")
    explicit = [item.get("split") for item in payload]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("split must be present on every record or none")
        split_by_id = {
            doc_id: str(value).casefold() for doc_id, value in zip(ids, explicit)
        }
    else:
        split = deterministic_document_split(ids, seed=seed)
        split_by_id = {
            doc_id: name
            for name, group in split.items()
            for doc_id in group
        }

    papers: list[PaperDocument] = []
    for raw in payload:
        doc_id = str(raw.get("doc_id", raw.get("id", "")))
        text = str(raw.get("original_text", raw.get("text", "")) or "")
        meta_title = str(raw.get("title", "") or "").strip()
        meta_abstract = str(raw.get("abstract", "") or "").strip()
        raw_headings = raw.get("headings", ())
        if isinstance(raw_headings, str):
            raw_headings = (raw_headings,)
        meta_headings = tuple(str(item).strip() for item in raw_headings if str(item).strip())
        metadata_empty = not (meta_title or meta_abstract or meta_headings)
        if metadata_empty:
            title, abstract, headings = parse_title_abstract_headings(text)
        else:
            title, abstract, headings = meta_title, meta_abstract, meta_headings
        papers.append(
            PaperDocument(
                doc_id=doc_id,
                split=split_by_id[doc_id],
                text=text,
                title=title,
                abstract=abstract,
                headings=headings,
                references=parse_references(raw),
                metadata_empty=metadata_empty,
            )
        )
    if quick:
        kept: list[PaperDocument] = []
        limits = {"train": quick_train, "dev": quick_dev, "test": quick_test}
        counts = {"train": 0, "dev": 0, "test": 0}
        for paper in papers:
            if counts[paper.split] >= limits[paper.split]:
                continue
            kept.append(paper)
            counts[paper.split] += 1
        papers = kept
    return papers


def split_groups(papers: list[PaperDocument]) -> dict[str, list[PaperDocument]]:
    groups: dict[str, list[PaperDocument]] = {"train": [], "dev": [], "test": []}
    for paper in papers:
        groups.setdefault(paper.split, []).append(paper)
    return groups
