"""Load SciREX records and parse structural markers that actually appear."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import EXPECTED_SOURCE_FIELDS, STRUCTURAL_MARKERS


_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in STRUCTURAL_MARKERS) + r")\s*:",
    re.IGNORECASE,
)
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
        "appendix",
        "acknowledgments",
        "acknowledgements",
    }
)
_HEADING_STOP = frozenset({"we", "the", "this", "in", "our"})


@dataclass(frozen=True)
class SourceRecord:
    doc_id: str
    methods: tuple[str, ...]
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    materials: tuple[str, ...]
    score: tuple[str, ...]
    original_text: str
    extra_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextSegment:
    segment_id: int
    type: str
    title: str
    text: str
    raw_start: int
    raw_end: int
    marker_start: int | None = None


@dataclass(frozen=True)
class SourceInspection:
    path: str
    format_name: str
    n_records: int
    fields_present: tuple[str, ...]
    field_frequency: dict[str, int]
    first_record_fields: tuple[str, ...]
    first_record_preview: dict[str, Any]
    expected_fields: tuple[str, ...]
    missing_expected_fields: tuple[str, ...]
    unexpected_fields: tuple[str, ...]
    schema_matches_expected: bool
    notes: tuple[str, ...]


def load_scirex_records(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Load JSON array or JSONL objects. Same strategy as load_payload / load_json_dataframe."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    raw = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
        format_name = "json"
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
        format_name = "jsonl"
    if isinstance(payload, dict):
        payload = payload.get("documents", payload)
        format_name = "json_object"
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("SciREX source must be a JSON array or JSONL objects")
    return payload, format_name


def _as_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            if not isinstance(item, (str, int, float)):
                raise TypeError(
                    f"field {field_name!r} contains {type(item).__name__}; "
                    "refusing to coerce nested objects"
                )
            text = str(item)
            if text.strip():
                items.append(text)
        return tuple(items)
    raise TypeError(
        f"field {field_name!r} has type {type(value).__name__}; "
        "expected list of strings"
    )


def parse_source_record(raw: Mapping[str, Any]) -> SourceRecord:
    missing = [name for name in ("doc_id", "original_text") if name not in raw]
    if missing:
        raise ValueError(f"source record missing required fields: {missing}")
    extra = tuple(sorted(name for name in raw if name not in EXPECTED_SOURCE_FIELDS))
    return SourceRecord(
        doc_id=str(raw["doc_id"]),
        methods=_as_string_tuple(raw.get("methods"), field_name="methods"),
        tasks=_as_string_tuple(raw.get("tasks"), field_name="tasks"),
        metrics=_as_string_tuple(raw.get("metrics"), field_name="metrics"),
        materials=_as_string_tuple(raw.get("materials"), field_name="materials"),
        score=_as_string_tuple(raw.get("score"), field_name="score"),
        original_text=str(raw["original_text"] or ""),
        extra_fields=extra,
    )


def _preview_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > 240:
            return value[:240] + "…"
        return value
    if isinstance(value, list):
        return value[:12]
    return value


def inspect_source(path: str | Path) -> SourceInspection:
    payload, format_name = load_scirex_records(path)
    field_counts: Counter[str] = Counter()
    for record in payload:
        field_counts.update(record.keys())
    first = payload[0] if payload else {}
    present = tuple(sorted(field_counts))
    first_fields = tuple(first.keys())
    missing = tuple(name for name in EXPECTED_SOURCE_FIELDS if name not in field_counts)
    unexpected = tuple(name for name in present if name not in EXPECTED_SOURCE_FIELDS)
    notes: list[str] = []
    if format_name == "jsonl":
        notes.append("file is JSONL, not a JSON array; parser uses line-wise JSON")
    if unexpected:
        notes.append(f"unexpected fields were left untouched: {list(unexpected)}")
    if missing:
        notes.append(f"expected fields absent from corpus: {list(missing)}")
    text_key = "original_text" if "original_text" in first else None
    if text_key is None and payload:
        notes.append("original_text is missing; refusing to substitute another text field")
    return SourceInspection(
        path=str(Path(path)),
        format_name=format_name,
        n_records=len(payload),
        fields_present=present,
        field_frequency={key: int(field_counts[key]) for key in present},
        first_record_fields=first_fields,
        first_record_preview={key: _preview_value(value) for key, value in first.items()},
        expected_fields=EXPECTED_SOURCE_FIELDS,
        missing_expected_fields=missing,
        unexpected_fields=unexpected,
        schema_matches_expected=not missing and not unexpected,
        notes=tuple(notes),
    )


def _heading_prefix(body: str) -> str:
    """Take a heading-like prefix that is an actual substring of body."""
    if not body.strip():
        return ""
    tokens = re.findall(r"\S+", body)
    if not tokens:
        return ""
    first_core = tokens[0].strip(".:;,")
    taken: list[str]
    if first_core.casefold() in _SINGLE_HEADINGS:
        taken = [tokens[0]]
        if (
            first_core.casefold() == "related"
            and len(tokens) > 1
            and tokens[1].casefold().strip(".:;,") == "work"
        ):
            taken = [tokens[0], tokens[1]]
    else:
        taken = []
        for word in tokens[:8]:
            core = word.strip(".:;,")
            if taken and (
                (word[:1].islower() and core.casefold() not in {"and", "of", "for", "to"})
                or core.casefold() in _HEADING_STOP
            ):
                break
            taken.append(word)
            if word.endswith("."):
                break
    if not taken:
        return ""
    pos = 0
    end = 0
    for word in taken:
        index = body.find(word, pos)
        if index < 0:
            break
        end = index + len(word)
        pos = end
    return body[:end].strip()


def parse_structural_segments(text: str) -> list[TextSegment]:
    """Split original_text on markers that actually occur. Offsets stay raw."""
    if not text:
        return []
    matches = list(_MARKER_RE.finditer(text))
    segments: list[TextSegment] = []
    segment_id = 0
    if not matches:
        return [
            TextSegment(
                segment_id=0,
                type="",
                title="",
                text=text,
                raw_start=0,
                raw_end=len(text),
            )
        ]

    first_start = matches[0].start()
    if first_start > 0:
        prefix = text[:first_start]
        segments.append(
            TextSegment(
                segment_id=segment_id,
                type="",
                title="",
                text=prefix,
                raw_start=0,
                raw_end=first_start,
            )
        )
        segment_id += 1

    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        marker_type = match.group(1).casefold()
        segments.append(
            TextSegment(
                segment_id=segment_id,
                type=marker_type,
                title=_heading_prefix(body),
                text=body,
                raw_start=body_start,
                raw_end=body_end,
                marker_start=match.start(),
            )
        )
        segment_id += 1
    return segments
