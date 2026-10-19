"""Download and parse all three official inline-tagged RuSERRC aspect files."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from untie.aspect_qa.normalize import normalize_entity

from .config import ASPECT_SPECS, OFFICIAL_PARTS, OFFICIAL_REF, OFFICIAL_REPOSITORY

_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z]+)\s*>")


@dataclass(frozen=True)
class TaggedSpan:
    tag: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_id: str
    source_part: str
    tagged_text: str
    clean_text: str
    spans: tuple[TaggedSpan, ...]
    parse_warnings: tuple[str, ...] = ()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, *, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "UnTIE-RuSERRC-V4.4-preparation/1.0"}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_official_parts(destination: Path, progress: Callable[[str], None]) -> dict[str, object]:
    """Resolve the branch once, then download every file from one immutable commit."""
    destination.mkdir(parents=True, exist_ok=False)
    api = "https://api.github.com/repos/iis-research-team/ruserrc-dataset/commits/" + OFFICIAL_REF
    progress(f"resolving official revision {OFFICIAL_REF}")
    revision = json.loads(_fetch(api, accept="application/vnd.github+json"))["sha"]
    files: list[dict[str, object]] = []
    for index, name in enumerate(OFFICIAL_PARTS, start=1):
        url = ("https://raw.githubusercontent.com/iis-research-team/ruserrc-dataset/"
               f"{revision}/ruserrc_aspects/{name}")
        progress(f"downloading {name} ({index}/{len(OFFICIAL_PARTS)})")
        payload = _fetch(url)
        target = destination / name
        target.write_bytes(payload)
        files.append({"name": name, "url": url, "bytes": len(payload), "sha256": sha256_path(target)})
    manifest = {"repository": OFFICIAL_REPOSITORY, "requested_ref": OFFICIAL_REF,
                "resolved_commit": revision, "files": files}
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_tagged_text(value: str) -> tuple[str, tuple[TaggedSpan, ...], tuple[str, ...]]:
    """Remove markup, preserve offsets, and report deterministic repairs of malformed nesting."""
    output: list[str] = []
    length = 0
    cursor = 0
    stack: list[tuple[str, int]] = []
    spans: list[TaggedSpan] = []
    warnings: list[str] = []
    canonical_tags = {name.casefold(): name for name in ASPECT_SPECS}

    def close_span(tag: str, start: int, end: int, current: str) -> None:
        raw = current[start:end]
        left, right = len(raw) - len(raw.lstrip()), len(raw.rstrip())
        span_start, span_end = start + left, start + right
        if span_end > span_start:
            spans.append(TaggedSpan(tag, span_start, span_end, current[span_start:span_end]))

    for match in _TAG_RE.finditer(str(value)):
        piece = value[cursor:match.start()]
        output.append(piece)
        length += len(piece)
        closing, source_tag = match.group(1), match.group(2)
        tag = canonical_tags.get(source_tag.casefold())
        if tag is None:
            raise ValueError(f"unknown inline aspect tag <{source_tag}>")
        if not closing:
            stack.append((tag, length))
        else:
            current = "".join(output)
            matching = next((i for i in range(len(stack) - 1, -1, -1) if stack[i][0] == tag), None)
            if matching is None:
                warnings.append(f"ignored unmatched closing tag </{source_tag}> at {match.start()}")
            else:
                while len(stack) - 1 > matching:
                    inner_tag, inner_start = stack.pop()
                    close_span(inner_tag, inner_start, length, current)
                    warnings.append(f"implicitly closed <{inner_tag}> before </{source_tag}> at {match.start()}")
                _, start = stack.pop()
                close_span(tag, start, length, current)
        cursor = match.end()
    output.append(value[cursor:])
    clean = "".join(output)
    while stack:
        tag, start = stack.pop()
        close_span(tag, start, len(clean), clean)
        warnings.append(f"implicitly closed <{tag}> at end of document")
    spans.sort(key=lambda item: (item.start, item.end, item.tag))
    for span in spans:
        if clean[span.start:span.end] != span.text:
            raise AssertionError("inline-tag offset drift")
    return clean, tuple(spans), tuple(warnings)


def _read_part(path: Path, part_index: int) -> Iterable[SourceDocument]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rows = csv.DictReader(stream, dialect=dialect)
        if rows.fieldnames != ["id", "text"]:
            raise ValueError(f"unexpected RuSERRC schema in {path}: {rows.fieldnames}")
        for row_index, row in enumerate(rows, start=1):
            source_id, tagged = str(row["id"]).strip(), str(row["text"] or "")
            if not source_id or not tagged.strip():
                raise ValueError(f"empty id/text in {path.name} row {row_index}")
            clean, spans, warnings = parse_tagged_text(tagged)
            yield SourceDocument(f"ruserrc:p{part_index}:{source_id}", source_id, path.name,
                                 tagged, clean, spans, warnings)


def load_official_parts(source_dir: Path) -> tuple[SourceDocument, ...]:
    documents: list[SourceDocument] = []
    for part_index, name in enumerate(OFFICIAL_PARTS, start=1):
        path = source_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing official RuSERRC part: {path}")
        documents.extend(_read_part(path, part_index))
    ids = [item.document_id for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate composite document_id")
    if len(documents) != 591:
        raise ValueError(f"expected all 591 official RuSERRC documents, found {len(documents)}")
    return tuple(documents)



def load_structured_csv(path: Path) -> tuple[SourceDocument, ...]:
    """Load the complete local 591-row mirror while ignoring its derived aspect columns."""
    documents: list[SourceDocument] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter=";")
        if not rows.fieldnames or not {"id", "text"}.issubset(rows.fieldnames):
            raise ValueError(f"unexpected structured RuSERRC schema: {rows.fieldnames}")
        for row_index, row in enumerate(rows, start=1):
            source_id = str(row["id"]).strip()
            tagged = str(row["text"] or "")
            if not source_id or not tagged.strip():
                raise ValueError(f"empty id/text in {path} row {row_index}")
            clean, spans, warnings = parse_tagged_text(tagged)
            documents.append(SourceDocument(
                document_id=f"ruserrc:local:{row_index:04d}:{source_id}",
                source_id=source_id,
                source_part=Path(path).name,
                tagged_text=tagged,
                clean_text=clean,
                spans=spans,
                parse_warnings=warnings,
            ))
    if len(documents) != 591:
        raise ValueError(f"expected all 591 RuSERRC documents, found {len(documents)}")
    return tuple(documents)

def answers_by_aspect(document: SourceDocument) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for span in document.spans:
        aspect = ASPECT_SPECS[span.tag][0]
        answer = normalize_entity(span.text)
        if answer:
            grouped[aspect].append(answer)
    return {aspect: tuple(values) for aspect, values in grouped.items()}
