from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from untie.aspect_qa.chunk_index import IndexedChunk
from untie.aspect_qa.normalize import normalize_entity


PUNCTUATION = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "»", "..."}
OPENING_PUNCT = {"(", "[", "{", "«", "“", '"', "'"}


@dataclass(frozen=True)
class SourceMention:
    mention_id: str
    native_type: str
    aspect: str
    text: str
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None


class FixedSentenceSplitter:
    def __init__(self, sentences: Sequence[str]) -> None:
        self.sentences = tuple(sentences)

    def split(self, text: str) -> list[str]:
        del text
        return list(self.sentences)


def claim_output(path: Path) -> Path:
    resolved = Path(path).absolute()
    if any(item.is_symlink() for item in (resolved, *resolved.parents)):
        raise ValueError("output path cannot contain symlinks")
    artifacts = PROJECT_ROOT / "artifacts"
    if resolved == PROJECT_ROOT or (PROJECT_ROOT in resolved.parents and artifacts not in resolved.parents):
        raise ValueError("project-local output must be below artifacts/")
    protected = (
        PROJECT_ROOT / "datasets", PROJECT_ROOT / "untie", PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "artifacts/qa_chunk_selection", PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4",
        PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_data",
    )
    for item in protected:
        candidate = item.resolve()
        if resolved == candidate or candidate in resolved.parents or resolved in candidate.parents:
            raise ValueError(f"refusing protected output path: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"output directory must be new: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)
    return resolved


def detokenize_with_offsets(tokens: Sequence[object]) -> tuple[str, tuple[tuple[int, int], ...]]:
    sequence = [str(token) for token in tokens]
    text = ""
    offsets: list[tuple[int, int]] = []
    for index, token in enumerate(sequence):
        separator = ""
        if index and token not in PUNCTUATION and sequence[index - 1] not in OPENING_PUNCT:
            separator = " "
        text += separator
        start = len(text)
        text += token
        offsets.append((start, len(text)))
    return text, tuple(offsets)


_INLINE = re.compile(r"\[(.*?)\|([^|\]]+)\|([A-Za-z_]+)\]", re.DOTALL)


def parse_inline_annotations(
    annotated: str,
    native_map: dict[str, str],
) -> tuple[str, tuple[SourceMention, ...]]:
    output: list[str] = []
    mentions: list[SourceMention] = []
    cursor = 0
    length = 0
    for match in _INLINE.finditer(str(annotated)):
        prefix = annotated[cursor:match.start()]
        output.append(prefix)
        length += len(prefix)
        span_text, mention_id, native_type = match.groups()
        if native_type not in native_map:
            raise ValueError(f"unknown inline annotation type: {native_type}")
        start = length
        output.append(span_text)
        length += len(span_text)
        mentions.append(SourceMention(
            mention_id=str(mention_id), native_type=native_type,
            aspect=native_map[native_type], text=span_text,
            char_start=start, char_end=length,
        ))
        cursor = match.end()
    output.append(annotated[cursor:])
    text = "".join(output)
    if "[" in text and "|" in text:
        raise ValueError("unparsed inline annotation remains")
    for mention in mentions:
        if text[mention.char_start:mention.char_end] != mention.text:
            raise AssertionError("inline annotation offset drift")
    return text, tuple(mentions)


def mentions_from_token_spans(
    ner: Sequence[Sequence[Sequence[object]]],
    tokens: Sequence[object],
    offsets: Sequence[tuple[int, int]],
    native_map: dict[str, str],
) -> tuple[SourceMention, ...]:
    mentions: list[SourceMention] = []
    seen: set[tuple[int, int, str]] = set()
    for sentence_annotations in ner:
        for raw in sentence_annotations:
            start, end, native_type = int(raw[0]), int(raw[1]), str(raw[2])
            if native_type not in native_map:
                raise ValueError(f"unknown SciER entity type: {native_type}")
            if not (0 <= start <= end < len(tokens)):
                raise ValueError(f"invalid token span: {start}:{end}/{len(tokens)}")
            key = (start, end, native_type)
            if key in seen:
                continue
            seen.add(key)
            char_start, char_end = offsets[start][0], offsets[end][1]
            surface, _ = detokenize_with_offsets(tokens[start:end + 1])
            mentions.append(SourceMention(
                mention_id=f"T{len(mentions) + 1}", native_type=native_type,
                aspect=native_map[native_type], text=surface,
                char_start=char_start, char_end=char_end,
                token_start=start, token_end=end + 1,
            ))
    return tuple(mentions)


def unique_normalized(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_entity(value)
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return tuple(out)


def chunks_fully_containing(mention: SourceMention, chunks: Sequence[IndexedChunk]) -> list[IndexedChunk]:
    return [c for c in chunks if c.char_start <= mention.char_start and mention.char_end <= c.char_end]


