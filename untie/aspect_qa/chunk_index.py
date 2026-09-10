"""Offset-aware chunking that reuses SentenceSplitter and ChunkBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..chunking import ChunkBuilder
from ..domain import Sentence
from ..protocols import WordTokenizer
from ..text import SentenceSplitter
from .source import TextSegment


@dataclass(frozen=True)
class LocatedSentence:
    sentence: Sentence
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_count: int
    section_type: str
    section_title: str
    segment_ids: tuple[int, ...]
    eligible_for_qa: bool


def _surface_token(token: str) -> str:
    if token.startswith("Ġ"):
        return token[1:]
    if token.startswith("▁"):
        return token[1:]
    if token.startswith("##"):
        return token[2:]
    return token


def locate_sentence(text: str, piece: str, cursor: int) -> tuple[int, int]:
    index = text.find(piece, cursor)
    if index >= 0:
        return index, index + len(piece)
    index = text.find(piece)
    if index >= 0:
        return index, index + len(piece)
    raise ValueError(f"sentence produced by SentenceSplitter was not found in original_text: {piece[:80]!r}")


def _token_windows_from_offsets(
    text: str,
    tokenizer: Any,
    usable_size: int,
) -> list[tuple[str, int, int, tuple[str, ...]]] | None:
    if not callable(tokenizer):
        return None
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
    except TypeError:
        return None
    offsets = encoded.get("offset_mapping")
    tokens = tokenizer.tokenize(text)
    if offsets is None:
        return None
    pairs = [(token, int(start), int(end)) for token, (start, end) in zip(tokens, offsets) if end > start]
    if not pairs:
        return None
    windows: list[tuple[str, int, int, tuple[str, ...]]] = []
    for start in range(0, len(pairs), usable_size):
        part = pairs[start : start + usable_size]
        char_start = part[0][1]
        char_end = part[-1][2]
        windows.append(
            (
                text[char_start:char_end],
                char_start,
                char_end,
                tuple(item[0] for item in part),
            )
        )
    return windows


def _token_windows_by_search(
    text: str,
    tokens: tuple[str, ...],
    tokenizer: WordTokenizer,
    usable_size: int,
) -> list[tuple[str, int, int, tuple[str, ...]]]:
    windows: list[tuple[str, int, int, tuple[str, ...]]] = []
    cursor = 0
    for start in range(0, len(tokens), usable_size):
        part_tokens = tokens[start : start + usable_size]
        reconstructed = tokenizer.convert_tokens_to_string(part_tokens)
        local = text.find(reconstructed, cursor)
        if local >= 0:
            windows.append(
                (reconstructed, local, local + len(reconstructed), part_tokens)
            )
            cursor = local + len(reconstructed)
            continue
        part_start = cursor
        search_at = cursor
        for token in part_tokens:
            surface = _surface_token(token)
            if not surface:
                continue
            found = text.find(surface, search_at)
            if found < 0:
                found = text.casefold().find(surface.casefold(), search_at)
                if found < 0:
                    found = search_at
                    search_at = min(len(text), search_at + 1)
                    continue
            if part_start == cursor and found >= cursor:
                part_start = found
            search_at = found + max(len(surface), 1)
        part_end = max(part_start, search_at)
        windows.append((text[part_start:part_end], part_start, part_end, part_tokens))
        cursor = part_end
    return windows


def _split_long_sentence(
    text: str,
    number: int,
    tokenizer: WordTokenizer,
    max_tokens: int,
) -> list[tuple[str, int, int, tuple[str, ...]]]:
    tokens = tuple(tokenizer.tokenize(text))
    if len(tokens) <= max_tokens:
        return [(text, 0, len(text), tokens)]
    usable_size = max(1, max_tokens - 2)
    windows = _token_windows_from_offsets(text, tokenizer, usable_size)
    if windows is None:
        windows = _token_windows_by_search(text, tokens, tokenizer, usable_size)
    return windows


def locate_sentences(
    text: str,
    tokenizer: WordTokenizer,
    *,
    splitter: SentenceSplitter | None = None,
    max_tokens: int,
) -> list[LocatedSentence]:
    splitter = splitter or SentenceSplitter()
    pieces = splitter.split(text)
    located: list[LocatedSentence] = []
    cursor = 0
    serial = 0
    token_cursor = 0
    for number, piece in enumerate(pieces):
        start, end = locate_sentence(text, piece, cursor)
        cursor = end
        windows = _split_long_sentence(piece, number, tokenizer, max_tokens)
        for part_text, rel_start, rel_end, tokens in windows:
            abs_start = start + rel_start
            abs_end = start + rel_end
            if text[abs_start:abs_end] != part_text and part_text in text[start:end]:
                local = text.find(part_text, start, end)
                abs_start, abs_end = local, local + len(part_text)
            sentence = Sentence(text[abs_start:abs_end], serial, tokens, tuple(tokenizer.encode(text[abs_start:abs_end])))
            token_end = token_cursor + len(tokens)
            located.append(
                LocatedSentence(
                    sentence=sentence,
                    char_start=abs_start,
                    char_end=abs_end,
                    token_start=token_cursor,
                    token_end=token_end,
                )
            )
            token_cursor = token_end
            serial += 1
    return located


def _segments_for_span(
    segments: Sequence[TextSegment],
    char_start: int,
    char_end: int,
) -> tuple[TextSegment, ...]:
    hit = [
        segment
        for segment in segments
        if segment.raw_start < char_end and segment.raw_end > char_start
    ]
    return tuple(hit)


def _span_inside(segments: Sequence[TextSegment], char_start: int, char_end: int) -> bool:
    if not segments:
        return False
    return any(segment.raw_start <= char_start and char_end <= segment.raw_end for segment in segments)


def build_indexed_chunks(
    text: str,
    tokenizer: WordTokenizer,
    *,
    max_tokens: int,
    overlap_tokens: int,
    segments: Sequence[TextSegment],
    include_bibliography_in_qa: bool,
    splitter: SentenceSplitter | None = None,
) -> list[IndexedChunk]:
    """One document → one chunking. Aspect instances reuse these chunk_ids."""
    located = locate_sentences(text, tokenizer, splitter=splitter, max_tokens=max_tokens)
    if not located:
        return []
    by_id = {id(item.sentence): item for item in located}
    builder = ChunkBuilder(tokenizer=tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunks = builder.build([item.sentence for item in located])
    bibliography = tuple(item for item in segments if item.type == "bibliography")
    indexed: list[IndexedChunk] = []
    for chunk_id, chunk in enumerate(chunks):
        parts = [by_id[id(sentence)] for sentence in chunk.sentences]
        char_start = min(item.char_start for item in parts)
        char_end = max(item.char_end for item in parts)
        overlapping = _segments_for_span(segments, char_start, char_end)
        if len(overlapping) == 1:
            section_type = overlapping[0].type
            section_title = overlapping[0].title
        else:
            section_type = ""
            section_title = ""
        fully_bibliography = bool(bibliography) and _span_inside(bibliography, char_start, char_end)
        eligible = include_bibliography_in_qa or not fully_bibliography
        indexed.append(
            IndexedChunk(
                chunk_id=chunk_id,
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_start=parts[0].token_start,
                token_end=parts[-1].token_end,
                token_count=sum(item.sentence.token_count for item in parts),
                section_type=section_type,
                section_title=section_title,
                segment_ids=tuple(item.segment_id for item in overlapping),
                eligible_for_qa=eligible,
            )
        )
    return indexed


def resolve_context_budget(
    tokenizer: WordTokenizer,
    questions: Sequence[str],
    *,
    configured_max_tokens: int,
    model_max_length: int = 512,
) -> int:
    """Chunk size is limited by the longest fixed aspect question, not per aspect."""
    model_max = getattr(tokenizer, "model_max_length", None) or model_max_length
    try:
        model_max = int(model_max)
    except (TypeError, ValueError):
        model_max = model_max_length
    if model_max > 10_000:
        model_max = model_max_length
    try:
        special = int(tokenizer.num_special_tokens_to_add(pair=True))  # type: ignore[attr-defined]
    except Exception:
        special = 3
    longest_question = max((len(tokenizer.tokenize(question)) for question in questions), default=0)
    budget = model_max - longest_question - special
    if budget < 8:
        raise ValueError(
            f"QA context budget too small: model_max={model_max}, "
            f"question_tokens={longest_question}, special={special}"
        )
    return min(int(configured_max_tokens), budget)
