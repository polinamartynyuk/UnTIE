"""Source-only SciREX aspect QA tables and QA-input rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..config import PipelineConfig
from ..keyword_tuning import deterministic_document_split
from ..protocols import WordTokenizer
from .chunk_index import IndexedChunk, build_indexed_chunks, resolve_context_budget
from .config import AspectQAPreprocessConfig
from .mentions import GoldMention, find_lexical_mentions
from .normalize import GoldEntity, deduplicate_entities, identity_key, normalize_entity
from .source import SourceRecord, TextSegment, parse_source_record, parse_structural_segments


def clean_document_text(raw_text: str) -> str:
    """Deterministic display form; mention offsets always point at raw_text."""
    return normalize_entity(raw_text)


def mention_in_bibliography(mention: GoldMention, segments: Sequence[TextSegment]) -> bool:
    return any(
        segment.type == "bibliography"
        and segment.raw_start <= mention.char_start
        and mention.char_end <= segment.raw_end
        for segment in segments
    )


def chunks_covering_span(
    chunks: Sequence[IndexedChunk],
    char_start: int,
    char_end: int,
) -> list[IndexedChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.char_start < char_end and chunk.char_end > char_start
    ]


@dataclass
class AspectInstance:
    document_id: str
    aspect: str
    question: str
    status: str
    raw_gold_answers: list[str]
    canonical_gold_answers: list[str]
    extractable_gold_answers: list[str]
    unextractable_gold_answers: list[str]
    external_enrichment: list[dict[str, Any]]
    enrichment_candidates: list[dict[str, Any]]
    chunk_ids: list[int]
    gold_supporting_chunk_ids: list[int]
    entities: list[GoldEntity] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    record: SourceRecord
    split: str
    segments: list[TextSegment]
    chunks: list[IndexedChunk]
    instances: list[AspectInstance]
    mentions: list[dict[str, Any]]
    chunk_max_tokens: int
    overlap_tokens: int


def _aspect_status(extractable: Sequence[str], unextractable: Sequence[str]) -> str:
    if extractable and not unextractable:
        return "ready"
    if extractable and unextractable:
        return "partially_extractable"
    return "no_extractable_gold"


def process_aspect(
    *,
    document_id: str,
    aspect: str,
    question: str,
    raw_values: Sequence[str],
    text: str,
    chunks: Sequence[IndexedChunk],
    segments: Sequence[TextSegment],
    include_bibliography_in_qa: bool,
) -> tuple[AspectInstance, list[dict[str, Any]]]:
    raw_gold = [str(item) for item in raw_values]
    entities = deduplicate_entities(raw_gold)
    mention_rows: list[dict[str, Any]] = []
    extractable: list[str] = []
    unextractable: list[str] = []
    supporting: set[int] = set()

    for entity in entities:
        found = find_lexical_mentions(
            text,
            canonical=entity.canonical,
            raw_variants=entity.raw_variants,
            normalized_variants=entity.normalized_variants,
            mention_prefix=f"{document_id}:{aspect}:{entity.identity_key}",
        )
        eligible_mentions: list[GoldMention] = []
        for mention in found:
            in_bib = mention_in_bibliography(mention, segments)
            covering = chunks_covering_span(chunks, mention.char_start, mention.char_end)
            evidence_ok = include_bibliography_in_qa or not in_bib
            chunk_ids = [
                chunk.chunk_id
                for chunk in covering
                if include_bibliography_in_qa or chunk.eligible_for_qa
            ]
            mention_rows.append(
                {
                    "document_id": document_id,
                    "aspect": aspect,
                    "canonical_answer": entity.canonical,
                    "raw_answer": mention.raw_entity,
                    "mention_id": mention.mention_id,
                    "matched_text": mention.matched_text,
                    "char_start": mention.char_start,
                    "char_end": mention.char_end,
                    "chunk_ids": chunk_ids,
                    "in_bibliography": in_bib,
                }
            )
            if evidence_ok:
                eligible_mentions.append(mention)
                for chunk in covering:
                    if include_bibliography_in_qa or chunk.eligible_for_qa:
                        supporting.add(chunk.chunk_id)
        if eligible_mentions:
            extractable.append(entity.canonical)
        else:
            unextractable.append(entity.canonical)

    instance = AspectInstance(
        document_id=document_id,
        aspect=aspect,
        question=question,
        status=_aspect_status(extractable, unextractable),
        raw_gold_answers=raw_gold,
        canonical_gold_answers=[entity.canonical for entity in entities],
        extractable_gold_answers=extractable,
        unextractable_gold_answers=unextractable,
        external_enrichment=[],
        enrichment_candidates=[],
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        gold_supporting_chunk_ids=sorted(supporting),
        entities=entities,
    )
    return instance, mention_rows


def process_document(
    raw: Mapping[str, Any],
    *,
    tokenizer: WordTokenizer,
    config: AspectQAPreprocessConfig,
    pipeline_config: PipelineConfig,
    chunk_max_tokens: int,
    split: str,
) -> ProcessedDocument:
    record = parse_source_record(raw)
    segments = parse_structural_segments(record.original_text)
    overlap = (
        config.overlap_tokens
        if config.overlap_tokens is not None
        else pipeline_config.overlap_tokens
    )
    chunks = build_indexed_chunks(
        record.original_text,
        tokenizer,
        max_tokens=chunk_max_tokens,
        overlap_tokens=overlap,
        segments=segments,
        include_bibliography_in_qa=config.include_bibliography_in_qa,
    )
    instances: list[AspectInstance] = []
    mentions: list[dict[str, Any]] = []
    field_values = {
        "methods": record.methods,
        "tasks": record.tasks,
        "metrics": record.metrics,
        "materials": record.materials,
    }
    for aspect, field_name in config.aspect_field_map.items():
        instance, rows = process_aspect(
            document_id=record.doc_id,
            aspect=aspect,
            question=config.aspect_questions[aspect],
            raw_values=field_values[field_name],
            text=record.original_text,
            chunks=chunks,
            segments=segments,
            include_bibliography_in_qa=config.include_bibliography_in_qa,
        )
        instances.append(instance)
        mentions.extend(rows)
    return ProcessedDocument(
        record=record,
        split=split,
        segments=segments,
        chunks=chunks,
        instances=instances,
        mentions=mentions,
        chunk_max_tokens=chunk_max_tokens,
        overlap_tokens=overlap,
    )


def assign_splits(
    doc_ids: Iterable[str],
    config: AspectQAPreprocessConfig,
) -> dict[str, str]:
    if config.fixed_split:
        name = str(config.fixed_split)
        if name not in {"train", "dev", "test"}:
            raise ValueError(f"fixed_split must be train, dev, or test, got {name!r}")
        return {str(doc_id): name for doc_id in doc_ids}
    grouped = deterministic_document_split(
        doc_ids,
        seed=config.split_seed,
        train_fraction=config.train_fraction,
        dev_fraction=config.dev_fraction,
    )
    mapping: dict[str, str] = {}
    for split_name, ids in grouped.items():
        for doc_id in ids:
            mapping[str(doc_id)] = split_name
    return mapping


def resolve_chunk_budget(
    tokenizer: WordTokenizer,
    config: AspectQAPreprocessConfig,
    pipeline_config: PipelineConfig,
) -> int:
    configured = config.chunk_max_tokens or pipeline_config.chunk_max_tokens
    return resolve_context_budget(
        tokenizer,
        list(config.aspect_questions.values()),
        configured_max_tokens=configured,
        model_max_length=config.model_max_length,
    )


def source_metadata_row(processed: ProcessedDocument) -> dict[str, Any]:
    record = processed.record
    return {
        "doc_id": record.doc_id,
        "split": processed.split,
        "methods": list(record.methods),
        "tasks": list(record.tasks),
        "metrics": list(record.metrics),
        "materials": list(record.materials),
        "score": list(record.score),
        "extra_fields": list(record.extra_fields),
    }


def document_row(processed: ProcessedDocument) -> dict[str, Any]:
    return {
        "document_id": processed.record.doc_id,
        "split": processed.split,
        "raw_text": processed.record.original_text,
        "clean_text": clean_document_text(processed.record.original_text),
        "n_chunks": len(processed.chunks),
        "n_segments": len(processed.segments),
        "chunk_max_tokens": processed.chunk_max_tokens,
        "overlap_tokens": processed.overlap_tokens,
    }


def chunk_rows(processed: ProcessedDocument) -> list[dict[str, Any]]:
    return [
        {
            "document_id": processed.record.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_text": chunk.text,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_start": chunk.token_start,
            "token_end": chunk.token_end,
            "section_type": chunk.section_type,
            "section_title": chunk.section_title,
            "eligible_for_qa": chunk.eligible_for_qa,
            "segment_ids": list(chunk.segment_ids),
            "token_count": chunk.token_count,
        }
        for chunk in processed.chunks
    ]


def aspect_instance_rows(processed: ProcessedDocument) -> list[dict[str, Any]]:
    return [
        {
            "document_id": instance.document_id,
            "split": processed.split,
            "aspect": instance.aspect,
            "question": instance.question,
            "status": instance.status,
            "raw_gold_answers": list(instance.raw_gold_answers),
            "canonical_gold_answers": list(instance.canonical_gold_answers),
            "extractable_gold_answers": list(instance.extractable_gold_answers),
            "unextractable_gold_answers": list(instance.unextractable_gold_answers),
            "external_enrichment": list(instance.external_enrichment),
            "enrichment_candidates": list(instance.enrichment_candidates),
            "chunk_ids": list(instance.chunk_ids),
            "gold_supporting_chunk_ids": list(instance.gold_supporting_chunk_ids),
        }
        for instance in processed.instances
    ]


def qa_input_rows(processed: ProcessedDocument) -> list[dict[str, Any]]:
    """ready / partially_extractable × eligible chunks. no_extractable_gold is excluded."""
    chunks = {chunk.chunk_id: chunk for chunk in processed.chunks}
    rows: list[dict[str, Any]] = []
    supporting_lookup = {
        instance.aspect: set(instance.gold_supporting_chunk_ids)
        for instance in processed.instances
    }
    for instance in processed.instances:
        if instance.status == "no_extractable_gold":
            continue
        for chunk_id in instance.chunk_ids:
            chunk = chunks[chunk_id]
            if not chunk.eligible_for_qa:
                continue
            rows.append(
                {
                    "document_id": instance.document_id,
                    "split": processed.split,
                    "aspect": instance.aspect,
                    "question": instance.question,
                    "status": instance.status,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "extractable_gold_answers": list(instance.extractable_gold_answers),
                    "gold_supporting": chunk.chunk_id in supporting_lookup[instance.aspect],
                    "enrichment_mode": "E0",
                }
            )
    return rows


def stable_sample(doc_ids: Sequence[str], k: int, *, seed: int = 42) -> list[str]:
    ranked = sorted(
        {str(doc_id) for doc_id in doc_ids},
        key=lambda doc_id: (
            hashlib.sha256(f"{seed}\0manual\0{doc_id}".encode("utf-8")).digest(),
            doc_id,
        ),
    )
    return ranked[:k]


def canonical_from_raw(raw_values: Sequence[str]) -> set[str]:
    return {identity_key(value) for value in raw_values if str(value).strip()}
