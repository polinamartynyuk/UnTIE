"""Source-only SciREX preprocessing for aspect-level extractive QA."""

from .config import (
    ASPECT_FIELD_MAP,
    ASPECT_QUESTIONS,
    AspectQAPreprocessConfig,
)
from .normalize import deduplicate_entities, identity_key, normalize_entity
from .pipeline import process_document, qa_input_rows
from .source import inspect_source, load_scirex_records

__all__ = [
    "ASPECT_FIELD_MAP",
    "ASPECT_QUESTIONS",
    "AspectQAPreprocessConfig",
    "deduplicate_entities",
    "identity_key",
    "inspect_source",
    "load_scirex_records",
    "normalize_entity",
    "process_document",
    "qa_input_rows",
]
