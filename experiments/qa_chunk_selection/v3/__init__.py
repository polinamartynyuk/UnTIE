"""Isolated SciREX chunk-selection V3.

Writes only to artifacts/qa_chunk_selection_v3/. Never overwrites E0
artifacts/qa_chunk_selection/, V2 artifacts/qa_chunk_selection_v2/,
or production UnTIE / Phase 1 trees.
"""

from .run import main

__all__ = ["main"]
