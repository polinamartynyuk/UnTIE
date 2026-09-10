"""Isolated SciREX chunk-selection V2.

Writes only to artifacts/qa_chunk_selection_v2/. Never overwrites E0
artifacts/qa_chunk_selection/ or production UnTIE / Phase 1 trees.
"""

from .run import main

__all__ = ["main"]
