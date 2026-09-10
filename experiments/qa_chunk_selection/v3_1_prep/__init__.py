"""Isolated V3.1 target-status overlay preparation.

Writes only to artifacts/qa_chunk_selection_v3_1_prep/.
Never overwrites E0, V2, or V3 experiment artifacts.
"""

from .run import main

__all__ = ["main"]
