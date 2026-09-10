"""Isolated V3.1-Silver ranking.

Writes only to artifacts/qa_chunk_selection_v3_1/.
Never overwrites E0, V2, V3, manual prep, or automated silver trees.
"""

from .run import main

__all__ = ["main"]
