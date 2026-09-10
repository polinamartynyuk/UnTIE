"""Isolated SciREX chunk-selection experiments A–D.

Does not modify production UnTIE pipelines or rewrite frozen SciREX / Phase 1
tables. Gold labels are used only for training (train split), model selection
(dev), and evaluation — never for test-time ranking.
"""

from .run import main

__all__ = ["main"]
