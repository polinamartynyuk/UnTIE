"""D2-v2 ablations of relevance/structure channels. Does not include QA-proxy (that's R3)."""

from __future__ import annotations

from typing import Sequence

from .schemas import RuntimeInstance
from .signals import (
    fuse_minmax,
    relevance_scores,
    rrf_dense_bm25,
    section_id_scores,
    section_title_missing,
    section_title_scores,
)

D2_VARIANTS = ("D2a", "D2b", "D2c", "D2d", "D2e", "D2f")


def d2_scores(
    instance: RuntimeInstance,
    *,
    variant: str,
    b_scores: Sequence[float],
    encode,
    relevance,
    rrf_k: int,
    weights: dict[str, float] | None = None,
) -> list[float]:
    weights = weights or {"b": 1.0, "rrf": 0.4, "section_title": 0.2, "section_id": 0.1, "a": 0.3}
    channels = [list(b_scores)]
    used = [float(weights.get("b", 1.0))]
    if variant in {"D2b", "D2d"}:
        channels.append(rrf_dense_bm25(instance, encode, rrf_k=rrf_k))
        used.append(float(weights.get("rrf", 0.4)))
    if variant in {"D2c", "D2d", "D2f"}:
        title = section_title_scores(instance)
        ident = section_id_scores(instance)
        missing = section_title_missing(instance)
        # title channel is 0 when missing; id is a separate channel, never a silent fallback
        channels.append(title)
        used.append(float(weights.get("section_title", 0.2)))
        channels.append(ident)
        used.append(float(weights.get("section_id", 0.1)))
        del missing
    if variant in {"D2e", "D2f"}:
        channels.append(relevance_scores(instance, relevance))
        used.append(float(weights.get("a", 0.3)))
    if variant == "D2a":
        return list(b_scores)
    return fuse_minmax(channels, used)
