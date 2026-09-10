"""Deterministic discourse / structure features. Never a final classifier."""

from __future__ import annotations

import re
from typing import Any

CURRENT_RE = re.compile(
    r"\b(?:we|our|in this work|in this paper|we propose|we introduce|we use|we evaluate|we train|our method|our model|our experiments)\b",
    re.I,
)
EXTERNAL_RE = re.compile(
    r"\b(?:et al\.|previous work|prior work|baseline|compared with|compared to|following|proposed by|reported by|according to)\b",
    re.I,
)
CITATION_RE = re.compile(
    r"(?:\[\d+[,\-\s\d]*\]|\(\s*[A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)?(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\s*\))",
)


def discourse_features(
    text: str,
    *,
    section_title: str = "",
    span_start: int | None = None,
    span_end: int | None = None,
    position: int = 0,
    n_chunks: int = 1,
) -> dict[str, Any]:
    blob = f"{section_title}\n{text or ''}"
    current = len(CURRENT_RE.findall(blob))
    external = len(EXTERNAL_RE.findall(blob))
    citations = list(CITATION_RE.finditer(blob))
    citation_distance = None
    if citations and span_start is not None:
        citation_distance = min(abs(match.start() - int(span_start)) for match in citations)
    section = (section_title or "").casefold()
    related = any(token in section for token in ("related", "background", "prior", "previous"))
    return {
        "current_paper_marker_score": float(current),
        "external_marker_score": float(external),
        "has_citation_marker": bool(citations),
        "citation_distance": citation_distance,
        "section_type": section_title or "",
        "related_work_section": related,
        "relative_document_position": float(position) / float(max(n_chunks - 1, 1)),
        "independent_external_context": bool(external or related),
    }
