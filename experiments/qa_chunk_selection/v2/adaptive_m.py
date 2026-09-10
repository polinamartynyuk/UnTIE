"""C3 adaptive M. Confidence features are scale-valid (normalized margin / top1)."""

from __future__ import annotations

from typing import Sequence

from experiments.qa_chunk_selection.ranking import min_max

from .schemas import RuntimeInstance, rank_ids


def ranking_confidence(scores: Sequence[float]) -> dict[str, float]:
    if not scores:
        return {"top1": 0.0, "margin": 0.0, "candidate_count": 0.0, "normalized_top1": 0.0, "normalized_margin": 0.0}
    scaled = min_max(scores)
    order = sorted(range(len(scores)), key=lambda index: -float(scores[index]))
    top1 = float(scaled[order[0]]) if scaled else 0.0
    top2 = float(scaled[order[1]]) if len(order) > 1 and scaled else 0.0
    raw_sorted = sorted((float(item) for item in scores), reverse=True)
    return {
        "top1": float(raw_sorted[0]),
        "margin": float(raw_sorted[0] - raw_sorted[1]) if len(raw_sorted) > 1 else float(raw_sorted[0]),
        "candidate_count": float(len(scores)),
        "normalized_top1": top1,
        "normalized_margin": float(top1 - top2),
    }


def choose_m(
    scores: Sequence[float],
    *,
    tau_conf: float,
    tau_margin: float,
    m_uncertain: int,
    max_m: int,
) -> int:
    n = max(len(scores), 1)
    conf = ranking_confidence(scores)
    if n <= 1:
        return 1
    if conf["normalized_top1"] >= float(tau_conf) and conf["normalized_margin"] >= float(tau_margin):
        return 1
    return int(min(max(int(m_uncertain), 1), n, int(max_m)))


def calibration_grid() -> list[tuple[float, float, int]]:
    grid = []
    for tau_conf in (0.55, 0.7, 0.85):
        for tau_margin in (0.05, 0.15, 0.3):
            for m_uncertain in (3, 5):
                grid.append((tau_conf, tau_margin, m_uncertain))
    return grid
