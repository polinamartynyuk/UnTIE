"""Silver diagnostic metrics. Never a substitute for original-gold Token F1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import hit_at_k, mean_or_nan, pairwise_pos_gt_hn
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.metrics_v3 import enrich_v3
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import p_silver_target_ids, tm_silver_ids


def enrich_silver(
    raw: Mapping[str, Any],
    gold: InstanceRecord,
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    training_overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    flags=None,
    skip_bertscore: bool = True,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    row = enrich_v3(
        raw,
        gold,
        training_overlay,
        flags=flags,
        skip_bertscore=skip_bertscore,
        ks=ks,
    )
    ranked = [int(item) for item in raw.get("ranked_ids") or []]
    scores = {int(key): float(value) for key, value in (raw.get("scores") or {}).items()}
    p_ids = sorted(p_silver_target_ids(gold, overlay))
    tm_ids = sorted(tm_silver_ids(gold, overlay))
    selected = raw.get("selected_chunk_id")
    selected_silver = overlay.get((gold.question_id, int(selected))) if selected is not None else None
    pair_tm, n_tm = pairwise_pos_gt_hn(scores, p_ids, tm_ids)
    known = selected_silver is not None
    tm_at_1 = bool(selected is not None and int(selected) in set(tm_ids))
    row.update(
        {
            "n_p_silver_target": len(p_ids),
            "n_tm_silver": len(tm_ids),
            "selected_silver_status": (selected_silver or {}).get("silver_target_status"),
            "selected_silver_confidence": (selected_silver or {}).get("silver_confidence"),
            "SilverTargetStatusCoverage@1": float(known),
            "SilverTargetMismatch@1": float(tm_at_1),
            "SilverPairwiseTargetVsMismatch": pair_tm,
            "n_silver_tm_pairs": n_tm,
            "SilverAbstentionAt1": float((selected_silver or {}).get("silver_target_status") == "silver_unresolved"),
            "SilverWrongTargetAnswer": float(tm_at_1),
            "has_target_and_tm_silver": bool(p_ids and tm_ids),
        }
    )
    for k in ks:
        top = ranked[: max(int(k), 0)]
        known_k = sum(1 for cid in top if overlay.get((gold.question_id, int(cid))))
        row[f"SilverTargetStatusSlotCoverage@{k}"] = float(known_k / max(len(top), 1)) if top else 0.0
        row[f"SilverTargetEvidenceHit@{k}"] = hit_at_k(ranked, p_ids, k)
    return row


def summarize_silver(rows: Sequence[Mapping[str, Any]], ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    payload: dict[str, Any] = {
        "n": len(rows),
        "n_unique_document_aspect": len({row.get("question_id") for row in rows}),
        "n_with_target_and_tm_silver": sum(1 for row in rows if row.get("has_target_and_tm_silver")),
        "SilverTargetStatusCoverage@1": mean_or_nan([row.get("SilverTargetStatusCoverage@1") for row in rows]),
        "SilverTargetMismatch@1_lower_bound": mean_or_nan([row.get("SilverTargetMismatch@1") for row in rows]),
        "SilverPairwiseTargetVsMismatch": mean_or_nan(
            [row.get("SilverPairwiseTargetVsMismatch") for row in rows if row.get("n_silver_tm_pairs")]
        ),
        "SilverWrongTargetAnswerRate": mean_or_nan([row.get("SilverWrongTargetAnswer") for row in rows]),
        "SilverAbstentionRate": mean_or_nan([row.get("SilverAbstentionAt1") for row in rows]),
        "note": "silver metrics are QA-based weak-supervision diagnostics, not human gold",
    }
    for k in ks:
        payload[f"SilverTargetStatusSlotCoverage@{k}"] = mean_or_nan(
            [row.get(f"SilverTargetStatusSlotCoverage@{k}") for row in rows]
        )
        payload[f"SilverTargetEvidenceHit@{k}"] = mean_or_nan([row.get(f"SilverTargetEvidenceHit@{k}") for row in rows])
    return payload
