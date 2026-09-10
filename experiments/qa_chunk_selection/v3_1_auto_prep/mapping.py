"""Deterministic consensus policy. No trainable target classifier in data prep."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .gold_relation import local_token_f1


def normalized_pair(left: str, right: str) -> bool:
    from experiments.qa_chunk_selection.metrics import normalized_answer

    return bool(left) and normalized_answer(left) == normalized_answer(right)


def paraphrase_agreement(spans: Sequence[str], canonical: str) -> dict[str, float]:
    values = [str(item or "") for item in spans if str(item or "").strip()]
    if not values:
        return {"span_text_agreement": 0.0, "answer_value_agreement": 0.0, "mean_pairwise_f1": 0.0, "n": 0}
    same = sum(1 for item in values if normalized_pair(item, canonical) or local_token_f1(item, [canonical]) >= 0.5)
    pair_f1 = []
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            pair_f1.append(local_token_f1(left, [right]))
    return {
        "span_text_agreement": same / len(values),
        "answer_value_agreement": same / len(values),
        "mean_pairwise_f1": (sum(pair_f1) / len(pair_f1)) if pair_f1 else 1.0,
        "n": len(values),
    }


def map_silver(
    *,
    qa_plausible: bool,
    consistency: float,
    target_score: float,
    external_score: float,
    gold: Mapping[str, Any],
    discourse: Mapping[str, Any],
    paraphrase_spans: Sequence[str],
    canonical_span: str,
    target_span: str,
    external_span: str,
    tau_consistency: float,
    delta: float,
    far_from_threshold: bool,
    near_threshold: bool,
) -> dict[str, Any]:
    flags: list[str] = []
    if not qa_plausible:
        return _row("silver_unresolved", "low", True, "stage_a_noncandidate", flags, gold, 0.0)
    target_better = float(target_score) > float(external_score) + float(delta)
    external_better = float(external_score) > float(target_score) + float(delta)
    current_ctx = float(discourse.get("current_paper_marker_score") or 0.0) > 0
    external_ctx = bool(discourse.get("independent_external_context"))
    para_conflict = len({_norm(item) for item in paraphrase_spans if item}) > 1 and consistency < tau_consistency
    mixed = current_ctx and external_ctx
    same_target_ext = bool(canonical_span) and (
        normalized_pair(canonical_span, target_span) or normalized_pair(canonical_span, external_span)
    )
    if para_conflict:
        flags.append("paraphrase_conflict")
    if mixed and not (target_better and gold.get("allowed_for_primary_te")):
        flags.append("mixed_context")
    if target_better and external_better:
        flags.append("ownership_conflict")

    ownership_margin = float(target_score) - float(external_score)
    if (
        qa_plausible
        and consistency >= tau_consistency
        and target_better
        and not external_better
        and gold.get("allowed_for_primary_te")
        and "paraphrase_conflict" not in flags
    ):
        status, conf = "silver_target_evidence", "high" if far_from_threshold and not mixed else "medium"
        if mixed:
            return _row(status, conf, False, "current_sufficient_with_competing", flags, gold, ownership_margin, competing=True)
        if conf == "low":
            return _row("silver_unresolved", "low", True, "low_confidence", flags, gold, ownership_margin)
        return _row(status, conf, False, "consensus_target", flags, gold, ownership_margin)

    if gold.get("new_value_candidate") and target_better and consistency >= tau_consistency:
        return _row("silver_unresolved", "medium", True, "new_value_candidate_not_primary", flags, gold, ownership_margin)

    independent = external_ctx or (same_target_ext and external_better)
    if qa_plausible and consistency >= tau_consistency and external_better and independent and not target_better:
        status, conf = "silver_target_mismatch", "high" if far_from_threshold else "medium"
        if conf == "low":
            return _row("silver_unresolved", "low", True, "low_confidence", flags, gold, ownership_margin)
        return _row(status, conf, False, "consensus_external", flags, gold, ownership_margin)

    if qa_plausible and current_ctx and not external_better and consistency < tau_consistency:
        return _row("silver_partial_support", "medium", True, "partial_current", flags, gold, ownership_margin)

    if near_threshold or flags:
        return _row("silver_unresolved", "low", True, "conflict_or_near_threshold", flags, gold, ownership_margin)
    return _row("silver_unresolved", "low", True, "insufficient_consensus", flags, gold, ownership_margin)


def _row(
    status: str,
    confidence: str,
    abstained: bool,
    reason: str,
    flags: list[str],
    gold: Mapping[str, Any],
    margin: float,
    competing: bool = False,
) -> dict[str, Any]:
    if confidence == "low":
        status = "silver_unresolved"
        abstained = True
    return {
        "silver_target_status": status,
        "silver_confidence": confidence,
        "abstained": abstained,
        "mapping_reason": reason,
        "contradiction_flags": flags,
        "contains_competing_values": competing,
        "qa_ownership_margin": margin,
        "qa_gold_relation": gold.get("qa_gold_relation"),
        "new_value_candidate": bool(gold.get("new_value_candidate")),
    }


def _norm(text: str) -> str:
    from experiments.qa_chunk_selection.metrics import normalized_answer

    return normalized_answer(text)
