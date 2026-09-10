"""B1-Silver / R1-Silver training helpers. Reuse V3 trainers; cluster-aware weights."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.answerability import TrainPair
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.answerability_v2 import WeightedPair
from experiments.qa_chunk_selection.v2.hn_flags import flag_index, is_training_eligible_hn
from experiments.qa_chunk_selection.v3.train import train_bce_v3, train_pairwise_v3
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import p_silver_target_ids, tm_silver_ids


def train_role_silver(item: InstanceRecord, chunk, overlay: Mapping, flags) -> str | None:
    if chunk.chunk_id in p_silver_target_ids(item, overlay):
        return "positive"
    if chunk.chunk_id in tm_silver_ids(item, overlay):
        return "target_mismatch"
    if chunk.label == "hard_negative" and is_training_eligible_hn(flags, item.question_id, chunk.chunk_id):
        return "hard_negative"
    if chunk.label == "easy_negative":
        return "easy_negative"
    return None


def unique_question_pairs_silver(
    records: Sequence[InstanceRecord],
    overlay,
    flags,
    *,
    ratio: tuple[int, int, int],
    easy_cap: int,
    allowed_chunk_ids: Mapping[str, set[int]] | None = None,
) -> tuple[list[WeightedPair], dict[str, Any]]:
    pos_n, hn_n, easy_n = ratio
    tm_n = max(hn_n, 1)
    total_w = float(pos_n + hn_n + tm_n + easy_n) or 1.0
    w_pos, w_hn, w_tm, w_easy = pos_n / total_w, hn_n / total_w, tm_n / total_w, easy_n / total_w
    questions = [item for item in records if p_silver_target_ids(item, overlay)]
    n_q = max(len(questions), 1)
    selected: list[WeightedPair] = []
    counts = defaultdict(int)
    for item in questions:
        by_role: dict[str, list] = defaultdict(list)
        allow = allowed_chunk_ids.get(item.question_id) if allowed_chunk_ids else None
        easies = 0
        for chunk in item.chunks:
            if allow is not None and chunk.chunk_id not in allow:
                continue
            role = train_role_silver(item, chunk, overlay, flags)
            if role is None:
                continue
            if role == "easy_negative":
                if easies >= int(easy_cap):
                    continue
                easies += 1
            by_role[role].append(chunk)
        classes = []
        if by_role["positive"]:
            classes.append((by_role["positive"], 1, "positive", w_pos))
        if by_role["hard_negative"]:
            classes.append((by_role["hard_negative"], 0, "hard_negative", w_hn))
        if by_role["target_mismatch"]:
            classes.append((by_role["target_mismatch"], 0, "target_mismatch", w_tm))
        if by_role["easy_negative"]:
            classes.append((by_role["easy_negative"], 0, "easy_negative", w_easy))
        share_sum = sum(share for *_rest, share in classes) or 1.0
        q_weight = 1.0 / n_q
        for chunks, label, kind, share in classes:
            clusters: dict[str, list] = defaultdict(list)
            for chunk in chunks:
                row = overlay.get((item.question_id, chunk.chunk_id)) or {}
                cluster = str(row.get("evidence_cluster_id") or f"{item.question_id}:{chunk.chunk_id}")
                clusters[cluster].append(chunk)
            n_clusters = max(len(clusters), 1)
            for cluster_chunks in clusters.values():
                representative = max(cluster_chunks, key=lambda chunk: len(chunk.text))
                unit = q_weight * (share / share_sum) / n_clusters
                pair = TrainPair(
                    question=item.question,
                    text=representative.text,
                    label=label,
                    document_id=item.document_id,
                    question_id=item.question_id,
                    chunk_id=representative.chunk_id,
                    kind=kind,
                )
                selected.append(WeightedPair(pair, unit))
                counts[kind] += 1
    report = {
        "n_questions": n_q,
        "n_pairs": len(selected),
        "counts": dict(counts),
        "weight_formula": "question-balanced weighted BCE; one weight unit per evidence cluster",
    }
    return selected, report


def train_b1_silver(
    train_records,
    calibration_records,
    overlay,
    flags_rows,
    **kwargs,
):
    """Reuse V3 BCE trainer; overlay already mapped to V3 field names for v3_target_positives."""
    return train_bce_v3(
        train_records,
        calibration_records,
        overlay,
        flags_rows,
        tag="b1_silver",
        **kwargs,
    )


def train_r1_silver(
    train_records,
    calibration_records,
    overlay,
    flags_rows,
    **kwargs,
):
    return train_bce_v3(
        train_records,
        calibration_records,
        overlay,
        flags_rows,
        tag="r1_silver",
        **kwargs,
    )


def train_pairwise_silver(
    train_records,
    calibration_records,
    overlay,
    flags_rows,
    **kwargs,
):
    return train_pairwise_v3(
        train_records,
        calibration_records,
        overlay,
        flags_rows,
        **kwargs,
    )


def flag_index_rows(flags_rows: Sequence[dict[str, Any]]):
    return flag_index(flags_rows)
