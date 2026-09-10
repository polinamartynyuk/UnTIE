"""S0: silver overlay audit before V3.1 training."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3_1_auto_prep.silver_roles import p_silver_target_ids, tm_silver_ids


def s0_audit(
    *,
    records: Sequence[InstanceRecord],
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
    b0_rows: Sequence[Mapping[str, Any]],
    b0_seed: int = 42,
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    log = progress or ProgressLogger()
    status_aspect: Counter[str] = Counter()
    status_subtype: Counter[str] = Counter()
    status_owner: Counter[str] = Counter()
    status_conf: Counter[str] = Counter()
    status_cluster: Counter[str] = Counter()
    for item in records:
        for chunk in item.chunks:
            row = overlay.get((item.question_id, chunk.chunk_id))
            if not row:
                continue
            status = str(row.get("silver_target_status") or "missing")
            status_aspect[f"{status}|{item.aspect}"] += 1
            status_subtype[f"{status}|{row.get('source_ambiguous_subtype')}"] += 1
            status_conf[f"{status}|{row.get('silver_confidence')}"] += 1
            status_cluster[f"{status}|{row.get('evidence_cluster_id')}"] += 1
            for owner in row.get("silver_owner_roles") or ["none"]:
                status_owner[f"{status}|{owner}"] += 1
    ranked = [dict(row) for row in b0_rows if int(row.get("seed") or b0_seed) == int(b0_seed)]
    if not ranked:
        seen = {}
        for row in b0_rows:
            seen[str(row["question_id"])] = dict(row)
        ranked = list(seen.values())
    by_q = {item.question_id: item for item in records}
    top_stats = {1: Counter(), 5: Counter(), 10: Counter()}
    both = 0
    for index, row in enumerate(ranked, start=1):
        log.count(index, max(len(ranked), 1), item="s0_b0_shortlist")
        item = by_q.get(str(row["question_id"]))
        if item is None:
            continue
        ids = [int(cid) for cid in (row.get("ranked_ids") or [])]
        p_ids = p_silver_target_ids(item, overlay)
        tm_ids = tm_silver_ids(item, overlay)
        if p_ids and tm_ids:
            both += 1
        for k in (1, 5, 10):
            top = set(ids[:k])
            top_stats[k]["P_silver_target"] += len(top & p_ids)
            top_stats[k]["TM_silver"] += len(top & tm_ids)
            top_stats[k]["unresolved"] += sum(
                1
                for cid in top
                if (overlay.get((item.question_id, cid)) or {}).get("silver_target_status") == "silver_unresolved"
            )
            top_stats[k]["questions_with_both"] += int(bool(top & p_ids) and bool(top & tm_ids))
    return {
        "n_overlay_rows": sum(1 for item in records for chunk in item.chunks if overlay.get((item.question_id, chunk.chunk_id))),
        "status_x_aspect": dict(status_aspect),
        "status_x_subtype": dict(status_subtype),
        "status_x_owner": dict(status_owner),
        "status_x_confidence": dict(status_conf),
        "n_status_x_cluster": len(status_cluster),
        "n_questions_with_target_and_tm": both,
        "b0_shortlist": {f"top{k}": dict(counter) for k, counter in top_stats.items()},
        "target_vs_tm_present_in_b0_shortlist": top_stats[5]["questions_with_both"] > 0,
        "note": "S0 uses historical V3 B0 OOF when available; it does not train a new ranker",
    }
