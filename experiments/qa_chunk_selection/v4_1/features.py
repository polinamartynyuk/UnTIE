"""Discover and freeze the actual V4 V6 feature scale. Do not renormalize for R1."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features

from .contract import SOURCE_OVERLAP, V4_V6_WEIGHTS
from .ranking import v6_block_scores


FEATURE_SPECS = (
    {
        "name": "b0_score",
        "definition": "Historical V3-B0 OOF score of the candidate chunk",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['b0_score']",
        "normalization": "none; raw cross-encoder score",
        "dtype": "float",
    },
    {
        "name": "b0_rank",
        "definition": "1-based rank of the chunk inside B0 top-K",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['b0_rank']",
        "normalization": "none; integer rank",
        "dtype": "float",
    },
    {
        "name": "exact_frequency",
        "definition": "Count of other top-K answers matching after V4 normalize_predicted_answer",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['exact_freq']",
        "normalization": "count in 0..K-1; not divided by K",
        "dtype": "float",
    },
    {
        "name": "mean_token_f1",
        "definition": "Mean SQuAD token F1 vs other top-K answers; V4 pair_token_f1 uses 0..100",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['mean_token_f1']",
        "normalization": "none; observed V4 scale is 0..100, not 0..1",
        "dtype": "float",
    },
    {
        "name": "majority_flag",
        "definition": "1 if candidate answer matches majority_answer of the shortlist",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['majority']",
        "normalization": "binary 0/1",
        "dtype": "float",
    },
    {
        "name": "source_cluster_support",
        "definition": "Number of source-overlap clusters among chunks with the same normalized answer",
        "v4_implementation": "experiments.qa_chunk_selection.v4.v6_reader.agreement_features['source_clusters']",
        "normalization": "count; overlap threshold frozen at V4 0.6",
        "dtype": "float",
    },
)


def observed_ranges(samples: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    names = ("b0_score", "b0_rank", "exact_freq", "mean_token_f1", "majority", "source_clusters", "S_answer")
    out: dict[str, dict[str, float]] = {}
    for name in names:
        values = [float(row.get(name) or 0.0) for row in samples]
        if not values:
            continue
        out[name] = {"min": min(values), "max": max(values), "n": float(len(values))}
    return out


def collect_v6_feature_rows(
    items: Sequence[InstanceRecord],
    b0_by_qid: Mapping[str, Mapping[str, Any]],
    qa_store: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    k: int,
    min_overlap: float = SOURCE_OVERLAP,
    limit: int = 0,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for item in items:
        row = b0_by_qid.get(item.question_id)
        if row is None:
            continue
        from experiments.qa_chunk_selection.v4.shortlist import top_k_ids

        qa_by_chunk = {
            cid: qa_store[(item.question_id, cid)]
            for cid in top_k_ids(row, k)
            if (item.question_id, cid) in qa_store
        }
        if not qa_by_chunk:
            continue
        feats = agreement_features(item, row, qa_by_chunk, k=k, min_overlap=min_overlap)
        for feat in feats.values():
            payload = dict(feat)
            payload.update(v6_block_scores(feat))
            rows.append(payload)
        if limit and len(rows) >= int(limit):
            break
    return rows


def feature_contract_payload(samples: Sequence[Mapping[str, float]] | None = None) -> dict[str, Any]:
    ranges = observed_ranges(samples or [])
    features = []
    rename = {"exact_frequency": "exact_freq", "majority_flag": "majority", "source_cluster_support": "source_clusters"}
    for spec in FEATURE_SPECS:
        key = rename.get(spec["name"], spec["name"])
        item = dict(spec)
        item["observed_range"] = ranges.get(key) or ranges.get(spec["name"])
        features.append(item)
    return {
        "note": "R1 must keep this V4 scale. Normalized [0,1] features are R2-exploratory only.",
        "v4_weights": dict(V4_V6_WEIGHTS),
        "S_answer": "exact_freq + mean_token_f1 + majority  (mean_token_f1 is 0..100)",
        "S_support": "source_clusters of agreeing answers",
        "S_i": "alpha * b0_score + beta * S_answer + gamma * S_support",
        "features": features,
        "observed_ranges": ranges,
    }
