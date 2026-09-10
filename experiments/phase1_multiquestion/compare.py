"""Compare Q0 / Q3 / Q5 from written parquet tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.phase1_scirex.tables import as_int_list


def _load(dir_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(dir_path / "qa_instances.parquet"),
        pd.read_parquet(dir_path / "qa_chunk_dataset.parquet"),
    )


def _ids(series: pd.Series) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for instance_id, values in series.items():
        for chunk_id in as_int_list(values):
            out.add((str(instance_id), int(chunk_id)))
    return out


def summarize_run(instances: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, Any]:
    n_pos = instances["robust_positive_chunk_ids"].map(lambda item: len(as_int_list(item)))
    n_hn = instances["hard_negative_chunk_ids"].map(lambda item: len(as_int_list(item)))
    n_sens = instances["question_sensitive_chunk_ids"].map(lambda item: len(as_int_list(item)))
    return {
        "instances": int(len(instances)),
        "documents": int(instances["document_id"].nunique()),
        "positive_chunks_any_correct": int(chunks["any_correct"].sum()) if "any_correct" in chunks else None,
        "canonical_correct_chunks": int(
            ((chunks["n_correct"] > 0) & (chunks.get("n_questions", 1) >= 1)).sum()
        ),
        "robust_positive_chunks": int((chunks["primary_class"] == "robust_positive").sum()),
        "question_sensitive_chunks": int((chunks["primary_class"] == "question_sensitive").sum()),
        "robust_negative_chunks": int((chunks["consistency_class"] == "robust_negative").sum()),
        "hard_negative_chunks": int((chunks["primary_class"] == "hard_negative").sum()),
        "no_robust_positive_instances": int((instances["instance_status"] == "no_robust_positive").sum()),
        "no_robust_positive_rate": float((instances["instance_status"] == "no_robust_positive").mean()),
        "pos_hn_instances": int(((n_pos > 0) & (n_hn > 0)).sum()),
        "mean_qa_consistency": float(chunks["qa_consistency"].mean()) if len(chunks) else 0.0,
        "question_sensitive_instances": int((n_sens > 0).sum()),
        "by_aspect": {
            str(aspect): {
                "robust_positive": int((g["primary_class"] == "robust_positive").sum()),
                "question_sensitive": int((g["primary_class"] == "question_sensitive").sum()),
                "hard_negative": int((g["primary_class"] == "hard_negative").sum()),
                "mean_qa_consistency": float(g["qa_consistency"].mean()),
            }
            for aspect, g in chunks.groupby("aspect")
        },
    }


def compare_runs(paths: dict[str, Path]) -> dict[str, Any]:
    loaded = {name: _load(path) for name, path in paths.items()}
    summaries = {name: summarize_run(inst, chunks) for name, (inst, chunks) in loaded.items()}
    hn_sets = {}
    pos_sets = {}
    for name, (inst, chunks) in loaded.items():
        hn_sets[name] = set(
            zip(
                chunks.loc[chunks["primary_class"] == "hard_negative", "qa_instance_id"],
                chunks.loc[chunks["primary_class"] == "hard_negative", "chunk_id"],
            )
        )
        pos_sets[name] = set(
            zip(
                chunks.loc[chunks["primary_class"] == "robust_positive", "qa_instance_id"],
                chunks.loc[chunks["primary_class"] == "robust_positive", "chunk_id"],
            )
        )
    overlap = {}
    names = list(paths)
    for left in names:
        for right in names:
            if left >= right:
                continue
            hn_l, hn_r = hn_sets[left], hn_sets[right]
            pos_l, pos_r = pos_sets[left], pos_sets[right]
            overlap[f"{left}_vs_{right}"] = {
                "hn_jaccard": (len(hn_l & hn_r) / len(hn_l | hn_r)) if (hn_l or hn_r) else 1.0,
                "hn_only_left": len(hn_l - hn_r),
                "hn_only_right": len(hn_r - hn_l),
                "robust_positive_jaccard": (
                    len(pos_l & pos_r) / len(pos_l | pos_r) if (pos_l or pos_r) else 1.0
                ),
            }
    return {"summaries": summaries, "set_overlap": overlap}


def write_comparison(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# Q0 vs Q3 vs Q5", ""]
    for name, stats in payload["summaries"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- robust_positive chunks: {stats['robust_positive_chunks']}")
        lines.append(f"- question_sensitive chunks: {stats['question_sensitive_chunks']}")
        lines.append(f"- robust_negative chunks: {stats['robust_negative_chunks']}")
        lines.append(f"- hard_negative chunks: {stats['hard_negative_chunks']}")
        lines.append(f"- no_robust_positive rate: {stats['no_robust_positive_rate']:.3f}")
        lines.append(f"- Pos+HN instances: {stats['pos_hn_instances']}")
        lines.append(f"- mean QAConsistency: {stats['mean_qa_consistency']:.3f}")
        lines.append("")
    lines.append("## HN / robust_positive overlap")
    lines.append("")
    lines.append(json.dumps(payload["set_overlap"], indent=2, ensure_ascii=False))
    lines.append("")
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
