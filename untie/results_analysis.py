"""Утилиты анализа JSON-результатов refactored-пайплайна UnTIE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STRATEGY_COLUMNS = (
    "score_chunk_strategy",
    "choose_cluster_strategy",
    "choose_answer_strategy",
)
INVALID_MARKER = "--No valid answers--"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_json(path)


def parse_metrics(metrics: Any) -> tuple[float, float]:
    """Поддерживает dict (новый формат) и JSON-строку (legacy)."""
    if isinstance(metrics, dict):
        return float(metrics.get("cosine_sim", 0)), float(metrics.get("lev_dist", 0))
    if metrics is None or (isinstance(metrics, float) and np.isnan(metrics)):
        return 0.0, 0.0
    try:
        text = str(metrics).replace("'", '"')
        parsed = json.loads(text)
        return float(parsed.get("cosine_sim", 0)), float(parsed.get("lev_dist", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, 0.0


def filter_valid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Отделяет строки с валидными ответами и стратегиями."""
    invalid_mask = (
        (df["final_answer"] == INVALID_MARKER)
        | df["final_answer"].isna()
        | (df["final_answer"].astype(str).str.strip() == "")
        | df["score_chunk_strategy"].eq(INVALID_MARKER)
        | df["choose_cluster_strategy"].eq(INVALID_MARKER)
        | df["choose_answer_strategy"].eq(INVALID_MARKER)
    )
    invalid_df = df[invalid_mask].copy()
    valid_df = df[~invalid_mask].copy()
    return valid_df, invalid_df


def split_evaluable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Отделяет строки, где стратегии выбирали между разными ответами.

    Старые EN-результаты не содержат диагностического поля, поэтому считаются
    evaluable целиком и сохраняют прежнее поведение анализа.
    """
    if "strategy_applicable" not in df.columns:
        return df.copy(), df.iloc[0:0].copy()
    mask = df["strategy_applicable"].fillna(False).astype(bool)
    return df[mask].copy(), df[~mask].copy()


def strategy_applicability_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Возвращает долю документов с реальным выбором между стратегиями."""
    total_documents = int(df["doc_id"].nunique()) if not df.empty else 0
    if total_documents == 0:
        return {
            "total_documents": 0,
            "evaluable_documents": 0,
            "evaluable_documents_pct": 0.0,
        }
    if "strategy_applicable" not in df.columns:
        evaluable_documents = total_documents
    else:
        per_document = df.groupby("doc_id")["strategy_applicable"].any()
        evaluable_documents = int(per_document.sum())
    return {
        "total_documents": total_documents,
        "evaluable_documents": evaluable_documents,
        "evaluable_documents_pct": evaluable_documents / total_documents * 100,
    }


def enrich_with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет числовые метрики и флаги улучшения/ухудшения."""
    out = df.copy()
    base = out["base_metrics"].apply(parse_metrics)
    corrected = out["corrected_metrics"].apply(parse_metrics)

    out["cosine_sim_base"] = base.apply(lambda x: x[0])
    out["lev_dist_base"] = base.apply(lambda x: x[1])
    out["cosine_sim_corrected"] = corrected.apply(lambda x: x[0])
    out["lev_dist_corrected"] = corrected.apply(lambda x: x[1])
    out["cosine_sim_diff"] = out["cosine_sim_corrected"] - out["cosine_sim_base"]
    out["lev_dist_diff"] = out["lev_dist_corrected"] - out["lev_dist_base"]

    out["cosine_improved"] = out["cosine_sim_diff"] > 0
    out["cosine_worsened"] = out["cosine_sim_diff"] < 0
    out["cosine_unchanged"] = out["cosine_sim_diff"] == 0
    out["lev_improved"] = out["lev_dist_diff"] < 0
    out["lev_worsened"] = out["lev_dist_diff"] > 0
    out["lev_unchanged"] = out["lev_dist_diff"] == 0
    out["overall_improved"] = out["cosine_improved"] & out["lev_improved"]
    out["overall_worsened"] = out["cosine_worsened"] & out["lev_worsened"]
    out["overall_mixed"] = ~(out["overall_improved"] | out["overall_worsened"] | (
        out["cosine_unchanged"] & out["lev_unchanged"]
    ))
    out["overall_unchanged"] = out["cosine_unchanged"] & out["lev_unchanged"]
    return out


def build_strategy_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for keys, group_df in df.groupby(list(STRATEGY_COLUMNS), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = "_".join(keys)
        groups[key] = group_df
    return groups


def summarize_group(group_df: pd.DataFrame) -> dict[str, float | int]:
    n = len(group_df)
    if n == 0:
        return {"total_samples": 0}

    def pct(series: pd.Series) -> float:
        return float(series.sum() / n * 100)

    return {
        "total_samples": n,
        "cosine_improved": int(group_df["cosine_improved"].sum()),
        "cosine_improved_pct": pct(group_df["cosine_improved"]),
        "cosine_worsened": int(group_df["cosine_worsened"].sum()),
        "cosine_worsened_pct": pct(group_df["cosine_worsened"]),
        "cosine_unchanged": int(group_df["cosine_unchanged"].sum()),
        "cosine_unchanged_pct": pct(group_df["cosine_unchanged"]),
        "lev_improved": int(group_df["lev_improved"].sum()),
        "lev_improved_pct": pct(group_df["lev_improved"]),
        "lev_worsened": int(group_df["lev_worsened"].sum()),
        "lev_worsened_pct": pct(group_df["lev_worsened"]),
        "lev_unchanged": int(group_df["lev_unchanged"].sum()),
        "lev_unchanged_pct": pct(group_df["lev_unchanged"]),
        "overall_improved": int(group_df["overall_improved"].sum()),
        "overall_improved_pct": pct(group_df["overall_improved"]),
        "overall_worsened": int(group_df["overall_worsened"].sum()),
        "overall_worsened_pct": pct(group_df["overall_worsened"]),
        "overall_mixed": int(group_df["overall_mixed"].sum()),
        "overall_mixed_pct": pct(group_df["overall_mixed"]),
        "overall_unchanged": int(group_df["overall_unchanged"].sum()),
        "overall_unchanged_pct": pct(group_df["overall_unchanged"]),
        "cosine_mean_diff": float(group_df["cosine_sim_diff"].mean()),
        "cosine_std_diff": float(group_df["cosine_sim_diff"].std()),
        "lev_mean_diff": float(group_df["lev_dist_diff"].mean()),
        "lev_std_diff": float(group_df["lev_dist_diff"].std()),
    }


def build_summary_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    rows = {name: summarize_group(group_df) for name, group_df in groups.items()}
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "strategy_combo"
    return summary.sort_values("overall_improved_pct", ascending=False)


def add_composite_score(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["composite_score"] = (
        out["overall_improved_pct"] * 0.4
        + out["cosine_improved_pct"] * 0.3
        + out["lev_improved_pct"] * 0.3
        - out["overall_worsened_pct"] * 0.2
    )
    return out


def aggregate_by_strategy(df: pd.DataFrame, strategy_col: str) -> pd.DataFrame:
    stats = df.groupby(strategy_col).agg(
        overall_improved=("overall_improved", "mean"),
        cosine_improved=("cosine_improved", "mean"),
        lev_improved=("lev_improved", "mean"),
        overall_worsened=("overall_worsened", "mean"),
        cosine_mean_diff=("cosine_sim_diff", "mean"),
        cosine_std_diff=("cosine_sim_diff", "std"),
        lev_mean_diff=("lev_dist_diff", "mean"),
        lev_std_diff=("lev_dist_diff", "std"),
        count=("doc_id", "count"),
    ).round(4)
    for col in ("overall_improved", "cosine_improved", "lev_improved", "overall_worsened"):
        stats[f"{col}_pct"] = stats[col] * 100
    return stats
