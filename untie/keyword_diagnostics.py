"""Диагностика pipeline настройки ключевых слов с явными причинами отсева."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .keyword_evidence import CachedDocumentEvidence
from .keyword_training import StrategyConfig
from .keyword_tuning import (
    KeywordEvidence,
    ObjectiveConfig,
    aggregate_candidate_pool,
    build_inverted_index,
    enrich_pool_from_train_references,
    estimate_term_activation,
    evaluate_objective,
    normalize_term as pool_normalize_term,
)

_TERM_RE = re.compile(r"[^\w\s-]+", re.UNICODE)

# Re-use pool filters from keyword_tuning without importing private sets directly.
from . import keyword_tuning as _kt

_DEFAULT_STOPWORDS = _kt._DEFAULT_STOPWORDS
_GENERIC_VERBS = _kt._GENERIC_VERBS


def load_evidence_directory(path: str | Path) -> dict[str, CachedDocumentEvidence]:
    """Load all cached document evidence JSON files."""
    root = Path(path)
    result: dict[str, CachedDocumentEvidence] = {}
    for file_path in sorted(root.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        document = CachedDocumentEvidence.from_dict(payload)
        result[document.doc_id] = document
    return result


def per_document_candidate_rows(
    documents: Mapping[str, CachedDocumentEvidence],
    doc_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """One row per (train document, extracted candidate word)."""
    rows: list[dict[str, Any]] = []
    for doc_id in sorted({str(item) for item in doc_ids}):
        document = documents.get(doc_id)
        if document is None:
            continue
        for candidate in document.candidates:
            rows.append(
                {
                    "doc_id": doc_id,
                    "word": candidate.word,
                    "lemma": candidate.lemma,
                    "stem": candidate.stem,
                    "attention_weight": candidate.attention_weight,
                    "score_difference": candidate.score_difference,
                    "matched_chunk_count": len(candidate.matched_chunk_indices),
                    "matched_chunk_indices": list(candidate.matched_chunk_indices),
                    "references": list(document.references),
                }
            )
    return rows


def _group_train_evidence(
    evidence: Iterable[KeywordEvidence],
    train_doc_ids: Iterable[str],
    *,
    stopwords: Iterable[str] = (),
    filter_generic: bool = True,
) -> tuple[set[str], dict[str, dict[str, Any]], set[str]]:
    train = {str(doc_id) for doc_id in train_doc_ids}
    stops = _DEFAULT_STOPWORDS | {pool_normalize_term(word) for word in stopwords}
    if filter_generic:
        stops = stops | _GENERIC_VERBS
    grouped: dict[str, dict[str, Any]] = {}
    rejected_terms: set[str] = set()
    for item in evidence:
        if item.doc_id not in train:
            continue
        term = pool_normalize_term(item.term)
        if not term:
            continue
        if term in stops or all(part in stops for part in term.split()):
            rejected_terms.add(term)
            continue
        bucket = grouped.setdefault(
            term,
            {"attention": [], "score_diff": [], "docs": set(), "chunk_docs": set()},
        )
        bucket["attention"].append(float(item.attention))
        bucket["score_diff"].append(float(item.score_diff))
        bucket["docs"].add(item.doc_id)
        if item.chunk_support_rate > 0:
            bucket["chunk_docs"].add(item.doc_id)
    return stops, grouped, rejected_terms


def audit_pool_aggregation(
    evidence: Iterable[KeywordEvidence],
    train_doc_ids: Iterable[str],
    *,
    min_document_support: int = 2,
    require_chunk_support: bool = True,
) -> list[dict[str, Any]]:
    """Explain why each train term entered or missed the candidate pool."""
    pool = aggregate_candidate_pool(
        evidence,
        train_doc_ids,
        min_document_support=min_document_support,
        require_chunk_support=require_chunk_support,
    )
    in_pool = {item.term for item in pool}
    _, grouped, stopword_terms = _group_train_evidence(evidence, train_doc_ids)
    rows: list[dict[str, Any]] = []
    for term in sorted(set(grouped) | stopword_terms):
        if term in stopword_terms and term not in grouped:
            rows.append(
                {
                    "term": term,
                    "stage": "aggregate_pool",
                    "decision": "rejected",
                    "reason": "stopword_or_generic",
                    "document_support": 0,
                    "chunk_support_rate": 0.0,
                }
            )
            continue
        bucket = grouped[term]
        docs = sorted(bucket["docs"])
        support = len(docs)
        chunk_rate = len(bucket["chunk_docs"]) / support if support else 0.0
        if term in in_pool:
            decision, reason = "kept", "in_pool"
        elif support < min_document_support:
            decision, reason = "rejected", "low_document_support"
        elif require_chunk_support and chunk_rate <= 0:
            decision, reason = "rejected", "no_chunk_support"
        else:
            decision, reason = "rejected", "filtered_by_pool_cap_or_other"
        rows.append(
            {
                "term": term,
                "stage": "aggregate_pool",
                "decision": decision,
                "reason": reason,
                "document_support": support,
                "chunk_support_rate": round(chunk_rate, 4),
                "median_attention": float(statistics.median(bucket["attention"]))
                if bucket["attention"]
                else 0.0,
                "median_score_diff": float(statistics.median(bucket["score_diff"]))
                if bucket["score_diff"]
                else 0.0,
                "supporting_docs": docs,
            }
        )
    return rows


def audit_enrichment(
    pool: Sequence[KeywordEvidence],
    documents: Mapping[str, CachedDocumentEvidence],
    train_doc_ids: Iterable[str],
    *,
    min_document_support: int = 8,
    max_terms: int = 30,
) -> list[dict[str, Any]]:
    """Explain reference-mined terms added or rejected during pool enrichment."""
    before = {item.term for item in pool}
    enriched = enrich_pool_from_train_references(
        pool,
        documents,
        train_doc_ids,
        min_document_support=min_document_support,
        max_terms=max_terms,
    )
    after = {item.term for item in enriched}
    added = [item for item in enriched if item.term not in before and item.is_enriched]
    rows: list[dict[str, Any]] = []
    for item in added:
        rows.append(
            {
                "term": item.term,
                "stage": "enrich_references",
                "decision": "kept",
                "reason": "enriched_from_train_references",
                "document_support": item.document_support,
                "chunk_support_rate": item.chunk_support_rate,
                "supporting_docs": list(item.supporting_docs),
            }
        )
    # Recompute rejected reference n-grams for transparency.
    train = {str(doc_id) for doc_id in train_doc_ids}
    counts: dict[str, set[str]] = defaultdict(set)
    for doc_id in train:
        document = documents.get(doc_id)
        if document is None:
            continue
        for reference in document.references:
            tokens = [
                pool_normalize_term(part)
                for part in _TERM_RE.sub(" ", str(reference).casefold()).split()
                if pool_normalize_term(part)
                and pool_normalize_term(part) not in _DEFAULT_STOPWORDS
                and pool_normalize_term(part) not in _GENERIC_VERBS
            ]
            for token in tokens:
                counts[token].add(doc_id)
            for left, right in zip(tokens, tokens[1:]):
                bigram = pool_normalize_term(f"{left} {right}")
                if bigram:
                    counts[bigram].add(doc_id)
    for term, docs in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0])):
        if term in after:
            continue
        if term in before:
            continue
        if len(docs) < min_document_support:
            reason = "low_enriched_document_support"
        else:
            reason = "enrichment_cap_or_existing_term"
        rows.append(
            {
                "term": term,
                "stage": "enrich_references",
                "decision": "rejected",
                "reason": reason,
                "document_support": len(docs),
                "chunk_support_rate": 0.0,
                "supporting_docs": sorted(docs),
            }
        )
    return rows


def audit_prescreen(
    candidate_terms: Sequence[str],
    evaluator: Callable[[tuple[str, ...], tuple[str, ...]], Mapping[str, Any]],
    doc_ids: Sequence[str],
    objective_config: ObjectiveConfig,
    *,
    top_k: int,
    harm_cap: float,
    inverted_index: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Prescreen audit with explicit rejection reasons (mirrors screen_candidate_terms)."""
    empty_eval = evaluate_objective(evaluator((), doc_ids), (), config=objective_config)
    scored: list[tuple[float, float, float, float, str]] = []
    rows: list[dict[str, Any]] = []

    for term in candidate_terms:
        predicted = None
        if inverted_index is not None:
            predicted = estimate_term_activation(term, inverted_index, doc_ids)
            if predicted < objective_config.min_activation_rate:
                rows.append(
                    {
                        "term": term,
                        "stage": "prescreen",
                        "decision": "rejected",
                        "reason": "low_predicted_activation",
                        "predicted_activation": round(predicted, 4),
                        "min_activation_rate": objective_config.min_activation_rate,
                    }
                )
                continue
        result = evaluate_objective(
            evaluator((term,), doc_ids), (term,), config=objective_config
        )
        if result.harm_rate > harm_cap:
            rows.append(
                {
                    "term": term,
                    "stage": "prescreen",
                    "decision": "rejected",
                    "reason": "harm_above_cap",
                    "harm_rate": round(result.harm_rate, 4),
                    "harm_cap": harm_cap,
                    "mean_gain_active": round(result.mean_gain_active, 4),
                    "activation_rate": round(result.activation_rate, 4),
                    "objective": round(result.objective, 4),
                    "predicted_activation": predicted,
                }
            )
            continue
        if result.mean_gain_active <= 0 and result.objective <= empty_eval.objective:
            rows.append(
                {
                    "term": term,
                    "stage": "prescreen",
                    "decision": "rejected",
                    "reason": "no_gain_vs_empty",
                    "mean_gain_active": round(result.mean_gain_active, 4),
                    "activation_rate": round(result.activation_rate, 4),
                    "objective": round(result.objective, 4),
                    "empty_objective": round(empty_eval.objective, 4),
                    "predicted_activation": predicted,
                }
            )
            continue
        scored.append(
            (
                result.mean_gain_active,
                result.activation_rate,
                result.objective,
                result.harm_rate,
                term,
            )
        )
        rows.append(
            {
                "term": term,
                "stage": "prescreen",
                "decision": "candidate",
                "reason": "passed_single_keyword_eval",
                "mean_gain_active": round(result.mean_gain_active, 4),
                "activation_rate": round(result.activation_rate, 4),
                "objective": round(result.objective, 4),
                "harm_rate": round(result.harm_rate, 4),
                "predicted_activation": predicted,
            }
        )

    if not scored:
        rows.clear()
        for term in candidate_terms:
            result = evaluate_objective(
                evaluator((term,), doc_ids), (term,), config=objective_config
            )
            scored.append(
                (
                    result.mean_gain_active,
                    result.activation_rate,
                    result.objective,
                    result.harm_rate,
                    term,
                )
            )
            rows.append(
                {
                    "term": term,
                    "stage": "prescreen",
                    "decision": "candidate",
                    "reason": "fallback_rescue_no_passed_terms",
                    "mean_gain_active": round(result.mean_gain_active, 4),
                    "activation_rate": round(result.activation_rate, 4),
                    "objective": round(result.objective, 4),
                    "harm_rate": round(result.harm_rate, 4),
                }
            )

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[4]))
    kept = {term for _, _, _, _, term in scored[: max(0, top_k)]}
    for row in rows:
        if row.get("decision") != "candidate":
            continue
        if row["term"] not in kept:
            row["decision"] = "rejected"
            row["reason"] = "below_screen_top_k"
            row["screen_top_k"] = top_k
            row["rank"] = next(
                index + 1
                for index, (_, _, _, _, term) in enumerate(scored)
                if term == row["term"]
            )
        else:
            row["decision"] = "kept"
            row["rank"] = next(
                index + 1
                for index, (_, _, _, _, term) in enumerate(scored)
                if term == row["term"]
            )
    return rows


def keyword_train_coverage_rows(
    pool: Sequence[KeywordEvidence],
    selected_keywords: Sequence[str],
    train_doc_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Matrix-style rows: selected keyword × train document presence."""
    pool_by_term = {item.term: item for item in pool}
    train = sorted({str(doc_id) for doc_id in train_doc_ids})
    rows: list[dict[str, Any]] = []
    for keyword in selected_keywords:
        term = pool_normalize_term(keyword)
        item = pool_by_term.get(term)
        supporting = set(item.supporting_docs if item else ())
        for doc_id in train:
            rows.append(
                {
                    "keyword": term,
                    "doc_id": doc_id,
                    "present_in_train_evidence": doc_id in supporting,
                    "document_support": item.document_support if item else 0,
                    "chunk_support_rate": item.chunk_support_rate if item else 0.0,
                }
            )
    return rows


def build_form_map_from_train_documents(
    documents: Mapping[str, CachedDocumentEvidence],
    train_doc_ids: Iterable[str],
) -> dict[str, tuple[str, str]]:
    """Pick the most frequent (lemma, stem) per surface word on train."""
    votes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc_id in train_doc_ids:
        document = documents.get(str(doc_id))
        if document is None:
            continue
        for candidate in document.candidates:
            votes[candidate.word].append((candidate.lemma, candidate.stem))
    return {
        word: max(set(values), key=values.count) for word, values in votes.items()
    }


def build_weighted_keyword_map(
    pool: Sequence[KeywordEvidence],
    documents: Mapping[str, CachedDocumentEvidence],
    train_doc_ids: Iterable[str],
) -> dict[str, Any]:
    from .keyword_training import _keyword_map

    return _keyword_map(pool, build_form_map_from_train_documents(documents, train_doc_ids))


def parse_tuning_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strategy_from_trace_payload(payload: Mapping[str, Any]) -> StrategyConfig:
    selected = payload["selected_strategy"]
    return StrategyConfig(
        selected["score_chunk_strategy"],
        selected["choose_cluster_strategy"],
        selected["choose_answer_strategy"],
    )


def sffs_trace_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten SFFS steps for the selected strategy across stability runs."""
    selected = strategy_from_trace_payload(payload)
    rows: list[dict[str, Any]] = []
    for outcome in payload.get("strategy_outcomes", []):
        strategy = outcome.get("strategy", {})
        name = "+".join(
            [
                strategy.get("score_chunk_strategy", ""),
                strategy.get("choose_cluster_strategy", ""),
                strategy.get("choose_answer_strategy", ""),
            ]
        )
        if name != selected.name:
            continue
        for run_index, trace in enumerate(outcome.get("traces", []), start=1):
            for step in trace:
                rows.append(
                    {
                        "run": run_index,
                        "iteration": step.get("iteration"),
                        "action": step.get("action"),
                        "keyword": step.get("keyword"),
                        "subset": tuple(step.get("subset", ())),
                        "subset_size": len(step.get("subset", ())),
                        "objective": step.get("objective"),
                        "panel_size": step.get("panel_size"),
                        "evaluations_used": step.get("evaluations_used"),
                    }
                )
    return rows


def stability_selection_rows(
    payload: Mapping[str, Any],
    *,
    stability_threshold: float,
) -> list[dict[str, Any]]:
    """Summarize stability-run finals and frequency-based keep/drop."""
    selected = strategy_from_trace_payload(payload)
    rows: list[dict[str, Any]] = []
    for outcome in payload.get("strategy_outcomes", []):
        strategy = outcome.get("strategy", {})
        name = "+".join(
            [
                strategy.get("score_chunk_strategy", ""),
                strategy.get("choose_cluster_strategy", ""),
                strategy.get("choose_answer_strategy", ""),
            ]
        )
        if name != selected.name:
            continue
        runs = outcome.get("selections", [])
        run_count = max(1, len(runs))
        frequency: dict[str, int] = defaultdict(int)
        for run_index, subset in enumerate(runs, start=1):
            for term in subset:
                frequency[pool_normalize_term(term)] += 1
            rows.append(
                {
                    "stage": "stability_run",
                    "run": run_index,
                    "subset": tuple(subset),
                    "subset_size": len(subset),
                    "decision": "run_final",
                    "reason": "sffs_completed",
                }
            )
        for term, count in sorted(frequency.items(), key=lambda item: (-item[1], item[0])):
            rate = count / run_count
            rows.append(
                {
                    "stage": "stability_threshold",
                    "term": term,
                    "frequency": count,
                    "run_count": run_count,
                    "selection_rate": round(rate, 4),
                    "stability_threshold": stability_threshold,
                    "decision": "kept" if rate >= stability_threshold else "rejected",
                    "reason": "above_stability_threshold"
                    if rate >= stability_threshold
                    else "below_stability_threshold",
                }
            )
    return rows


def funnel_summary(rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """High-level counts per pipeline stage."""
    summary: list[dict[str, Any]] = []
    for stage, rows in rows_by_stage.items():
        kept = sum(1 for row in rows if row.get("decision") in {"kept", "candidate", "in_pool", "run_final"})
        rejected = sum(1 for row in rows if row.get("decision") == "rejected")
        summary.append(
            {
                "stage": stage,
                "rows": len(rows),
                "kept_or_active": kept,
                "rejected": rejected,
            }
        )
    return summary
