from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from untie.keyword_tuning import (
    DocumentEvaluation,
    JSONCheckpointCache,
    KeywordEvidence,
    ObjectiveConfig,
    SearchConfig,
    SearchResult,
    _pad_subset_to_min_keywords,
    aggregate_candidate_pool,
    deterministic_document_split,
    enrich_pool_from_train_references,
    evaluate_objective,
    finalize_keyword_subset,
    prune_keyword_subset,
    sequential_forward_floating_selection,
    stability_selection,
    stable_subset_hash,
)


def test_split_has_no_leakage_and_is_order_independent() -> None:
    docs = [f"doc-{index}" for index in range(40)]
    first = deterministic_document_split(docs, seed=17)
    second = deterministic_document_split(reversed(docs), seed=17)
    assert first == second
    assert set(first["train"]).isdisjoint(first["dev"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["dev"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(docs)


def test_candidate_pool_is_train_only_normalized_and_robust() -> None:
    evidence = [
        KeywordEvidence("  Useful-Term ", "train-1", 0.2, 0.3, chunk_support_rate=1.0),
        KeywordEvidence("useful term", "train-2", 0.4, 0.5, chunk_support_rate=1.0),
        KeywordEvidence("USEFUL term", "train-3", 100.0, 100.0, chunk_support_rate=1.0),
        KeywordEvidence("leaked", "test-1", 99.0, 99.0, chunk_support_rate=1.0),
        KeywordEvidence("the", "train-1", 1.0, 1.0, chunk_support_rate=1.0),
    ]
    pool = aggregate_candidate_pool(
        evidence,
        {"train-1", "train-2", "train-3"},
        min_document_support=2,
    )
    assert [item.term for item in pool] == ["useful term"]
    assert pool[0].supporting_docs == ("train-1", "train-2", "train-3")
    assert pool[0].attention == pytest.approx(0.4)
    assert pool[0].score_diff == pytest.approx(0.5)
    with pytest.raises(FrozenInstanceError):
        pool[0].term = "changed"  # type: ignore[misc]


def test_subset_hash_is_stable_and_order_insensitive() -> None:
    assert stable_subset_hash([" Beta ", "alpha", "alpha"]) == stable_subset_hash(
        ["ALPHA", "beta"]
    )


def _landscape_evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]):
    # This landscape exercises floating removal: a is initially strongest, but
    # once b and c interact, b+c is better than a+b+c.
    values = {
        (): 0.0,
        ("a",): 10.0,
        ("b",): 9.0,
        ("c",): 0.0,
        ("a", "b"): 11.0,
        ("a", "c"): 10.0,
        ("b", "c"): 21.0,
        ("a", "b", "c"): 20.0,
    }
    quality = values[tuple(sorted(subset))]
    return {
        doc_id: DocumentEvaluation(doc_id, quality, fallback=False)
        for doc_id in doc_ids
    }


def _objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        downside_penalty=0,
        harm_penalty=0,
        fallback_penalty=0,
        size_penalty=0,
        bootstrap_samples=50,
        bootstrap_seed=3,
    )


def test_search_is_deterministic_and_floating_removes_harmful_term() -> None:
    config = SearchConfig(
        max_keywords=3,
        evaluation_budget=100,
        patience=2,
        seed=5,
    )
    first = sequential_forward_floating_selection(
        ["c", "b", "a"],
        ["d2", "d1"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=config,
    )
    second = sequential_forward_floating_selection(
        ["a", "b", "c"],
        ["d1", "d2"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=config,
    )
    assert first == second
    assert first.keywords == ("b", "c")
    assert any(step.action == "remove" and step.keyword == "a" for step in first.trace)
    assert first.evaluation.mean_gain == pytest.approx(21.0)


def test_beneficial_interacting_terms_are_selected() -> None:
    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]):
        score = {
            (): 0.0,
            ("x",): 0.1,
            ("y",): 0.05,
            ("x", "y"): 2.0,
        }[tuple(sorted(subset))]
        return {doc_id: score for doc_id in doc_ids}

    result = sequential_forward_floating_selection(
        ["x", "y"],
        ["d"],
        evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(max_keywords=2, evaluation_budget=20, patience=1),
    )
    assert result.keywords == ("x", "y")


def test_max_budget_terminates_explicitly() -> None:
    result = sequential_forward_floating_selection(
        ["a", "b", "c"],
        ["d1", "d2"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(max_keywords=3, evaluation_budget=2, patience=2),
    )
    assert result.stop_reason == "evaluation_budget"
    assert result.evaluations_used == 2


def test_multi_fidelity_promotes_shortlist_before_acceptance() -> None:
    panel_sizes_seen: list[int] = []

    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]):
        panel_sizes_seen.append(len(doc_ids))
        if not subset:
            score = 0.0
        elif subset == ("a",):
            score = 10.0 if len(doc_ids) <= 2 else -1.0
        else:
            score = 5.0 if len(doc_ids) <= 2 else 3.0
        return {doc_id: score for doc_id in doc_ids}

    result = sequential_forward_floating_selection(
        ["a", "b"],
        [f"d-{index}" for index in range(8)],
        evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(
            max_keywords=1,
            evaluation_budget=20,
            patience=1,
            beam_width=2,
            panel_sizes=(2, 8),
        ),
    )
    assert result.keywords == ("b",)
    assert 2 in panel_sizes_seen and 8 in panel_sizes_seen


def test_checkpoint_resume_matches_uninterrupted_search(tmp_path) -> None:
    path = tmp_path / "keyword-search.json"
    partial = sequential_forward_floating_selection(
        ["a", "b", "c"],
        ["d1", "d2"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(max_keywords=3, evaluation_budget=4, patience=2),
        checkpoint=path,
    )
    assert partial.stop_reason == "evaluation_budget"
    resumed = sequential_forward_floating_selection(
        ["a", "b", "c"],
        ["d1", "d2"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(max_keywords=3, evaluation_budget=100, patience=2),
        checkpoint=path,
    )
    uninterrupted = sequential_forward_floating_selection(
        ["a", "b", "c"],
        ["d1", "d2"],
        _landscape_evaluator,
        objective_config=_objective(),
        search_config=SearchConfig(max_keywords=3, evaluation_budget=100, patience=2),
    )
    assert resumed.keywords == uninterrupted.keywords
    assert resumed.evaluation == uninterrupted.evaluation
    assert resumed.trace == uninterrupted.trace
    assert JSONCheckpointCache(path, "wrong-fingerprint").entries == {}


def test_stability_selection_returns_frequency_set_and_jaccard() -> None:
    selected, score = stability_selection(
        [{"a", "b"}, {"a", "b", "c"}, {"a", "c"}],
        threshold=2 / 3,
    )
    assert selected == ("a", "b", "c")
    assert score == pytest.approx((2 / 3 + 1 / 3 + 2 / 3) / 3)


def test_fallback_penalty_skips_harmless_fallback() -> None:
    config = ObjectiveConfig(
        fallback_penalty=0.5,
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        min_activation_rate=0,
        activation_weight=0,
        win_rate_weight=0,
    )
    harmless = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.5, baseline_quality=0.5, fallback=True),
            DocumentEvaluation("d2", 0.4, baseline_quality=0.4, fallback=True),
        ],
        ("term",),
        config=config,
    )
    empty = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.5, baseline_quality=0.5, fallback=True),
            DocumentEvaluation("d2", 0.4, baseline_quality=0.4, fallback=True),
        ],
        (),
        config=config,
    )
    assert harmless.objective == pytest.approx(0.0)
    assert empty.objective == pytest.approx(0.0)

    harmful = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.3, baseline_quality=0.5, fallback=True),
        ],
        ("term",),
        config=config,
    )
    assert harmful.objective == pytest.approx(-0.7)


def test_enrich_pool_from_train_references_adds_task_terms() -> None:
    class Doc:
        def __init__(self, references: tuple[str, ...]) -> None:
            self.references = references

    documents = {
        "train-1": Doc(("Semantic Segmentation",)),
        "train-2": Doc(("Semantic Segmentation",)),
    }
    pool = enrich_pool_from_train_references(
        (),
        documents,
        ("train-1", "train-2"),
        min_document_support=2,
    )
    terms = {item.term for item in pool}
    assert "semantic" in terms
    assert "segmentation" in terms
    assert "semantic segmentation" in terms


def test_finalize_prefers_best_run_over_empty_stable() -> None:
    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]):
        if subset == ("good",):
            quality = 1.0
        elif subset == ("bad",):
            quality = -1.0
        else:
            quality = 0.0
        return {
            doc_id: DocumentEvaluation(doc_id, quality, baseline_quality=0.0, fallback=False)
            for doc_id in doc_ids
        }

    objective = ObjectiveConfig(
        fallback_penalty=0,
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        min_activation_rate=0,
        activation_weight=0,
        win_rate_weight=0,
    )
    selections = [("good",), ("bad",), ("other",)]
    search_results = [
        SearchResult(
            ("good",),
            evaluate_objective(
                evaluator(("good",), ("d1",)), ("good",), config=objective
            ),
            (),
            "converged",
            1,
        ),
        SearchResult(
            ("bad",),
            evaluate_objective(
                evaluator(("bad",), ("d1",)), ("bad",), config=objective
            ),
            (),
            "converged",
            1,
        ),
        SearchResult(
            ("other",),
            evaluate_objective(
                evaluator(("other",), ("d1",)), ("other",), config=objective
            ),
            (),
            "converged",
            1,
        ),
    ]
    keywords, evaluation, _ = finalize_keyword_subset(
        selections,
        search_results,
        ("good", "bad", "other"),
        evaluator,
        ("d1",),
        objective,
        policy="relaxed",
        stability_threshold=0.7,
        require_non_empty=True,
    )
    assert keywords == ("good",)
    assert evaluation.mean_gain == pytest.approx(1.0)


def test_idle_fallback_penalized_for_nonempty_subset() -> None:
    config = ObjectiveConfig(
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0.1,
        min_activation_rate=0,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )
    idle = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.5, baseline_quality=0.5, fallback=True),
            DocumentEvaluation("d2", 0.4, baseline_quality=0.4, fallback=True),
        ],
        ("term",),
        config=config,
    )
    empty = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.5, baseline_quality=0.5, fallback=True),
            DocumentEvaluation("d2", 0.4, baseline_quality=0.4, fallback=True),
        ],
        (),
        config=config,
    )
    assert idle.objective < empty.objective


def test_conditional_gain_prefers_active_subset() -> None:
    config = ObjectiveConfig(
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        min_activation_rate=0,
        activation_weight=0.15,
        win_rate_weight=0,
        confidence_weight=0,
        use_conditional_gain=True,
    )
    rare = evaluate_objective(
        [
            DocumentEvaluation("d1", 0.3, baseline_quality=0.0, fallback=False),
            *[
                DocumentEvaluation(f"d{index}", 0.0, baseline_quality=0.0, fallback=True)
                for index in range(2, 11)
            ],
        ],
        ("rare",),
        config=config,
    )
    common = evaluate_objective(
        [
            *[
                DocumentEvaluation(
                    f"d{index}",
                    0.25,
                    baseline_quality=0.0,
                    fallback=False,
                )
                for index in range(1, 6)
            ],
            *[
                DocumentEvaluation(f"d{index}", 0.0, baseline_quality=0.0, fallback=True)
                for index in range(6, 11)
            ],
        ],
        ("common",),
        config=config,
    )
    assert common.activation_rate > rare.activation_rate
    assert common.objective > rare.objective


def test_min_activation_gate_rejects_idle_subset() -> None:
    config = ObjectiveConfig(
        min_activation_rate=0.5,
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )
    result = evaluate_objective(
        [
            DocumentEvaluation("d1", 1.0, baseline_quality=0.0, fallback=False),
            DocumentEvaluation("d2", 0.0, baseline_quality=0.0, fallback=True),
            DocumentEvaluation("d3", 0.0, baseline_quality=0.0, fallback=True),
        ],
        ("term",),
        config=config,
    )
    assert result.objective == float("-inf")


def test_prune_respects_min_keywords() -> None:
    config = ObjectiveConfig(
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        min_activation_rate=0,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )

    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]) -> dict[str, DocumentEvaluation]:
        per_doc = 0.5 - 0.05 * len(subset)
        return {
            doc_id: DocumentEvaluation(doc_id, per_doc, baseline_quality=0.0, fallback=False)
            for doc_id in doc_ids
        }

    pruned, _ = prune_keyword_subset(
        ("a", "b", "c", "d"),
        evaluator,
        ("d1", "d2"),
        config,
        min_keywords=3,
    )
    assert len(pruned) == 3


def test_report_objective_skips_activation_gate() -> None:
    config = ObjectiveConfig(
        min_activation_rate=0.5,
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )
    evaluations = [
        DocumentEvaluation("d1", 1.0, baseline_quality=0.0, fallback=False),
        DocumentEvaluation("d2", 0.0, baseline_quality=0.0, fallback=True),
        DocumentEvaluation("d3", 0.0, baseline_quality=0.0, fallback=True),
    ]
    gated = evaluate_objective(evaluations, ("term",), config=config)
    assert gated.objective == float("-inf")

    report_config = replace(config, apply_activation_gate=False)
    reported = evaluate_objective(evaluations, ("term",), config=report_config)
    assert math.isfinite(reported.objective)


def test_pad_subset_to_min_keywords() -> None:
    config = ObjectiveConfig(
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        min_activation_rate=0,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )

    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]) -> dict[str, DocumentEvaluation]:
        per_doc = 0.4 + 0.05 * len(subset)
        return {
            doc_id: DocumentEvaluation(doc_id, per_doc, baseline_quality=0.0, fallback=False)
            for doc_id in doc_ids
        }

    padded, evaluation = _pad_subset_to_min_keywords(
        ("a",),
        ("a", "b", "c", "d"),
        evaluator,
        ("d1", "d2"),
        config,
        min_keywords=3,
        harm_cap=0.5,
    )
    assert len(padded) == 3
    assert evaluation.subset == padded


def test_finalize_falls_back_to_best_any_and_pads() -> None:
    config = ObjectiveConfig(
        harm_penalty=0,
        downside_penalty=0,
        size_penalty=0,
        inactive_fallback_penalty=0,
        min_activation_rate=0.5,
        activation_weight=0,
        win_rate_weight=0,
        confidence_weight=0,
    )

    def evaluator(subset: tuple[str, ...], doc_ids: tuple[str, ...]) -> dict[str, DocumentEvaluation]:
        score = 0.2 * len(subset)
        return {
            doc_id: DocumentEvaluation(doc_id, score, baseline_quality=0.0, fallback=False)
            for doc_id in doc_ids
        }

    selections = [("a",), ("b",)]
    search_results = [
        SearchResult(
            ("a",),
            evaluate_objective(evaluator(("a",), ("d1",)), ("a",), config=config),
            (),
            "converged",
            1,
        ),
        SearchResult(
            ("b",),
            evaluate_objective(evaluator(("b",), ("d1",)), ("b",), config=config),
            (),
            "converged",
            1,
        ),
    ]
    keywords, evaluation, _ = finalize_keyword_subset(
        selections,
        search_results,
        ("a", "b", "c", "d"),
        evaluator,
        ("d1", "d2", "d3"),
        config,
        policy="union",
        stability_threshold=0.0,
        require_non_empty=True,
        min_keywords=3,
        harm_cap=0.5,
    )
    assert len(keywords) >= 2
    assert keywords
    assert evaluation.subset == keywords
