from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from untie.keyword_tuning import (
    DocumentEvaluation,
    JSONCheckpointCache,
    KeywordEvidence,
    ObjectiveConfig,
    SearchConfig,
    aggregate_candidate_pool,
    deterministic_document_split,
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
        KeywordEvidence("  Useful-Term ", "train-1", 0.2, 0.3),
        KeywordEvidence("useful term", "train-2", 0.4, 0.5),
        KeywordEvidence("USEFUL term", "train-3", 100.0, 100.0),
        KeywordEvidence("leaked", "test-1", 99.0, 99.0),
        KeywordEvidence("the", "train-1", 1.0, 1.0),
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
