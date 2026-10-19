from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection_v8.features import pairlex
from experiments.qa_chunk_selection_v9.contract import (
    AGREEMENT_GROUP,
    CONFIGS,
    CONFIG_COLUMNS,
    H2_COLUMNS,
    H3_COLUMNS,
    PAIRLEX_GROUP,
    PAIRLEX_INCREMENTAL_GROUP,
    SELECT_FOLDS,
    frozen_contract,
)
from experiments.qa_chunk_selection_v9.io_guard import assert_v9_write
from experiments.qa_chunk_selection_v9.selection import make_select_split_manifest, select_configuration


def _prediction(config, question, seed, fold, f1, selected, role="SELECT_OOF"):
    return {
        "runtime": {
            "question_id": question,
            "seed": seed,
            "select_fold": fold,
            "split_role": role,
            "canonical_document_id": f"doc-{question}",
            "selected_chunk_id": selected,
        },
        "evaluation": {
            "token_f1": f1,
            "baseline_token_f1": 50.0,
            "hit@1": 1,
            "baseline_hit@1": 1,
            "hn@1": 0,
            "baseline_hn@1": 0,
        },
    }


def _selector_rows():
    result = {config: [] for config in CONFIGS}
    for q in range(12):
        for seed in (42, 43, 44):
            result["C0"].append(_prediction("C0", f"q{q}", seed, q % 4, 50.0, 1))
            result["CA"].append(_prediction("CA", f"q{q}", seed, q % 4, 51.0, 2))
            result["CP"].append(_prediction("CP", f"q{q}", seed, q % 4, 50.0, 1))
            result["CAP"].append(_prediction("CAP", f"q{q}", seed, q % 4, 51.0, 2))
    return result


def test_exact_h2_and_agreement_schema():
    assert CONFIG_COLUMNS["C0"] == H2_COLUMNS
    assert CONFIG_COLUMNS["CA"] == H3_COLUMNS
    assert AGREEMENT_GROUP == tuple(x for x in H3_COLUMNS if x not in H2_COLUMNS)


def test_pairlex_is_already_in_exact_h2_and_union_adds_no_duplicate():
    assert set(PAIRLEX_GROUP) <= set(H2_COLUMNS)
    assert PAIRLEX_INCREMENTAL_GROUP == ()
    assert CONFIG_COLUMNS["CP"] == CONFIG_COLUMNS["C0"]
    assert CONFIG_COLUMNS["CAP"] == CONFIG_COLUMNS["CA"]
    assert len(CONFIG_COLUMNS["CAP"]) == len(set(CONFIG_COLUMNS["CAP"]))


def test_pairlex_exact_empty_answer_semantics():
    assert pairlex("", "") == 1.0
    assert pairlex("answer", "") == 0.0
    assert pairlex("same answer", "same answer") == 1.0


def test_selector_uses_seed_averaged_select_oof_and_complexity_tie():
    selected = select_configuration(_selector_rows(), (42, 43, 44))
    assert selected["selected_config"] == "CA"
    assert selected["candidates"]["CA"]["eligible"]
    assert not selected["candidates"]["CP"]["eligible"]
    assert selected["candidates"]["CA"]["stable_changed_questions"] == 12
    assert selected["seeds_are_not_independent_rows"]


def test_selector_rejects_outer_eval_rows():
    rows = _selector_rows()
    rows["C0"][0]["runtime"]["split_role"] = "OUTER_EVAL"
    with pytest.raises(AssertionError):
        select_configuration(rows, (42, 43, 44))


def test_select_split_is_four_fold_document_grouped_and_stable():
    rows = [
        {"canonical_document_id": f"d{i}", "question_id": f"q{i}"}
        for i in range(64)
    ]
    first = make_select_split_manifest(rows)
    second = make_select_split_manifest(list(reversed(rows)))
    assert first["select_folds"] == SELECT_FOLDS == 4
    assert first["sha256"] == second["sha256"]
    assert set(first["fold_document_counts"]) == {"0", "1", "2", "3"}


def test_contract_freezes_nested_selection_and_no_new_models():
    value = frozen_contract()
    assert value["select_folds"] == 4
    assert value["selector"]["eps_select"] == 0.2
    assert value["same_neural_forwards_s1_s2"]
    assert value["pairlex_already_in_exact_h2"]


def test_v9_write_guard():
    with pytest.raises(RuntimeError):
        assert_v9_write(Path("artifacts/qa_chunk_selection_v8_dual/forbidden.json"))
    allowed = assert_v9_write(Path("artifacts/qa_chunk_selection_v9_h2_acs/test/out.json"))
    assert "qa_chunk_selection_v9_h2_acs" in str(allowed)



def test_family_bootstrap_supports_smoke_without_scier():
    from experiments.qa_chunk_selection_v9.analyze import _family_bootstrap

    def row(system, value):
        return {
            "runtime": {
                "question_id": "q1", "seed": 42,
                "canonical_document_id": "doc1", "system": system,
            },
            "evaluation": {"token_f1": value},
        }

    data = {"scirex": {"S0": [row("S0", 10.0)], "S1": [row("S1", 11.0)]}}
    result = _family_bootstrap(data, "S1", "S0", 20)
    assert result["family_balanced_delta"] == pytest.approx(1.0)
    assert result["scier_family_delta"] is None
    assert not result["scier_coupled_document_resampling"]


def test_pipeline_uses_v9_owned_cache_writer():
    from experiments.qa_chunk_selection_v9.pipeline import DataStore

    assert DataStore.__module__ == "experiments.qa_chunk_selection_v9.data"
