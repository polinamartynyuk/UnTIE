from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS
from experiments.qa_chunk_selection.v5_question_conditioned_residual.core import (
    choose_budget_threshold, feature_rows, gate_row, pair_rows, runtime_choice,
    uncertainty,
)
from experiments.qa_chunk_selection.v5_question_conditioned_residual.historical import tier
from experiments.qa_chunk_selection.v5_question_conditioned_residual.run import prepare_output
from experiments.qa_chunk_selection.v5_question_conditioned_residual.benchmark_worker import evaluation_records


def _rows():
    return feature_rows("q","d","aspect",42,[1,2,3],{1:3.,2:2.,3:1.},
        {1:"a",2:"b",3:"c"},{1:.1,2:.9,3:.2},
        {1:"insufficient_evidence",2:"sufficient_positive",3:"insufficient_evidence"},3)


def test_uncertainty_is_finite_and_constant_safe():
    row=uncertainty([1.,1.,1.],7)
    assert row["top1_zscore"]==0
    assert row["candidate_count"]==7
    assert all(np.isfinite(value) for value in row.values())


def test_gate_target_is_evidence_actionability():
    assert gate_row(_rows())["target"]==1
    rows=_rows(); rows[0]["label_tier"]="sufficient_positive"
    assert gate_row(rows)["target"]==0


def test_mirrored_pairs_are_balanced_and_question_weighted():
    pairs=pair_rows(_rows(),"A1")
    assert len(pairs)==4
    assert {row["target"] for row in pairs}=={0,1}
    assert abs(sum(row["weight"] for row in pairs)-1.0)<1e-12
    for left,right in zip(pairs[::2],pairs[1::2]):
        assert left["delta_b0"]==-right["delta_b0"]
        assert left["delta_q_answer_ranker"]==-right["delta_q_answer_ranker"]


def test_budget_threshold_uses_cost_only_and_k1_never_routes():
    p=[.9,.8,.7]; k=[5,1,2]
    tau=choose_budget_threshold(p,k,budget=2.0)
    routed=[prob>=tau and count>1 for prob,count in zip(p,k)]
    assert np.mean([count if route else 1 for route,count in zip(routed,k)])<=2.0+1e-8
    assert routed[1] is False


def test_runtime_switch_is_strictly_above_half():
    class Fixed:
        def __init__(self,p):self.p=p
        def predict_proba(self,frame):return np.array([[1-self.p,self.p] for _ in range(len(frame))])
    rows=_rows()
    assert runtime_choice(rows,Fixed(.5),"A1")[0]["chunk_id"]==1
    assert runtime_choice(rows,Fixed(.5001),"A1")[0]["chunk_id"]==2


def test_historical_inputs_exist_and_outputs_are_separate():
    for spec in CELLS.values():
        for path in (spec.records,spec.splits,spec.qa_cache,spec.historical_b0,spec.historical_reference,spec.qa_model/"config.json"):
            assert Path(path).is_file(), path
        assert "qa_chunk_selection_v5" not in str(spec.records)


def test_scier_cells_share_corpus_but_not_candidate_files():
    assert CELLS["scier_384"].corpus==CELLS["scier_128"].corpus=="SciER"
    assert CELLS["scier_384"].records!=CELLS["scier_128"].records



def test_prepare_output_accepts_launcher_created_empty_directory(tmp_path):
    output=tmp_path/"run"
    output.mkdir()
    assert prepare_output(output,resume=False)==output


def test_prepare_output_refuses_nonempty_directory_without_resume(tmp_path):
    import pytest
    output=tmp_path/"run"; output.mkdir(); (output/"existing.json").write_text("{}")
    with pytest.raises(FileExistsError):prepare_output(output,resume=False)
    assert prepare_output(output,resume=True)==output


def test_scirex_benchmark_filters_historical_universe_before_cv_lookup():
    rows=evaluation_records(CELLS["scirex"],run=0,limit=None)
    assignment=set(__import__("json").loads(CELLS["scirex"].splits.read_text())["assignment"])
    assert rows
    assert all(row.document_id in assignment for row in rows)
    assert all(str(row.question_id) in {str(x["question_id"]) for x in (__import__("json").loads(line) for line in CELLS["scirex"].historical_b0.read_text().splitlines() if line.strip())} for row in rows)
