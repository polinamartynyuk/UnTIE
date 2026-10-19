from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pytest

from experiments.qa_chunk_selection_v8.contract import (CELLS,H1_COLUMNS,H2_COLUMNS,H3_COLUMNS,SEM_COMMON,
    PRIMARY_AF,PRIMARY_HRR,SUPPORT_COMMON,frozen_contract)
from experiments.qa_chunk_selection_v8.features import assert_runtime_gold_free,normalize_weights,pairlex
from experiments.qa_chunk_selection_v8.io_guard import assert_v8_write
from experiments.qa_chunk_selection_v8.models import apply_policy,tune_policy

def _rows(n=16):
    rows=[]
    for q in range(n):
        utility=2.0 if q%3 else -1.0
        row={name:0.0 for name in H3_COLUMNS}
        row.update(question_id=f"q{q}",document_id=f"d{q}",canonical_document_id=f"d{q}",document_sha256=f"h{q}",
          question_lineage_id=f"l{q}",dataset_cell="scirex",corpus_family="scirex",language="en",canonical_aspect="METHOD",
          chunk_id=2,base_chunk_id=1,b0_rank=2,candidate_f1=50+utility,base_f1=50.0,candidate_tier="sufficient_positive" if utility>0 else "insufficient_evidence",
          base_tier="insufficient_evidence",candidate_hn=0,base_hn=0,utility=utility,harm_target=int(utility<0),weight=1.0)
        row["delta_fulllex"]=utility/10;row["delta_agreement"]=utility/5;rows.append(row)
    return rows

def test_contract_is_one_schema_for_all_cells():
    h=frozen_contract("hrr");f=frozen_contract("v44_af")
    assert tuple(h["cells"])==CELLS==tuple(f["cells"])
    assert h["primary"]==PRIMARY_HRR and f["primary"]==PRIMARY_AF
    assert h["no_manual_language_rules"] and f["no_manual_language_rules"]
    assert SEM_COMMON is False and SUPPORT_COMMON is True

def test_feature_ladder_is_monotone_and_sem_globally_disabled():
    assert set(H1_COLUMNS)<set(H2_COLUMNS)<set(H3_COLUMNS)
    assert "delta_sem" not in H3_COLUMNS

def test_pairlex_exact_v7_empty_semantics():
    assert pairlex("","")==1.0
    assert pairlex("answer","")==0.0
    assert pairlex("same answer","same answer")==1.0

def test_runtime_rejects_gold_fields():
    assert_runtime_gold_free({"selected_chunk_id":1,"answer":"x"})
    with pytest.raises(AssertionError):assert_runtime_gold_free({"gold_answers":["x"]})

def test_history_write_guard():
    with pytest.raises(RuntimeError):assert_v8_write(Path("artifacts/qa_chunk_selection_v7_rapr/forbidden.json"))
    path=assert_v8_write(Path("artifacts/qa_chunk_selection_v8_dual/test/out.json"))
    assert "qa_chunk_selection_v8_dual" in str(path)

def test_question_weighting_and_meta_oof_policy():
    rows=_rows();audit=normalize_weights(rows,pooled=False)
    assert audit["questions"]==len(rows)
    policy=tune_policy(rows,"H3",H3_COLUMNS,42,4,True)
    assert policy.selection["selection_source"]=="META_PRED_OOF"
    assert policy.selection["inner_folds_used"]==4
    assert set(policy.selection["standardized_coefficients"])==set(H3_COLUMNS)
    choices,_=apply_policy(rows,policy,True)
    assert len(choices)==len(rows)

def test_no_nonfinite_selection_values():
    policy=tune_policy(_rows(),"H3",H3_COLUMNS,42,4,True)
    def visit(value):
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(policy.selection)
