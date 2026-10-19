from pathlib import Path
import pytest
from experiments.qa_chunk_selection_v7.contract import ADMISSION_NUMERIC,MAX_QA,SELECTOR_NUMERIC,selector_columns
from experiments.qa_chunk_selection_v7.features import assert_runtime_gold_free,normalize_weights,pairlex
from experiments.qa_chunk_selection_v7.io_guard import assert_v7_write
from experiments.qa_chunk_selection_v7.models import _simulate,fold_for,policy_metrics

def row(q="q",chunk=1,rank=2,utility=10):
    return {"dataset_cell":"scirex","question_id":q,"chunk_id":chunk,"b0_chunk_id":0,"b0_rank":rank,"candidate_f1":50+utility,"baseline_f1":50,
      "candidate_tier":"sufficient_positive" if utility>0 else "insufficient_evidence","baseline_tier":"insufficient_evidence",
      "candidate_hn":0,"baseline_hn":0,"effective_k":5}

def test_v7_namespace_and_historical_guard():
    with pytest.raises(PermissionError):assert_v7_write(Path("artifacts/qa_chunk_selection_v6_universal/no.json"))
    assert "qa_chunk_selection_v7_rapr" in str(assert_v7_write(Path("artifacts/qa_chunk_selection_v7_rapr/test/x.json")))
def test_pairlex_empty_policy_and_gold_free_runtime():
    assert pairlex("","")==1;assert pairlex("text","")==0;assert pairlex("same","same")==1
    assert_runtime_gold_free({"question_id":"q","evaluated_candidates":[{"pairlex":1.0}]})
    with pytest.raises(AssertionError):assert_runtime_gold_free({"actual_utility":1})
def test_primary_features_exclude_forbidden_identity_and_aspect():
    joined=" ".join((*ADMISSION_NUMERIC,*SELECTOR_NUMERIC)).lower()
    assert "aspect" not in joined and "dataset" not in joined and "source" not in joined and "sem" not in joined
    assert "pairlex" not in selector_columns("S3-noLEX")
def test_progressive_stop_and_hard_cap():
    rows=[row(chunk=1,rank=2,utility=5),row(chunk=2,rank=3,utility=20),row(chunk=3,rank=4,utility=-10)]
    mu=[2,10,-4];harm=[.1,.1,.9];admit=[.9,.8,.7];thresholds={"tau_admit":.3,"tau_utility":1,"tau_harm":.2}
    choices=_simulate(rows,mu,harm,admit,"S3",thresholds)
    assert choices[0]["selected_chunk_id"]==1 and choices[0]["qa_forwards"]==2
    assert policy_metrics(choices)["max_qa_forwards"]<=MAX_QA
def test_fallback_policy_has_no_extra_qa_even_for_full_system():
    rows=[row(chunk=1,rank=2),row(chunk=2,rank=3)]
    thresholds={"tau_admit":float("inf"),"tau_utility":float("inf"),"tau_harm":0.0}
    choices=_simulate(rows,[2,2],[.1,.1],[.9,.9],"B0_FALLBACK",thresholds)
    assert choices[0]["qa_forwards"]==1 and choices[0]["switch"]==0

def test_admission_order_differs_from_b0_diagnostic():
    rows=[row(chunk=1,rank=2),row(chunk=2,rank=3)];thresholds={"tau_admit":.3,"tau_utility":1,"tau_harm":.2}
    primary=_simulate(rows,[2,2],[.1,.1],[.4,.9],"S3",thresholds)
    b0=_simulate(rows,[2,2],[.1,.1],[.4,.9],"S3-B0-order",thresholds)
    assert primary[0]["selected_chunk_id"]==2 and b0[0]["selected_chunk_id"]==1
def test_question_and_family_weights():
    rows=[]
    for cell,family,lineage in (("scier_384","scier","L"),("scier_128","scier","L"),("russerrc","russerrc","R")):
      for i in range(2):rows.append({"dataset_cell":cell,"corpus_family":family,"question_id":cell,"question_lineage_id":lineage,"weight":.5})
    audit=normalize_weights(rows,True);assert audit["families"]["scier"]==pytest.approx(audit["families"]["russerrc"])
def test_meta_fold_is_document_grouped_and_seed_specific():
    assert fold_for("doc",42,4)==fold_for("doc",42,4);assert any(fold_for(f"doc{i}",42,4)!=fold_for(f"doc{i}",43,4) for i in range(20))
def test_launcher_uses_project_venv_and_project_logs():
    text=Path("experiments/qa_chunk_selection_v7/launch_background.sh").read_text();assert 'untie_venv/bin/python3' in text;assert 'logs/qa_chunk_selection_v7' in text;assert ')/../.." && pwd)' in text

def test_prediction_evidence_metrics_stay_in_evaluation_namespace():
    from experiments.qa_chunk_selection_v7.pipeline import predictions
    common={"question_id":"q","dataset_cell":"scirex","document_id":"d","canonical_document_id":"d",
      "document_sha256":"h","question_lineage_id":"l","canonical_aspect":"task","language":"en","effective_k":2}
    runtime=[{**common,"chunk_id":0,"b0_rank":1,"b0_score_raw":2.0,"b0_score_norm":1.0,"qa_answer":"wrong"},
             {**common,"chunk_id":1,"b0_rank":2,"b0_score_raw":1.0,"b0_score_norm":-1.0,"qa_answer":"right"}]
    target={"gold_answers":["right"],"baseline_f1":0.0,"candidate_f1":{"0":0.0,"1":100.0},
      "label_tier":{"0":"insufficient_evidence","1":"sufficient_positive"},"hard_negative":{"0":1,"1":0},
      "chunk_label":{"0":"hard_negative","1":"positive"},"primary_class":{"0":"hard_negative","1":"robust_positive"}}
    row=predictions("scirex",runtime,[],{"q":target},"S0","local",0,42)[0]
    assert {"hit@k","hn@k","f1_given_selected_positive","amb@1","gsrn@1"} <= set(row["evaluation"])
    assert "gold_answers" not in row["runtime"] and row["evaluation"]["hit@k"]==1 and row["evaluation"]["hn@k"]==1

def test_fallback_thresholds_are_json_finite(tmp_path):
    from experiments.qa_chunk_selection_v7.io_guard import validate_finite
    from experiments.qa_chunk_selection_v7.models import tune_policy
    rows=[]
    # Positive utility is impossible and HN safety is violated for every active switch,
    # which forces the explicit B0 fallback branch.
    for q in range(8):
      for rank in (2,3):
       value=row(q=f"q{q}",chunk=rank,rank=rank,utility=-5)
       value.update({"utility":-5.0,"canonical_document_id":f"d{q}","language":"en","corpus_family":"scirex","question_lineage_id":f"l{q}",
         "weight":.5,"candidate_hn":1,"baseline_hn":0,"admission_target":1,"harm_target":1,
         "candidate_rank_norm":(rank-1)/2,"b0_score_norm":0.0,"delta_b0_norm":0.0,"candidate_count":3,
         "gap12_norm":0.0,"gap13_norm":0.0,"b0_entropy_norm":0.0,"top1_z":0.0,"delta_qarel_raw":0.0,
         "pairlex":0.0,"answer_valid_i":1,"answer_valid_0":1,"both_empty":0,"qarel_x_margin":0.0,
         "pairlex_x_margin":0.0,"qarel_x_pairlex":0.0,"b0_x_margin":0.0})
       rows.append(value)
    policy=tune_policy(rows,"S3",42,4)
    assert policy.selection["fold_policy"]=="B0_FALLBACK"
    assert policy.selection["selected_metrics"]["tau_admit"]=="inf"
    assert policy.selection["selected_metrics"]["tau_utility"]=="inf"
    validate_finite(policy.selection)
