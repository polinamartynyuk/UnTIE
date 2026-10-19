"""V4.4 anti-leakage, isolation, hierarchical gates and synthetic integration."""
from dataclasses import replace, fields
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection.v4_4.config import ExperimentConfigV44
from experiments.qa_chunk_selection.v4_4.contract import contract, V6_GLOBAL
from experiments.qa_chunk_selection.v4_4.routing import (
    RouteFeature, calibrate, available, use_reader, support_veto, choose_prediction,
)
from experiments.qa_chunk_selection.v4_4.io_guard import claim_output, load_unlocked_snapshot
from experiments.qa_chunk_selection.v4_4.select import primary_gate, select_candidate
from experiments.qa_chunk_selection.v4_4.freeze import build_draft
from experiments.qa_chunk_selection.v4_4.encoder import ReplayEncoder
from experiments.qa_chunk_selection.v4_4.evaluate import paired_delta, identity
from experiments.qa_chunk_selection.v4_4.diagnostics import routing_audit, confidence_bins
from experiments.qa_chunk_selection.v4_3.encoder import text_cache_key
from experiments.qa_chunk_selection.v4_3.features import support_gain, support_rel, semantic_pack
from experiments.qa_chunk_selection.v4_3.text import local_evidence_sentence
from experiments.qa_chunk_selection.v3.data import synthetic_records


def feature(index=0, **kwargs):
    base = RouteFeature(f"q{index}", f"d{index}", 42, "train", index/100, True, 1, 0)
    return replace(base, **kwargs)


def test_q70_only_noneval_available_feature_rows():
    source = [feature(i) for i in range(100)]
    poison = [feature(200+i, role="eval", confidence=1e9) for i in range(100)]
    poison += [feature(400+i, available=False, confidence=-1e9) for i in range(100)]
    result = calibrate(source+poison)
    assert result["tau"] == pytest.approx(np.quantile([i/100 for i in range(100)], .7))
    assert result["n_threshold_source_rows"] == 100
    assert {qid for qid, _ in result["source_keys"]} == {f"q{i}" for i in range(100)}
    assert result["uses_gold"] is False
    assert not {"gold_answers", "source_label", "token_f1", "aspect"} & {f.name for f in fields(RouteFeature)}


def test_duplicate_threshold_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        calibrate([feature(), feature()])


@pytest.mark.parametrize("count,enabled", [(0, False), (99, False), (100, True)])
def test_minimum_router_exact_fallback(count, enabled):
    threshold = calibrate([feature(i) for i in range(count)])
    assert threshold["enabled"] is enabled
    v6, reader = {"selected_chunk_id": 0, "qa_answer": "v6", "ranked_ids": [0,1]}, {"selected_chunk_id": 1, "qa_answer": "reader", "ranked_ids": [1,0]}
    chosen, switched = choose_prediction(v6, reader, feature(confidence=1.), threshold)
    assert chosen == (reader if enabled else v6)
    assert switched is enabled


@pytest.mark.parametrize("n,cid,confidence,expected", [(3, 1, .8, False), (4, 1, .8, True), (5, 9, .8, False), (5, 1, None, False), (5, 1, float('nan'), False)])
def test_availability(n, cid, confidence, expected):
    pack = {"n_valid": n, "best_triple": [0,1,2], "consensus_conf": confidence}
    assert available(pack, cid) is expected
    f = feature(available=available(pack, cid), confidence=confidence)
    assert use_reader(f, {"enabled": True, "tau": .7}) is expected


def test_switch_threshold_equality_and_unavailable_exact_v6():
    threshold = {"enabled": True, "tau": .7}
    a, b = {"top": 0}, {"top": 1}
    assert choose_prediction(a,b,feature(confidence=.7),threshold) == (b, True)
    assert choose_prediction(a,b,feature(confidence=.8,available=False),threshold) == (a, False)


def test_support_q20_potential_switch_distribution():
    source = [feature(i, support_rel=i/100, confidence=-100) for i in range(100)]
    excluded = [feature(200, reader_id=0, support_rel=-100), feature(201, role="eval", support_rel=100),
                feature(202, available=False, support_rel=100)]
    threshold = calibrate(source+excluded, support=True)
    assert threshold["n_threshold_source_rows"] == 100
    assert threshold["tau"] == pytest.approx(.198)
    assert support_veto(feature(support_rel=0), threshold, switched=True)
    assert not support_veto(feature(support_rel=threshold["tau"]), threshold, switched=True)
    assert not support_veto(feature(support_rel=0), threshold, switched=False)


def test_support_formula_and_zero_variance():
    gain = support_gain({0:.4,1:.5}, {0:.7,1:.8}, [0,1])
    assert gain == pytest.approx({0:.3,1:.3})
    assert support_rel(gain) == {0:0.,1:0.}


def test_local_evidence_uses_predicted_span_location():
    text = "Old method. We introduce a new method."
    start = text.rindex("method")
    evidence = local_evidence_sentence(text, start_char=start, end_char=start+6, predicted_answer="method")
    assert evidence["text"] == "We introduce a new method."


def test_semantic_confidence_no_self_and_no_nan():
    vectors = {cid: np.array([1.,0.]) for cid in range(4)}
    pack = semantic_pack({cid:"same" for cid in range(4)}, vectors, top_ids=list(range(4)))
    assert pack["consensus_conf"] == pytest.approx(0.)
    assert all(v == 0 for v in pack["sem_rel"].values())


def test_h1_h2_hierarchy_and_diagnostic_exclusion():
    v6 = {"hn@1": .05, "hit@1": .43}
    asem = {"hn@1": .06, "hit@1": .425}
    delta = {"mean_delta": 1., "ci95_low": .1, "ci95_high": 2.}
    h1 = primary_gate(v6, asem, delta)
    assert h1["pass"]
    alex = {"hn@1": .055, "hit@1": .43}
    noninferior = {"mean_delta": -.1, "ci95_low": -.49, "ci95_high": .3}
    assert select_candidate(v6,asem,alex,h1,noninferior)["preferred"] == "ALEX"
    assert select_candidate(v6,asem,alex,{"pass":False},None)["preferred"] == "V6-K5"
    assert not primary_gate(v6,asem,{**delta,"ci95_low":0})["pass"]
    assert not primary_gate(v6,{**asem,"hn@1":.06001},delta)["pass"]
    assert not primary_gate(v6,{**asem,"hit@1":.42499},delta)["pass"]
    assert select_candidate(v6,asem,{**alex,"hn@1":.08},h1,delta)["preferred"] == "ASEM"
    assert select_candidate(v6,asem,alex,h1,{**noninferior,"ci95_low":-.5})["preferred"] == "ASEM"
    with pytest.raises(ValueError):
        build_draft("PREF-SUPPORT",None,{})


@pytest.mark.parametrize("relative", ["artifacts/qa_chunk_selection_v4_3", "artifacts/qa_chunk_selection_v4_2/new",
    "untie/new", "datasets/new", "artifacts", "experiments/qa_chunk_selection/v4_4"])
def test_protected_roots(tmp_path, relative):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(ValueError):
        claim_output(root/relative, root)
    assert not (root/relative).exists()


def test_no_overwrite_failed_completed_or_symlink(tmp_path):
    root = tmp_path/"project"
    root.mkdir()
    path = claim_output(root/"artifacts/new",root)
    (path/"valuable.txt").write_text("old")
    with pytest.raises(FileExistsError):
        claim_output(path, root)
    assert (path/"valuable.txt").read_text() == "old"
    link = tmp_path/"alias"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        claim_output(link/"new",root)


def test_mixed_snapshot_skips_locked_before_deserialization(tmp_path):
    unlocked = synthetic_records(n_docs=1)[0]
    file = tmp_path/"snapshot.jsonl"
    # Intentionally invalid locked record: must not be decoded as an InstanceRecord.
    file.write_text(json.dumps({"question_id":"locked", "split":"internal_test", "chunks":"poison"})+"\n"+json.dumps(unlocked.to_dict())+"\n")
    assert load_unlocked_snapshot(file,{unlocked.question_id}) == (unlocked,)
    with pytest.raises(ValueError, match="Locked"):
        load_unlocked_snapshot(file,{"locked"})


def test_readonly_embedding_reuse_and_missing_failure(tmp_path):
    previous = tmp_path/"old"
    previous.mkdir()
    spec = {"model_name":"sentence-transformers/all-MiniLM-L6-v2", "model_hash":"abc", "smoke":False,
            "frozen":True, "normalization_version":"v4.3-nfc-ws-v1"}
    (previous/"encoder_spec.json").write_text(json.dumps(spec))
    key = text_cache_key("answer",model_hash="abc",normalization_version=spec["normalization_version"])
    path = previous/"answer_embeddings/vec"/f"{key}.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.array([1.,0.]))
    before = path.read_bytes()
    encoder = ReplayEncoder(previous,tmp_path/"new")
    assert encoder.embed_one("answer",store="answer") == pytest.approx([1.,0.])
    assert encoder.backend is None
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        encoder.embed_one("missing",store="answer")


def test_seed_pairing_strict_and_document_bootstrap():
    a = [{"question_id":"q", "seed":s,"document_id":"d","token_f1":float(s)} for s in (42,43,44)]
    b = [{**r,"token_f1":r["token_f1"]+1} for r in reversed(a)]
    assert paired_delta(a,b,n_boot=20)["mean_delta"] == 1
    with pytest.raises(ValueError):
        paired_delta(a,b[:-1],n_boot=20)
    with pytest.raises(ValueError):
        paired_delta(a,b+[b[0]],n_boot=20)


def test_draft_one_runtime_vector_and_no_test_ready():
    draft = build_draft("ASEM",calibrate([feature(i) for i in range(100)]),{})
    assert draft["runtime"]["v6_weights"] == V6_GLOBAL
    assert draft["runtime"]["seed_policy"]["canonical_seed"] == 42
    assert "per_run" not in json.dumps(draft["runtime"])
    assert not draft["ready_for_internal_test"]
    assert not draft["b0_final_trained"]


def test_smoke_full_pipeline_and_fixed_reader_determinism(tmp_path, monkeypatch):
    from experiments.qa_chunk_selection.v4_4.pipeline import PipelineV44
    from experiments.qa_chunk_selection.v4_3.ranking import lexcore_scores, semcore_scores
    from experiments.qa_chunk_selection.v4_4.contract import LEX_FIXED, SEM_FIXED
    def forbidden(*a, **k):
        raise AssertionError("Smoke attempted real model/data load")
    monkeypatch.setattr("experiments.qa_chunk_selection.v4_4.encoder.MiniLMEncoder",forbidden)
    monkeypatch.setattr("experiments.qa_chunk_selection.v4_4.pipeline.load_unlocked_snapshot",forbidden)
    pipe = PipelineV44(ExperimentConfigV44(output_dir=tmp_path/"smoke",smoke=True))
    assert pipe.run() == 0
    assert pipe.selection["preferred"] == "V6-K5"
    assert pipe.selection["h2"]["tested"] is False
    for key in list(pipe.b0)[:3]:
        item, b0 = pipe.items[key[0]],pipe.b0[key]
        qa = {cid:pipe.qa_store[key[0],cid] for cid in top_k(pipe,key)}
        for name, fn, kwargs in (("LEX-FIX",lexcore_scores,LEX_FIXED),("SEM-FIX",semcore_scores,SEM_FIXED)):
            args = (item,b0,qa) if name == "LEX-FIX" else (item,b0,qa,{int(k):v for k,v in pipe.packs[key]["sem_rel"].items()})
            expected = fn(*args,k=5,min_overlap=.6,**kwargs)
            assert {cid:pipe.fixed[key,name][1][cid] for cid in top_k(pipe,key)} == pytest.approx(expected)
            assert fn(*args,k=5,min_overlap=.6,**kwargs) == expected
    output = tmp_path/"smoke"
    assert (output/"reports/results_v4_4.md").is_file()
    assert json.loads((output/"status.json").read_text())["exit_code"] == 0
    assert (output/"progress.jsonl").read_text().count('"stage"') > 20
    assert json.loads((output/"support_diagnostic/status.json").read_text())["skipped"]


def top_k(pipe,key):
    return pipe.b0[key]["ranked_ids"][:5]


def test_conditional_support_and_soft_are_diagnostic_only(tmp_path, monkeypatch):
    from experiments.qa_chunk_selection.v4_4.pipeline import PipelineV44
    pipe = PipelineV44(ExperimentConfigV44(output_dir=tmp_path/"conditional",smoke=True,soft=True))
    for stage in ("init","references","cache","features","thresholds","replay"):
        getattr(pipe,f"stage_{stage}")()
    pipe.selection = {"preferred":"ASEM", "h1":{"pass":True}}
    pipe.stage_support()
    assert "ASOFT" in pipe.predictions and "PREF-SUPPORT" in pipe.predictions
    assert pipe.selection["preferred"] == "ASEM"
    # Small sample disables support; every prediction must be exactly PREF.
    assert identity(pipe.predictions["ASEM"],pipe.predictions["PREF-SUPPORT"])["ok"]
    # Exercise an enabled Q20 diagnostic with controlled runtime features: all
    # candidate switches lack support and must return the exact V6 prediction.
    pipe.config = replace(pipe.config, soft=False)
    v6_by_key = {(r['question_id'],r['seed']):r for r in pipe.predictions['V6-K5']}
    for row in pipe.predictions['ASEM']:
        row['reader_only_used'] = True
    for run in range(5):
        eval_features = []
        for row in pipe.predictions['ASEM']:
            if row['cv_run'] != run:
                continue
            key = row['question_id'],row['seed']
            v6_id = v6_by_key[key]['selected_chunk_id']
            eval_features.append(RouteFeature(key[0],row['document_id'],key[1],'eval',.9,True,(v6_id+1)%4,v6_id))
        calibration = [feature(i,question_id=f'calibration-{i}',role='train',support_rel=1.) for i in range(100)]
        pipe.routes[run,'ASEM'] = {(f.question_id,f.seed):f for f in eval_features+calibration}
    monkeypatch.setattr(pipe,'_support',lambda key: {cid:1. if key[0].startswith('calibration-') else -1. for cid in range(4)})
    pipe.stage_support()
    assert all(r['support_veto'] for r in pipe.predictions['PREF-SUPPORT'])
    assert identity(pipe.predictions['V6-K5'],pipe.predictions['PREF-SUPPORT'])['ok']
    assert pipe.selection['preferred'] == 'ASEM'


def test_launcher_refuses_protected_output_before_log_creation(tmp_path):
    import os
    import subprocess
    import sys
    from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
    log = PROJECT_ROOT/'logs/v44_should_never_be_created_test.log'
    result = subprocess.run(['bash',str(PROJECT_ROOT/'experiments/qa_chunk_selection/v4_4/launch_background.sh')],
        env={**os.environ,'PYTHON':sys.executable,'OUT_DIR':str(PROJECT_ROOT/'artifacts/qa_chunk_selection_v4_3'),
             'LOG':str(log),'SMOKE':'1'},capture_output=True,text=True)
    assert result.returncode != 0
    assert not log.exists()


def test_features_do_not_use_gold_or_source_labels():
    from experiments.qa_chunk_selection.v4_3.ranking import lexcore_scores, semcore_scores
    item = synthetic_records(n_docs=1)[0]
    b0 = {'ranked_ids':[0,1,2,3], 'scores':{'0':4.,'1':3.,'2':2.,'3':1.}}
    qa = {cid:{'qa_answer':str(cid%2)} for cid in range(4)}
    mutated = replace(item,gold_answers=('poison',),gold_spans=(),chunks=tuple(
        replace(c,label='hard_negative',primary_class='hard_negative',gold_supporting=False) for c in item.chunks))
    semrel = {0:1.,1:0.,2:1.,3:0.}
    assert lexcore_scores(item,b0,qa,k=5,min_overlap=.6,lambda_m=.25,lambda_t=2.) == lexcore_scores(mutated,b0,qa,k=5,min_overlap=.6,lambda_m=.25,lambda_t=2.)
    assert semcore_scores(item,b0,qa,semrel,k=5,min_overlap=.6,lambda_m=.5,lambda_s=.25) == semcore_scores(mutated,b0,qa,semrel,k=5,min_overlap=.6,lambda_m=.5,lambda_s=.25)


def test_support_cross_sentence_span_uses_complete_window():
    from experiments.qa_chunk_selection.v4_4.evidence import local_evidence_sentence
    text = 'First target. Second target. Last sentence.'
    start, end = text.index('target'), text.index('Second')+len('Second')
    evidence = local_evidence_sentence(text,start_char=start,end_char=end,predicted_answer=text[start:end])
    assert evidence['mode'] == 'token_window'
    assert text[start:end] in evidence['text']
