from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.cluster_discriminative_keywords.candidates import merge_pools
from experiments.cluster_discriminative_keywords.contract import experiment_contract
from experiments.cluster_discriminative_keywords.discrimination import (
    KeywordProfile,
    discriminative_score,
    keyword_in_text,
    load_profiles_by_pool,
    normalize_keyword,
    profile_from_row,
    score_profiles,
    shrinkage_lambda,
    support_threshold,
)
from experiments.cluster_discriminative_keywords.io_util import assert_isolated_output
from experiments.cluster_discriminative_keywords.qa_eval import centroid_index, predict_consensus
from experiments.cluster_discriminative_keywords.ranking import rank_metrics, summarize_ranks
from experiments.cluster_discriminative_keywords.run import build_parser
from untie.keyword_evidence import CachedChunkAnswer, CachedDocumentEvidence


def test_contract_keeps_roberta_and_frozen_clustering() -> None:
    contract = experiment_contract()
    assert "RoBERTa" in str(contract["qa_reader"]) or "cached" in str(contract["qa_reader"]).casefold()
    assert "C_keybert" in str(contract["clustering"])
    assert "replacement" in str(contract["keybert_is_not"]).casefold()
    assert "negative diff" in str(contract["diff_is_not"]).casefold()


def test_discriminative_score_sign_matches_the_prompt() -> None:
    assert discriminative_score(0.8, 0.1) > 0
    assert discriminative_score(0.1, 0.8) < 0
    assert abs(discriminative_score(0.4, 0.4)) < 1e-9


def test_keyword_match_is_phrase_aware() -> None:
    assert keyword_in_text("frost heave", "The frost heave prediction method")
    assert not keyword_in_text("frost heave", "frosty weather heaves")
    assert normalize_keyword("  Object\tDetection ") == "object detection"


def test_support_threshold_adapts_to_cluster_size() -> None:
    assert support_threshold(40, docs=2, fraction=0.05) == 2
    assert support_threshold(40, docs=2, fraction=0.10) == 4
    assert shrinkage_lambda(30, n0=30) == 0.5
    assert shrinkage_lambda(70, n0=30) < shrinkage_lambda(20, n0=30)


def test_score_profiles_drops_low_support_and_keeps_negative_h() -> None:
    profiles = [
        KeywordProfile(cluster_id=0, keyword="objective of study"),
        KeywordProfile(cluster_id=0, keyword="experimental setup"),
        KeywordProfile(cluster_id=0, keyword="once only"),
    ]
    positive = ["The objective of study is frost heave prediction."]
    hard = ["The experimental setup uses a numerical method."]
    easy = ["Table shows results."]
    docs = positive * 3 + hard * 3
    scored = score_profiles(profiles, positive, hard, easy, docs)
    keys = {item.keyword: item for item in scored}
    assert "objective of study" in keys
    assert keys["objective of study"].discriminative_score > 0
    assert keys["experimental setup"].discriminative_score < 0
    assert "once only" not in keys


def test_merge_pools_unions_sources_without_mixing_weights() -> None:
    left = {"task": KeywordProfile(cluster_id=0, keyword="task", source_attention=1, attention_score=0.2)}
    right = {"task": KeywordProfile(cluster_id=0, keyword="task", source_topic=1, topic_score=1.5)}
    merged = merge_pools(0, [left, right])
    item = merged["task"]
    assert item.source_attention == 1
    assert item.source_topic == 1
    assert item.attention_score == 0.2
    assert item.topic_score == 1.5


def test_rank_metrics_and_centroid_subset() -> None:
    metrics = rank_metrics([0.1, 0.9, 0.2], gold_index=1)
    assert metrics["evidence_recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    summary = summarize_ranks([metrics])
    assert summary["evidence_recall@5"] == 1.0
    sim = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    assert centroid_index(sim, [0, 1, 2]) in {0, 1}


def test_output_dir_refuses_protected_trees() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(root / "artifacts" / "keybert_topic_clustering" / "en_full", root)
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(root / "artifacts" / "answer_aggregation" / "en_full", root)
    allowed = assert_isolated_output(
        root / "artifacts" / "cluster_discriminative_keywords" / "en", root
    )
    assert "cluster_discriminative_keywords" in str(allowed)


def test_launch_script_points_at_isolated_output() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "cluster_discriminative_keywords"
        / "launch_background.sh"
    )
    raw = script.read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert "nohup" in text
    assert "untie_venv/bin/python" in text
    assert "artifacts/cluster_discriminative_keywords" in text
    assert "--cluster-dir" in text
    assert "hierarchical_topic_optimized" not in text.split("OUT_DIR=", 1)[1][:500]
    assert 'QUESTION="${QUESTION:-Which task was solved?}"' in text
    assert 'RESUME="${RESUME:-0}"' in text
    assert "--resume" in text


def test_parser_exposes_resume() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--dataset", "x.json", "--output-dir", "out", "--resume"]
    )
    assert args.resume is True


def test_profile_csv_roundtrip(tmp_path: Path) -> None:
    original = KeywordProfile(
        cluster_id=2,
        keyword="object detection",
        source_attention=1,
        attention_score=0.4,
        discriminative_score=1.25,
        final_weight=0.9,
        ngram=2,
    )
    loaded = profile_from_row({**original.to_row(), "pool": "K4"})
    assert loaded.keyword == "object detection"
    assert loaded.cluster_id == 2
    assert loaded.final_weight == pytest.approx(0.9)
    path = tmp_path / "profiles.csv"
    path.write_text(
        "cluster_id,keyword,source_attention,source_diff,source_question_keybert,"
        "source_topic,attention_score,diff_score,question_score,topic_score,"
        "positive_frequency,negative_frequency,hard_negative_frequency,"
        "document_support,document_support_count,discriminative_score,"
        "global_discriminative_score,final_weight,ngram,pool\n"
        "1,frost heave,1,0,0,0,0.2,0,0,0,0.5,0.1,0.1,0.2,3,1.1,0.4,0.8,2,K4\n",
        encoding="utf-8",
    )
    grouped = load_profiles_by_pool(path)
    assert grouped["K4"][1][0].keyword == "frost heave"
    assert grouped["K4"][1][0].final_weight == pytest.approx(0.8)


def test_predict_consensus_passes_cached_answer_kwargs() -> None:
    chunks = (
        CachedChunkAnswer(0, "The task is object detection.", 6, "object detection", 0.9),
        CachedChunkAnswer(1, "Experimental setup is described.", 5, "experimental setup", 0.2),
    )
    evidence = CachedDocumentEvidence(
        doc_id="d1",
        aspect_id="task",
        question="Which task was solved?",
        references=("object detection",),
        chunks=chunks,
        answer_similarity=((1.0, 0.15), (0.15, 1.0)),
        baseline_answer="object detection",
        baseline_confidence=0.9,
        candidates=(),
        fingerprint="test",
    )
    prediction, meta = predict_consensus(evidence, [0.8, 0.1])
    assert prediction is not None
    assert meta["mode"] == "consensus"
    assert meta["qa_chunks"] == 2
