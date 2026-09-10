from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from experiments.keybert_qa_integration.config import (
    ExperimentParams,
    ModeFlags,
    core_mode_specs,
    default_params,
)
from experiments.keybert_qa_integration.contract import qa_contract
from experiments.keybert_qa_integration.hard_negatives import contains_gold_span
from experiments.keybert_qa_integration.io_util import assert_isolated_output
from experiments.keybert_qa_integration.scoring import (
    aligned_keyword_scores,
    apply_global_local,
    keybert_chunk_scores,
    mix_named_channels,
    reweight_attention,
    select_channel_plan,
)
from untie.domain import Sentence, TextChunk
from untie.ranking import WeightedKeyword, mix_score_channels


def _chunk(index: int, text: str) -> TextChunk:
    return TextChunk(
        sentences=(Sentence(text=text, number=index, tokens=tuple(text.split())),),
        text=text,
        token_count=len(text.split()),
    )


def _keyword(word: str, *, attention: float = 1.0, diff: float = 0.4) -> WeightedKeyword:
    return WeightedKeyword(
        word=word,
        lemma=word,
        stem=word,
        attention_weight=attention,
        score_difference=diff,
    )


def test_qa_contract_keeps_roberta_and_rejects_minilm_as_qa() -> None:
    contract = qa_contract()
    assert contract["actual_class"] == "untie.models.ExtractiveQuestionAnswerer"
    assert "attention" in contract["keybert_role"].casefold() or "prior" in contract["keybert_role"].casefold()
    assert "MiniLM" in str(contract["not_used_as_qa"])
    assert "replacement" in str(contract["keybert_is_not"]).casefold()


def test_boolean_flags_toggle_independently() -> None:
    base = ModeFlags()
    assert base.is_baseline()
    for field in fields(ModeFlags):
        if field.type not in {bool, "bool"}:
            continue
        on = replace(base, **{field.name: True})
        off = replace(on, **{field.name: False})
        assert getattr(on, field.name) is True
        assert getattr(off, field.name) is False
        if field.name in {
            "use_attention",
            "use_diff",
            "use_keybert_chunk_score",
            "use_answer_consensus",
            "use_keybert_clusters",
            "use_global_local",
            "use_keybert_filter",
        }:
            assert not on.is_baseline()
            assert off.is_baseline()


def test_channel_plan_attention_only_vs_bundled_att_diff() -> None:
    params = ExperimentParams()
    attention = select_channel_plan(ModeFlags(use_attention=True), params)
    assert attention["attention"] == params.attention_weight
    assert attention["bundled"] == 0.0
    assert attention["keybert"] == 0.0
    bundled = select_channel_plan(
        ModeFlags(use_attention=True, use_diff=True, attention_source="static"),
        params,
    )
    assert bundled["bundled"] == params.attention_weight
    assert bundled["attention"] == 0.0
    independent = select_channel_plan(
        ModeFlags(
            use_attention=True,
            use_diff=True,
            use_keybert=True,
            use_keybert_chunk_score=True,
            mix_diff_independent=True,
        ),
        params,
    )
    assert independent["attention"] == params.attention_weight
    assert independent["diff"] == params.diff_weight
    assert independent["keybert"] == params.keybert_weight
    kb_only = select_channel_plan(
        ModeFlags(use_keybert=True, use_keybert_chunk_score=True),
        params,
    )
    assert kb_only["attention"] == 0.0
    assert kb_only["diff"] == 0.0
    assert kb_only["bundled"] == 0.0
    assert kb_only["keybert"] == params.keybert_weight


def test_weight_ratio_one_is_attention_only_and_half_uses_diff() -> None:
    chunks = [
        _chunk(0, "graph parsing with neural attention"),
        _chunk(1, "unrelated cooking recipes and baking"),
    ]
    keywords = [_keyword("parsing", attention=1.0, diff=0.0)]
    att = aligned_keyword_scores(chunks, keywords, weight_ratio=1.0, keep_unmatched=True)
    diff = aligned_keyword_scores(chunks, keywords, weight_ratio=0.0, keep_unmatched=True)
    mixed = aligned_keyword_scores(chunks, keywords, weight_ratio=0.5, keep_unmatched=True)
    assert len(att) == 2
    assert att[0] > att[1]
    assert diff[0] == pytest.approx(0.0)
    assert mixed[0] > mixed[1]
    assert mixed[0] < att[0]


def test_mix_named_channels_minmax_then_weighted_sum() -> None:
    mixed = mix_named_channels(
        {"attention": [0.0, 10.0], "keybert": [5.0, 1.0]},
        {"attention": 1.0, "keybert": 0.2, "diff": 0.0},
    )
    expected = mix_score_channels([[0.0, 10.0], [5.0, 1.0]], [1.0, 0.2])
    assert mixed == pytest.approx(expected)
    assert mixed[0] == pytest.approx(0.2)
    assert mixed[1] == pytest.approx(1.0)


def test_soft_filter_formula_and_hard_filter_does_not_zero() -> None:
    keywords = [_keyword("task", attention=2.0)]
    soft = reweight_attention(
        keywords, [0.2], mode="soft", threshold=0.5, floor=0.25
    )
    assert soft[0].attention_weight == pytest.approx(2.0 * (0.5 + 0.5 * 0.2))
    hard = reweight_attention(
        keywords, [0.1], mode="hard_threshold", threshold=0.5, floor=0.25
    )
    assert hard[0].attention_weight == pytest.approx(0.5)
    assert hard[0].attention_weight > 0.0
    kept = reweight_attention(
        keywords, [0.9], mode="hard_threshold", threshold=0.5, floor=0.25
    )
    assert kept[0].attention_weight == pytest.approx(2.0)
    with pytest.raises(ValueError, match="floor"):
        reweight_attention(keywords, [0.1], mode="hard_threshold", threshold=0.5, floor=0.0)


def test_global_local_formula() -> None:
    keywords = [_keyword("task", attention=2.0)]
    updated = apply_global_local(
        keywords,
        [0.5],
        {"task": 0.4},
        alpha=1.0,
        beta=1.0,
    )
    assert updated[0].attention_weight == pytest.approx(2.0 * (1 + 0.4) * (1 + 0.5))


def test_keybert_weighted_and_max_scores() -> None:
    chunks = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    phrases = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    phrases = phrases / np.linalg.norm(phrases, axis=1, keepdims=True)
    chunks = chunks / np.linalg.norm(chunks, axis=1, keepdims=True)
    maximum = keybert_chunk_scores(chunks, phrases, [0.9, 0.1], variant="max", mean_top_n=1)
    weighted = keybert_chunk_scores(
        chunks, phrases, [1.0, 0.0], variant="weighted", mean_top_n=1
    )
    assert maximum[0] >= maximum[1]
    assert weighted[0] > weighted[1]
    empty = keybert_chunk_scores(
        chunks, np.zeros((0, 2), dtype=np.float32), [], variant="max", mean_top_n=1
    )
    assert list(empty) == [0.0, 0.0]


def test_core_modes_cover_the_prompt_table() -> None:
    names = [item.name for item in core_mode_specs()]
    assert names == [
        "qa_baseline",
        "qa_attention",
        "qa_attention_diff",
        "qa_keybert",
        "qa_attention_keybert",
        "qa_attention_diff_keybert",
    ]
    assert core_mode_specs()[3].flags.use_attention is False
    assert core_mode_specs()[3].flags.use_keybert_chunk_score is True
    assert core_mode_specs()[1].flags.use_keybert is False
    params = default_params()
    assert params.keybert_weight == 0.2
    assert params.keybert_top_k == 10
    assert params.keybert_sim_threshold == 0.5


def test_gold_span_check_does_not_use_test_gold_accidentally() -> None:
    assert contains_gold_span("The task is pose estimation here", ["pose estimation"], min_chars=4)
    assert not contains_gold_span("The task is classification", ["pose estimation"], min_chars=4)


def test_output_dir_refuses_production_and_hyp2_trees() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(
            root / "artifacts" / "hierarchical_topic_optimized" / "en" / "evaluation",
            root,
        )
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(
            root / "artifacts" / "keybert_topic_clustering" / "en_full",
            root,
        )
    allowed = assert_isolated_output(
        root / "artifacts" / "keybert_qa_integration" / "en",
        root,
    )
    assert "keybert_qa_integration" in str(allowed)


def test_predict_baseline_and_keybert_only_do_not_mix_attention() -> None:
    from experiments.keybert_qa_integration.predict import DocumentFeatures, predict_document
    from experiments.keybert_topic_clustering.keyphrases import Keyphrase
    from untie.keyword_evidence import CachedChunkAnswer, CachedDocumentEvidence

    chunks = (
        CachedChunkAnswer(0, "unrelated baking methods", 3, "baking", 0.2),
        CachedChunkAnswer(1, "pose estimation is solved", 5, "pose estimation", 0.9),
    )
    evidence = CachedDocumentEvidence(
        doc_id="d1",
        aspect_id="task",
        question="Which task was solved?",
        references=("pose estimation",),
        chunks=chunks,
        answer_similarity=((1.0, 0.1), (0.1, 1.0)),
        baseline_answer="pose estimation",
        baseline_confidence=0.9,
        candidates=(),
        fingerprint="x",
    )
    phrases = (Keyphrase("pose estimation", 0.9),)
    phrase_emb = np.asarray([[1.0, 0.0]], dtype=np.float32)
    chunk_emb = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    att = (_keyword("baking", attention=5.0),)
    features = DocumentFeatures(
        evidence=evidence,
        phrases=phrases,
        phrase_embeddings=phrase_emb,
        chunk_embeddings=chunk_emb,
        attention_keywords=att,
        attention_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    params = ExperimentParams(keep_unmatched=True, keybert_top_k=1)
    baseline, meta = predict_document(
        features,
        ModeFlags(),
        params,
        static_keywords=[],
        global_frequency={},
    )
    assert baseline == "pose estimation"
    assert meta["selection"] == "baseline_aggregator"
    kb_spec = core_mode_specs()[3]
    prediction, kb_meta = predict_document(
        features,
        kb_spec.flags,
        params,
        static_keywords=[],
        global_frequency={},
    )
    assert prediction == "pose estimation"
    assert kb_meta["selection"] == "select_clustered"


def test_launch_script_exists_and_points_at_isolated_output() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "keybert_qa_integration"
        / "launch_background.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "run.py" in text
    assert "artifacts/keybert_qa_integration" in text
    assert "nohup" in text
    assert "untie_venv/bin/python" in text
    after_out = text.split("OUT_DIR=", 1)[1][:500]
    assert "hierarchical_topic_optimized" not in after_out
    assert "--run-training" not in text
