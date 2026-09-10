from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.v2.schemas import assert_no_gold_fields, to_runtime
from experiments.qa_chunk_selection.v3.attention import smoke_signature, synthetic_attention, token_masks_from_encoding
from experiments.qa_chunk_selection.v3.config import DEFAULT_STAGES_V3, parse_stages
from experiments.qa_chunk_selection.v3.contract import E0_ARTIFACT_ROOT, V2_ARTIFACT_ROOT, V3_ARTIFACT_ROOT, experiment_contract_v3
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v3.freeze import assert_eval_split_allowed
from experiments.qa_chunk_selection.v3.io_util import PROJECT_ROOT, assert_isolated_output
from experiments.qa_chunk_selection.v3.metrics_v3 import enrich_v3, summarize_v3
from experiments.qa_chunk_selection.v3.overlay import sample_annotation_queue, validate_overlay_row
from experiments.qa_chunk_selection.v3.roles import original_positives, train_role, v3_target_positives
from experiments.qa_chunk_selection.v3.selection import select_among
from experiments.qa_chunk_selection.v3.span_ranker import label_candidate
from experiments.qa_chunk_selection.v3.split import build_cv_splits, cyclic_roles, records_for_role


def _chunk(chunk_id: int, text: str, label: str, **kwargs) -> ChunkRecord:
    payload = dict(
        chunk_id=chunk_id,
        text=text,
        label=label,
        primary_class={
            "positive": "robust_positive",
            "hard_negative": "hard_negative",
            "easy_negative": "easy_negative",
            "ambiguous": "gold_supporting_robust_negative",
        }[label],
        position=chunk_id,
        section_id="intro",
        section_title="Introduction",
        char_start=kwargs.pop("char_start", chunk_id * 40),
        char_end=kwargs.pop("char_end", chunk_id * 40 + len(text)),
        token_start=chunk_id,
        token_end=chunk_id + 4,
        gold_supporting=label == "positive",
        previous_chunk_id=None if chunk_id == 0 else chunk_id - 1,
        next_chunk_id=chunk_id + 1,
    )
    payload.update(kwargs)
    return ChunkRecord(**payload)


def _instance(*chunks: ChunkRecord, document_id: str = "doc-a", split: str = "train") -> InstanceRecord:
    gold = "semantic segmentation"
    return InstanceRecord(
        document_id=document_id,
        question_id=f"{document_id}::task",
        aspect="task",
        question="What task is addressed in the paper?",
        gold_answers=(gold,),
        split=split,
        chunks=chunks,
        gold_spans=(
            GoldSpan(chunk_id=chunks[0].chunk_id, start_char=0, end_char=min(len(chunks[0].text), len(gold)), text=gold)
            if chunks
            else ()
        ),
    )


def test_v3_contract_isolates_e0_v2_and_production() -> None:
    contract = experiment_contract_v3()
    assert contract["e0_artifacts"] == E0_ARTIFACT_ROOT
    assert contract["v2_artifacts"] == V2_ARTIFACT_ROOT
    assert contract["v3_artifacts"] == V3_ARTIFACT_ROOT
    assert "never" in str(contract["gold_at_inference"]).lower()
    assert contract["d3_default"] is False


def test_v3_refuses_e0_v2_and_phase1_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v2" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "untie" / "x")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v3")
    assert allowed.name == "qa_chunk_selection_v3"


def test_internal_test_locked_without_freeze(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        assert_eval_split_allowed("internal_test", allow_internal_test=False, output_dir=tmp_path)
    with pytest.raises(PermissionError):
        assert_eval_split_allowed("internal_test", allow_internal_test=True, output_dir=tmp_path)


def test_runtime_has_no_gold() -> None:
    inst = _instance(_chunk(0, "the task is semantic segmentation", "positive"), _chunk(1, "hn", "hard_negative"))
    payload = to_runtime(inst).to_dict()
    assert_no_gold_fields(payload)


def test_cv_splits_no_overlap_and_cyclic_roles() -> None:
    records = synthetic_records(n_docs=10)
    payload = build_cv_splits(records, n_folds=5, seed=42)
    assigned = payload["assignment"]
    assert set(assigned.values()) <= set(range(5))
    roles = cyclic_roles(5)
    assert roles[0]["eval"] == [4]
    assert roles[0]["calibration"] == [3]
    for run in payload["runs"]:
        train = set(run["train"])
        cal = set(run["calibration"])
        ev = set(run["eval"])
        assert not (train & cal)
        assert not (train & ev)
        assert not (cal & ev)
        eval_docs = {doc for doc, fold in assigned.items() if fold in ev}
        train_docs = {doc for doc, fold in assigned.items() if fold in train}
        assert eval_docs.isdisjoint(train_docs)
        eval_recs = records_for_role(records, payload, run=run["run"], role="eval")
        train_recs = records_for_role(records, payload, run=run["run"], role="train")
        assert {item.document_id for item in eval_recs}.isdisjoint({item.document_id for item in train_recs})


def test_overlay_does_not_rewrite_source_label() -> None:
    gold = _instance(_chunk(0, "pos", "positive"), _chunk(1, "amb", "ambiguous"))
    row = {
        "chunk_id": 1,
        "source_label": "ambiguous",
        "v3_target_status": "target_mismatch",
        "v3_gold_relation": "no_gold_match_expected",
    }
    validate_overlay_row(row, gold)
    with pytest.raises(ValueError):
        validate_overlay_row({**row, "source_label": "hard_negative"}, gold)


def test_p_v3_target_requires_allowed_gold_relation() -> None:
    gold = _instance(_chunk(0, "pos", "positive"), _chunk(1, "amb", "ambiguous"))
    overlay = {
        (gold.question_id, 1): {
            "question_id": gold.question_id,
            "chunk_id": 1,
            "v3_target_status": "target_evidence",
            "v3_gold_relation": "new_valid_value_candidate",
        }
    }
    assert [c.chunk_id for c in original_positives(gold)] == [0]
    assert [c.chunk_id for c in v3_target_positives(gold, overlay)] == [0]
    overlay[(gold.question_id, 1)]["v3_gold_relation"] = "matches_existing_gold"
    assert [c.chunk_id for c in v3_target_positives(gold, overlay)] == [0, 1]


def test_partial_support_masked_from_train() -> None:
    gold = _instance(_chunk(0, "pos", "positive"), _chunk(1, "amb", "ambiguous"))
    overlay = {
        (gold.question_id, 1): {"question_id": gold.question_id, "chunk_id": 1, "v3_target_status": "partial_support"}
    }
    assert train_role(gold, gold.chunks[1], overlay, {}) is None


def test_target_mismatch_metrics_use_coverage() -> None:
    gold = _instance(
        _chunk(0, "semantic segmentation", "positive"),
        _chunk(1, "other dataset semantic segmentation", "ambiguous"),
    )
    raw = {
        "document_id": gold.document_id,
        "question_id": gold.question_id,
        "ranked_ids": [1, 0],
        "selected_chunk_id": 1,
        "qa_chunk_id": 1,
        "qa_answer": "semantic segmentation",
        "scores": {0: 0.1, 1: 0.9},
    }
    overlay = {
        (gold.question_id, 1): {
            "question_id": gold.question_id,
            "chunk_id": 1,
            "v3_target_status": "target_mismatch",
        }
    }
    row = enrich_v3(raw, gold, overlay)
    assert row["ambiguous_at_1_original"] == 1.0
    assert row["target_mismatch_at_1"] == 1.0
    assert row["wrong_target_answer"] == 1.0
    labeled = label_candidate(gold, {"chunk_id": 1, "qa_answer": "semantic segmentation", "chunk_text": gold.chunks[1].text}, overlay)
    assert labeled["span_correct_target"] == 0
    assert labeled["wrong_target"] == 1
    summary = summarize_v3([row])
    assert summary["target_mismatch@1_lower_bound"] == 1.0


def test_annotator_queue_hides_model_scores() -> None:
    records = synthetic_records(n_docs=4)
    scores = {(item.question_id, chunk.chunk_id): 0.9 for item in records for chunk in item.chunks}
    payload = sample_annotation_queue(records, scores=scores, hard_n=3, repr_n=3, seed=1)
    assert payload["queue"]
    assert all("b_score" not in row for row in payload["queue"])
    assert any("b_score" in row for row in payload["metadata"])


def test_selection_not_pairwise_lexicographic() -> None:
    rows = [
        {"name": "b0", "token_f1": 20.0, "hn@1": 0.04, "pairwise_clean_hn": 0.80, "hit@1_original": 0.3, "target_evidence_hit@1_v3": 0.3, "target_status_coverage@1": 0.5},
        {"name": "b3", "token_f1": 28.0, "hn@1": 0.03, "pairwise_clean_hn": 0.70, "hit@1_original": 0.4, "target_evidence_hit@1_v3": 0.4, "target_status_coverage@1": 0.5},
        {"name": "b1", "token_f1": 22.0, "hn@1": 0.20, "pairwise_clean_hn": 0.50, "hit@1_original": 0.35, "target_evidence_hit@1_v3": 0.35, "target_status_coverage@1": 0.5},
    ]
    selected = select_among(rows, baseline_name="b0", epsilon_hn=0.02, epsilon_pairwise_hn=0.15, min_coverage=0.95)
    assert selected["selected"] == "b3"
    assert "b1" in selected["rejected"]


def test_token_masks_from_sequence_ids() -> None:
    class _Enc(dict):
        def sequence_ids(self, example: int = 0):
            return [None, 0, 0, None, 1, 1, 1, None]

    encoded = _Enc()
    encoded["offset_mapping"] = [[(0, 0), (0, 4), (5, 12), (0, 0), (0, 3), (4, 10), (11, 16), (0, 0)]]
    encoded["special_tokens_mask"] = [[1, 0, 0, 1, 0, 0, 0, 1]]
    masks = token_masks_from_encoding(encoded)
    assert int(masks.question.sum()) == 2
    assert int(masks.context.sum()) == 3
    assert int(masks.special.sum()) == 3


def test_attention_signature_length_normalized() -> None:
    sig = smoke_signature("short question", "a context with a predicted span here", seed=3)
    assert sig["gold_dependent"] is False
    assert sig["predicted_span_length"] >= 1
    assert len(sig["vector"]) > 10
    A = synthetic_attention(2, 2, 8, seed=0)
    assert np.allclose(A.sum(axis=-1), 1.0, atol=1e-6)


def test_default_stages_exclude_test_and_gated() -> None:
    assert "final-eval" not in DEFAULT_STAGES_V3
    assert "d3" not in DEFAULT_STAGES_V3
    assert "freeze-final" not in DEFAULT_STAGES_V3
    assert parse_stages("all") == DEFAULT_STAGES_V3
    assert "b0" in parse_stages("all")
