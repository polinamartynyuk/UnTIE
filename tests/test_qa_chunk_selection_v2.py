from __future__ import annotations

from pathlib import Path

import pytest

from experiments.qa_chunk_selection.schema import ChunkRecord, GoldSpan, InstanceRecord
from experiments.qa_chunk_selection.v2.adaptive_m import choose_m
from experiments.qa_chunk_selection.v2.answer_aggregation_v2 import aggregate_d5, unique_supports
from experiments.qa_chunk_selection.v2.answerability_v2 import sample_dynamic_epoch, unique_question_pairs
from experiments.qa_chunk_selection.v2.cache import cache_key, text_hash
from experiments.qa_chunk_selection.v2.config import DEFAULT_STAGES_V2, parse_stages
from experiments.qa_chunk_selection.v2.contract import E0_ARTIFACT_ROOT, experiment_contract_v2
from experiments.qa_chunk_selection.v2.evaluate_v2 import build_raw_prediction
from experiments.qa_chunk_selection.v2.freeze import assert_eval_split_allowed
from experiments.qa_chunk_selection.v2.hn_flags import hn_quality_rows
from experiments.qa_chunk_selection.v2.io_util import PROJECT_ROOT, assert_isolated_output
from experiments.qa_chunk_selection.v2.reader_oracle import refuse_oracle_for_selection
from experiments.qa_chunk_selection.v2.schemas import assert_extractive_trace, assert_no_gold_fields, source_offsets, to_runtime
from experiments.qa_chunk_selection.v2.sentence_evidence import d4_training_examples, project_assembled_span, sentence_units
from experiments.qa_chunk_selection.v2.split import apply_document_roles, split_document_ids, split_sanity
from experiments.qa_chunk_selection.v2.statistics import average_across_seeds
from experiments.qa_chunk_selection.v2.timing import timed_block


def _chunk(chunk_id: int, text: str, label: str, **kwargs) -> ChunkRecord:
    payload = dict(
        chunk_id=chunk_id,
        text=text,
        label=label,
        primary_class={
            "positive": "robust_positive",
            "hard_negative": "hard_negative",
            "easy_negative": "easy_negative",
            "ambiguous": "question_sensitive",
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


def _instance(
    *chunks: ChunkRecord,
    document_id: str = "doc-a",
    split: str = "train",
    gold: str = "semantic segmentation",
) -> InstanceRecord:
    return InstanceRecord(
        document_id=document_id,
        question_id=f"{document_id}::task",
        aspect="task",
        question="What task is addressed in the paper?",
        gold_answers=(gold,),
        split=split,
        chunks=chunks,
        gold_spans=(
            GoldSpan(
                chunk_id=chunks[0].chunk_id,
                start_char=0,
                end_char=min(len(chunks[0].text), len(gold)),
                text=gold,
            )
            if chunks
            else ()
        ),
    )


def test_v2_contract_isolates_e0_and_production() -> None:
    contract = experiment_contract_v2()
    assert "unchanged" in str(contract["production_pipeline"])
    assert contract["e0_artifacts"] == E0_ARTIFACT_ROOT
    assert contract["d1_default"] is False
    assert contract["d3_default"] is False
    assert "qa_confidence_proxy" in str(contract["qa_span_score_is"])


def test_v2_refuses_e0_and_phase1_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "qa_chunk_selection" / "oops")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "x")
    with pytest.raises(ValueError):
        assert_isolated_output(PROJECT_ROOT / "untie" / "x")
    allowed = assert_isolated_output(tmp_path / "qa_chunk_selection_v2")
    assert allowed.name == "qa_chunk_selection_v2"


def test_runtime_instance_has_no_gold_fields() -> None:
    inst = _instance(_chunk(0, "the task is semantic segmentation", "positive"), _chunk(1, "hn", "hard_negative"))
    runtime = to_runtime(inst)
    payload = runtime.to_dict()
    assert_no_gold_fields(payload)
    assert "gold_answers" not in payload
    assert "label" not in payload["chunks"][0]


def test_raw_prediction_rejects_gold() -> None:
    inst = _instance(_chunk(0, "semantic segmentation", "positive"))
    runtime = to_runtime(inst)
    row = build_raw_prediction(
        runtime,
        ranking_method="b2",
        scores=[0.9],
        ranked_ids=[0],
        qa={
            "qa_answer": "semantic segmentation",
            "qa_chunk_id": 0,
            "qa_chunk_start_char": 0,
            "qa_chunk_end_char": 22,
            "qa_span_score": 1.0,
        },
    )
    assert "gold_answers" not in row
    with pytest.raises(ValueError):
        assert_no_gold_fields({"gold_answers": ["x"]})


def test_source_offsets_reconstruct_answer() -> None:
    text = "the task is semantic segmentation here"
    local_start = text.index("semantic segmentation")
    local_end = local_start + len("semantic segmentation")
    chunk = _chunk(0, text, "positive", char_start=100, char_end=100 + len(text))
    src_start, src_end = source_offsets(chunk, local_start, local_end)
    answer = text[local_start:local_end]
    assert answer == "semantic segmentation"
    assert_extractive_trace(
        chunk_text=text,
        answer=answer,
        local_start=local_start,
        local_end=local_end,
        source_start=src_start,
        source_end=src_end,
        chunk_char_start=chunk.char_start,
    )
    assert src_start == 100 + local_start


def test_document_split_has_no_overlap() -> None:
    records = [
        _instance(_chunk(0, "pos", "positive"), document_id=f"d{i}", split="train")
        for i in range(20)
    ]
    roles = split_document_ids(records, seed=42, fractions=(0.6, 0.15, 0.15, 0.1))
    assigned = apply_document_roles(
        records,
        roles,
        official_dev=[_instance(_chunk(0, "p", "positive"), document_id="dev1", split="dev")],
    )
    summary = split_sanity(assigned)
    docs = [
        {row.document_id for row in assigned if row.split == split}
        for split in ("train", "calibration", "development", "internal_test")
    ]
    for i, left in enumerate(docs):
        for right in docs[i + 1 :]:
            assert not (left & right)
    assert summary["seen_exploratory_dev"]["n_documents"] == 1
    again = split_document_ids(records, seed=42, fractions=(0.6, 0.15, 0.15, 0.1))
    assert again == roles
    assert set(roles) == {"train", "calibration", "development", "internal_test"}
    assert all(roles[name] for name in roles)


def test_internal_test_lock(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        assert_eval_split_allowed("internal_test", allow_internal_test=False, output_dir=tmp_path)
    with pytest.raises(PermissionError):
        assert_eval_split_allowed("internal_test", allow_internal_test=True, output_dir=tmp_path)
    assert_eval_split_allowed("development", allow_internal_test=False, output_dir=tmp_path)


def test_hn_flags_do_not_relabel() -> None:
    inst = _instance(
        _chunk(0, "semantic segmentation", "positive"),
        _chunk(1, "semantic segmentation appears in hard negative", "hard_negative"),
        _chunk(2, "unrelated appendix", "hard_negative"),
    )
    rows = hn_quality_rows([inst], include_hn_gold_warning=False)
    flagged = [row for row in rows if row["chunk_id"] == 1][0]
    clean = [row for row in rows if row["chunk_id"] == 2][0]
    assert flagged["label"] == "hard_negative"
    assert flagged["hn_gold_string_warning"] is True
    assert flagged["training_eligible_hn"] is False
    assert clean["training_eligible_hn"] is True


def test_noisy_hn_excluded_from_default_b2_training() -> None:
    inst = _instance(
        _chunk(0, "semantic segmentation", "positive"),
        _chunk(1, "semantic segmentation appears in hard negative", "hard_negative"),
        _chunk(2, "unrelated appendix", "hard_negative"),
    )
    flags = {(row["question_id"], row["chunk_id"]): row for row in hn_quality_rows([inst])}
    weighted, _report = unique_question_pairs([inst], flags, ratio=(1, 3, 1))
    hn_ids = [row.pair.chunk_id for row in weighted if row.pair.kind == "hard_negative"]
    assert 1 not in hn_ids
    assert 2 in hn_ids


def test_question_balanced_weights_do_not_scale_with_negatives() -> None:
    small = _instance(
        _chunk(0, "pos", "positive"),
        _chunk(1, "hn", "hard_negative"),
        document_id="small",
    )
    large_chunks = [_chunk(0, "pos", "positive")] + [_chunk(i, f"hn{i}", "hard_negative") for i in range(1, 21)]
    large = _instance(*large_chunks, document_id="large")
    flags = {(row["question_id"], row["chunk_id"]): row for row in hn_quality_rows([small, large])}
    weighted, report = unique_question_pairs([small, large], flags, ratio=(1, 3, 1))
    by_q: dict[str, float] = {}
    for row in weighted:
        by_q[row.pair.question_id] = by_q.get(row.pair.question_id, 0.0) + row.weight
    vals = list(by_q.values())
    assert max(vals) / min(vals) < 1.01
    assert report["question_weight_spread"] < 1.01


def test_dynamic_negative_sampling_is_deterministic_and_prefers_unique_hn() -> None:
    chunks = [_chunk(0, "pos", "positive")] + [_chunk(i, f"hn{i}", "hard_negative") for i in range(1, 8)]
    inst = _instance(*chunks, document_id="dyn")
    flags = {(row["question_id"], row["chunk_id"]): row for row in hn_quality_rows([inst])}
    first, _usage = sample_dynamic_epoch([inst], flags, ratio=(1, 3, 1), seed=42, epoch=1)
    second, _usage2 = sample_dynamic_epoch([inst], flags, ratio=(1, 3, 1), seed=42, epoch=1)
    assert [(row.chunk_id, row.kind) for row in first] == [(row.chunk_id, row.kind) for row in second]
    hn_ids = [row.chunk_id for row in first if row.kind == "hard_negative"]
    assert len(hn_ids) == len(set(hn_ids))


def test_cache_invalidates_on_text_or_model_change() -> None:
    left = cache_key(chunk_text_hash=text_hash("a"), question_text_hash=text_hash("q"), model="m1")
    right = cache_key(chunk_text_hash=text_hash("b"), question_text_hash=text_hash("q"), model="m1")
    other = cache_key(chunk_text_hash=text_hash("a"), question_text_hash=text_hash("q"), model="m2")
    q_changed = cache_key(chunk_text_hash=text_hash("a"), question_text_hash=text_hash("q2"), model="m1")
    assert left != right
    assert left != other
    assert left != q_changed


def test_adaptive_m_bounds() -> None:
    assert choose_m([0.9], tau_conf=0.1, tau_margin=0.0, m_uncertain=5, max_m=8) == 1
    m = choose_m([0.51, 0.49, 0.1, 0.1], tau_conf=0.9, tau_margin=0.5, m_uncertain=3, max_m=8)
    assert m in {1, 3}
    assert m <= 4
    again = choose_m([0.51, 0.49, 0.1, 0.1], tau_conf=0.9, tau_margin=0.5, m_uncertain=3, max_m=8)
    assert again == m


def test_d4_cross_sentence_gold_is_not_silently_positive() -> None:
    text = "Hello world. Next sentence has gold."
    chunk = _chunk(0, text, "positive", char_start=0, char_end=len(text))
    inst = InstanceRecord(
        document_id="d",
        question_id="d::task",
        aspect="task",
        question="q",
        gold_answers=("world. Next",),
        split="train",
        chunks=(chunk,),
        gold_spans=(GoldSpan(chunk_id=0, start_char=6, end_char=18, text="world. Next"),),
    )
    units = sentence_units(inst)
    assert any(unit["cross_sentence_gold"] for unit in units)
    flags = {(inst.question_id, 0): {"training_eligible_hn": False}}
    examples = d4_training_examples(inst, flags)
    assert all(not unit.get("cross_sentence_gold") for unit in examples)


def test_d4_assembled_span_roundtrip() -> None:
    mapping = [
        {
            "assembled_start": 0,
            "assembled_end": 5,
            "source_chunk_id": 1,
            "source_start_char": 10,
            "source_end_char": 15,
        }
    ]
    proj = project_assembled_span("hello", mapping, 0, 5)
    assert proj["source_start_char"] == 10
    assert proj["text"] == "hello"


def test_d5_dedup_overlap_and_no_raw_logit_logsumexp() -> None:
    spans = [
        {
            "answer_text": "cnn",
            "text": "cnn",
            "source_start_char": 0,
            "source_end_char": 3,
            "chunk_id": 0,
            "candidate_score": 0.8,
            "qa_span_score": 12.0,
        },
        {
            "answer_text": "cnn",
            "text": "cnn",
            "source_start_char": 0,
            "source_end_char": 3,
            "chunk_id": 1,
            "candidate_score": 0.7,
            "qa_span_score": 40.0,
        },
        {
            "answer_text": "lstm",
            "text": "lstm",
            "source_start_char": 50,
            "source_end_char": 54,
            "chunk_id": 2,
            "candidate_score": 0.2,
            "qa_span_score": 1.0,
        },
    ]
    unique = unique_supports(spans)
    assert len(unique) == 2
    d5b = aggregate_d5(spans, mode="D5b", score_field="candidate_score")
    assert d5b["text"] == "cnn"
    assert d5b["n_unique_regions"] == 1
    assert d5b["score_field"] == "candidate_score"


def test_oracle_cannot_be_used_for_selection() -> None:
    with pytest.raises(RuntimeError):
        refuse_oracle_for_selection()


def test_metric_scales_documented() -> None:
    from experiments.qa_chunk_selection.metrics import qa_row

    row = qa_row("semantic segmentation", ["semantic segmentation"])
    assert 0.0 <= row["char_f1"] <= 1.0
    assert 0.0 <= row["rouge_l_f1"] <= 1.0
    assert row["token_f1"] >= 50.0


def test_seed_averaging_is_per_question_not_pseudo_replication() -> None:
    rows = {
        42: [{"document_id": "d", "question_id": "q", "token_f1": 10.0}],
        43: [{"document_id": "d", "question_id": "q", "token_f1": 30.0}],
    }
    merged = average_across_seeds(rows, "token_f1")
    assert len(merged) == 1
    assert merged[0]["token_f1"] == pytest.approx(20.0)


def test_default_stages_exclude_d1_d3_d4_d5_and_final() -> None:
    stages = parse_stages("all")
    assert stages == DEFAULT_STAGES_V2
    assert "d1" not in stages
    assert "d3" not in stages
    assert "d4" not in stages
    assert "d5" not in stages
    assert "final-eval" not in stages
    assert "b2-train" in stages
    assert "e0-regression" in stages


def test_e0_parser_still_defaults_to_e0_output() -> None:
    from experiments.qa_chunk_selection.run import build_parser

    args = build_parser().parse_args([])
    assert str(args.output_dir).endswith("qa_chunk_selection")
    assert "qa_chunk_selection_v2" not in str(args.output_dir)


def test_v2_parser_defaults_to_v2_output() -> None:
    from experiments.qa_chunk_selection.v2.run import build_parser

    args = build_parser().parse_args([])
    assert str(args.output_dir).endswith("qa_chunk_selection_v2")
    assert Path(args.e0_artifact_dir).name == "qa_chunk_selection"


def test_freeze_hash_ignores_cli_stages() -> None:
    from experiments.qa_chunk_selection.v2.config import ExperimentConfigV2, parse_stages

    left = ExperimentConfigV2(stages=parse_stages("all"))
    right = ExperimentConfigV2(stages=parse_stages("freeze-final"), allow_internal_test=True, device="cuda:0")
    assert left.config_hash() == right.config_hash()


def test_timed_block_records_elapsed_and_peak() -> None:
    with timed_block() as stats:
        total = sum(range(1000))
        del total
    assert stats["elapsed_s"] >= 0.0
    assert "peak_gpu_mb" in stats


def test_e0_regression_r1_if_artifacts_exist() -> None:
    from experiments.qa_chunk_selection.v2.regression import E0_COMPARE, regression_r1, regression_r2_compare

    e0 = PROJECT_ROOT / "artifacts" / "qa_chunk_selection"
    if not (e0 / "runs" / "b" / "42" / "dev" / "metrics.json").exists():
        pytest.skip("E0 artifacts are not present")
    result = regression_r1(e0, atol=1e-6, rtol=1e-5)
    assert result["ok"], result["mismatches"][:5]
    missing = regression_r2_compare({}, e0, atol=1e-6, rtol=1e-5)
    assert not missing["ok"]
    assert any(row.get("method") == "d2" and row.get("error") == "missing replay summary" for row in missing["mismatches"])
    assert set(E0_COMPARE) <= {"a", "b", "c", "d1", "d2"}
