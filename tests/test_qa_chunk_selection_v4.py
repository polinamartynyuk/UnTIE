from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import synthetic_records
from experiments.qa_chunk_selection.v4.clusters import overlapping_ids
from experiments.qa_chunk_selection.v4.config import ExperimentConfigV4, parse_stages
from experiments.qa_chunk_selection.v4.contract import experiment_contract_v4
from experiments.qa_chunk_selection.v4.heads import rank_by_scores, softmax
from experiments.qa_chunk_selection.v4.io_guard import assert_isolated_output, refuse_completed_output
from experiments.qa_chunk_selection.v4.keybert_features import compatibility_features, lexical_keyphrases
from experiments.qa_chunk_selection.v4.pipeline import PipelineV4
from experiments.qa_chunk_selection.v4.run import main as v4_main
from experiments.qa_chunk_selection.v4.v4_ownership import pair_documents
from experiments.qa_chunk_selection.v4.v5_calibrate import collect_training
from experiments.qa_chunk_selection.v4.v7_reconsider import build_v7_pairs


PROTECTED = (
    "artifacts/qa_chunk_selection",
    "artifacts/qa_chunk_selection_v2",
    "artifacts/qa_chunk_selection_v3",
    "artifacts/qa_chunk_selection_v3_1_prep",
    "artifacts/qa_chunk_selection_v3_1_auto",
    "artifacts/qa_chunk_selection_v3_1_auto_v2",
    "artifacts/qa_chunk_selection_v3_1",
    "artifacts/qa_chunk_selection_v3_d45",
)


def _item_with_overlap():
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    first = item.chunks[0]
    chunks = []
    for chunk in item.chunks:
        if chunk.chunk_id == 1:
            chunks.append(replace(chunk, char_start=first.char_start, char_end=first.char_end))
        else:
            chunks.append(chunk)
    return replace(item, chunks=tuple(chunks))


def test_v4_contract_keeps_prior_trees_readonly() -> None:
    contract = experiment_contract_v4()
    assert contract["v4_artifacts"] == "artifacts/qa_chunk_selection_v4"
    assert "never overwrite" in str(contract["prior_trees"])
    assert contract["internal_test"] == "locked"
    assert contract["b0_live"].startswith("historical V3-B0 max5")
    assert "qa_silver" in contract


def test_v4_refuses_protected_writes(tmp_path: Path) -> None:
    for relative in PROTECTED:
        with pytest.raises(ValueError):
            assert_isolated_output(PROJECT_ROOT / relative / "oops")
    allowed_tmp = assert_isolated_output(tmp_path / "qa_chunk_selection_v4")
    assert allowed_tmp.name == "qa_chunk_selection_v4"
    from experiments.qa_chunk_selection.v4.contract import PROTECTED_WRITE_ROOTS, V4_ARTIFACT_ROOT

    assert V4_ARTIFACT_ROOT not in PROTECTED_WRITE_ROOTS


def test_v4_cli_refuses_prior_and_completed(tmp_path: Path) -> None:
    assert v4_main(["--output-dir", str(PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v3"), "--smoke"]) == 2
    completed = tmp_path / "qa_chunk_selection_v4"
    (completed / "analysis").mkdir(parents=True)
    (completed / "analysis" / "compare.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        refuse_completed_output(completed, allow=False)
    assert v4_main(["--output-dir", str(completed), "--smoke"]) == 2


def test_encoder_device_auto_is_never_passed_through() -> None:
    from experiments.qa_chunk_selection.v4.keybert_features import KeyphraseExtractor, resolve_encoder_device

    resolved = resolve_encoder_device("auto")
    assert resolved != "auto"
    assert resolved.startswith("cpu") or resolved.startswith("cuda")
    extractor = KeyphraseExtractor(
        backend="unused",
        top_n=2,
        ngram_low=1,
        ngram_high=2,
        use_mmr=False,
        smoke=True,
        device="auto",
    )
    assert extractor.device == "cpu"


def test_losco_excludes_overlap_loco_excludes_self_only() -> None:
    item = _item_with_overlap()
    losco = overlapping_ids(item, 0, min_overlap=0.6)
    loco = overlapping_ids(item, 0, min_overlap=1.1)
    assert 0 in losco and 1 in losco
    assert loco == {0}
    phrases = {int(chunk.chunk_id): lexical_keyphrases(chunk.text, top_n=8, ngram_high=2) for chunk in item.chunks}
    embeddings = {row["phrase"]: [float(len(row["phrase"]))] for rows in phrases.values() for row in rows}
    idf = {phrase: 1.0 for phrase in embeddings}
    losco_feats = compatibility_features(item, 0, phrases, embeddings, idf, min_overlap=0.6, mode="losco")
    loco_feats = compatibility_features(item, 0, phrases, embeddings, idf, min_overlap=0.6, mode="loco")
    assert losco_feats["n_excluded"] > loco_feats["n_excluded"]


def test_chunk_prediction_repairs_historical_b0_qa() -> None:
    from experiments.qa_chunk_selection.v4.evaluate import chunk_prediction, coerce_extractive_qa, qa_fields_from_row

    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    chunk = item.positives[0]
    answer = item.gold_answers[0]
    start = chunk.text.find(answer)
    assert start >= 0
    scores = {int(row.chunk_id): float(4 - row.chunk_id) for row in item.chunks}
    slim = {"qa_answer": answer, "qa_chunk_id": chunk.chunk_id}
    pred = chunk_prediction(item, ranking_method="B0-live", scores=scores, qa=slim, skip_bertscore=True)
    assert pred["qa_answer"] == answer
    leaked = qa_fields_from_row({"qa_answer": answer, "gold_answers": ["nope"], "qa_chunk_id": chunk.chunk_id})
    assert "gold_answers" not in leaked
    repaired, leftover = coerce_extractive_qa(
        item,
        {
            "qa_answer": answer,
            "qa_chunk_id": chunk.chunk_id,
            "qa_chunk_start_char": start,
            "qa_chunk_end_char": start + len(answer),
            "gold_answers": ["leak"],
        },
        scores=scores,
    )
    assert leftover == ""
    assert repaired["qa_chunk_start_char"] == start
    assert "gold_answers" not in repaired


def test_softmax_does_not_change_b0_order() -> None:
    scores = [2.4, 1.1, 0.3, -0.8]
    ids = [3, 1, 7, 2]
    assert rank_by_scores(ids, scores) == rank_by_scores(ids, softmax(scores))


def test_synthetic_pairing_stays_inside_split() -> None:
    records = [item for item in synthetic_records(n_docs=8) if item.positives]
    profiles = {item.document_id: [1.0, 0.0] for item in records}
    pairs = pair_documents(records, profiles, seed=42)
    assert pairs
    assert all(left.split == right.split for left, right, _sim in pairs)
    assert all(left.document_id != right.document_id for left, right, _sim in pairs)
    assert all(left.aspect == right.aspect for left, right, _sim in pairs)


def test_ambiguous_never_used_as_strong_negative() -> None:
    item = next(record for record in synthetic_records(n_docs=1) if record.positives)
    amb_ids = {int(chunk.chunk_id) for chunk in item.chunks if chunk.label == "ambiguous"}
    assert amb_ids
    b0 = {
        item.question_id: {
            "ranked_ids": [int(chunk.chunk_id) for chunk in item.chunks],
            "scores": {str(chunk.chunk_id): float(4 - chunk.chunk_id) for chunk in item.chunks},
        }
    }
    qa_store = {(item.question_id, int(chunk.chunk_id)): {"qa_raw_score": 0.1} for chunk in item.chunks}
    matrix, labels = collect_training([item], b0, qa_store, {}, k=4, easy_cap=4)
    assert labels
    v7 = build_v7_pairs([item], b0, {}, [], k=4, easy_cap=4, use_anchor=False)
    assert all(int(pair.pair.chunk_id) not in amb_ids for pair in v7)
    labeled_chunks = []
    easy_used = 0
    for cid in [int(chunk.chunk_id) for chunk in item.chunks]:
        chunk = item.chunk_by_id(cid)
        if chunk.label == "positive":
            labeled_chunks.append((cid, 1))
        elif chunk.label in {"hard_negative", "easy_negative"}:
            labeled_chunks.append((cid, 0))
            easy_used += int(chunk.label == "easy_negative")
    assert all(cid not in amb_ids for cid, _label in labeled_chunks)
    assert len(matrix) == len(labeled_chunks) or len(matrix) <= len(labeled_chunks)


def test_normalization_and_features_do_not_use_gold_at_rank_time() -> None:
    from experiments.qa_chunk_selection.v4 import answers, keybert_features, qa_shortlist, v1_keybert

    for module in (answers, keybert_features, qa_shortlist, v1_keybert):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "gold_answers" not in source
        assert "gold_spans" not in source


def test_parse_stages_aliases() -> None:
    assert parse_stages("phase1")[0] == "init"
    assert "v1" in parse_stages("phase1")
    assert "v7" not in parse_stages("phase1")
    assert "v7" in parse_stages("full")
    with pytest.raises(ValueError):
        parse_stages("not-a-stage")


def test_v4_smoke_phase1(tmp_path: Path) -> None:
    config = ExperimentConfigV4(
        output_dir=tmp_path / "qa_chunk_selection_v4",
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=parse_stages("phase1"),
        smoke=True,
        n_bootstrap=20,
    )
    assert PipelineV4(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "progress.jsonl").exists()
    assert (root / "contract.json").exists()
    assert (root / "analysis" / "compare.json").exists()
    assert (root / "reports" / "results_v4.md").exists()
    assert (root / "analysis" / "V1-A3" / "metrics.json").exists()
    assert (root / "analysis" / "V2-A2" / "metrics.json").exists()
    progress = (root / "progress.jsonl").read_text(encoding="utf-8")
    assert "STAGE" in progress or "stage" in progress.lower()


def test_v4_smoke_full(tmp_path: Path) -> None:
    config = ExperimentConfigV4(
        output_dir=tmp_path / "qa_chunk_selection_v4_full",
        v3_artifact_dir=tmp_path / "missing_v3",
        stages=parse_stages("full"),
        smoke=True,
        n_bootstrap=20,
        b0_epochs=1,
        b0_patience=1,
    )
    assert PipelineV4(config).run(list(config.stages)) == 0
    root = config.output_dir
    assert (root / "analysis" / "v4_diagnostic.json").exists()
    assert (root / "analysis" / "V5-A1" / "metrics.json").exists()
    assert (root / "analysis" / "V6" / "metrics.json").exists()
    assert (root / "analysis" / "V7" / "metrics.json").exists()
    assert (root / "reports" / "results_v4.md").exists()
