from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.qa_chunk_selection_v6.contract import GATE_NUMERIC, feature_columns
from experiments.qa_chunk_selection_v6.features import (
    alternative_rows, assert_runtime_gold_free, canonical_document_id,
    enrich_question, evidence_pair_rows,
)
from experiments.qa_chunk_selection_v6.io_guard import assert_v6_write
from experiments.qa_chunk_selection_v6.models import (
    _question_choices, fold_for, metrics, normalize_training_weights,
    select_switch,
)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    label: str
    primary_class: str
    char_start: int
    char_end: int


class Item:
    document_id = "doc"
    question_id = "q"
    gold_answers = ("right answer",)
    chunks = (
        Chunk(0, "wrong", "hard_negative", "hard_negative", 0, 5),
        Chunk(1, "right answer", "positive", "qa_positive", 6, 18),
        Chunk(2, "other", "easy_negative", "easy_negative", 19, 24),
    )
    def chunk_by_id(self, cid): return next(c for c in self.chunks if c.chunk_id == int(cid))


def source_rows(equal=False):
    scores = [1.0, 1.0, 1.0] if equal else [3.0, 2.0, 1.0]
    answers = ["wrong", "right answer", "other"]
    tiers = ["insufficient_evidence", "sufficient_positive", "insufficient_evidence"]
    return [{"question_id":"q", "document_id":"doc", "aspect_id":"method", "seed":42,
             "chunk_id":i, "b0_rank":i+1, "b0_score_raw":scores[i], "q_answer_ranker_raw":scores[2-i],
             "source_support_norm":float(i), "qa_answer":answers[i], "label_tier":tiers[i], "candidate_count":3}
            for i in range(3)]


def test_historical_write_guard_and_v6_namespace():
    with pytest.raises(PermissionError):
        assert_v6_write(Path("artifacts/qa_chunk_selection_v5/forbidden.json"))
    path = assert_v6_write(Path("artifacts/qa_chunk_selection_v6_universal/test/write.json"))
    assert "qa_chunk_selection_v6_universal" in str(path)


def test_flat_scores_k_and_runtime_gold_separation():
    runtime, target = enrich_question("russerrc", source_rows(equal=True), Item(), outer_fold=0, seed=42, meta_source=True)
    assert all(row["flat_score_vector"] == 1 for row in runtime)
    assert all(row["b0_score_norm"] == 0 for row in runtime)
    assert runtime[0]["has_rank3"] == 1
    for row in runtime: assert_runtime_gold_free(row)
    alternatives = alternative_rows(runtime, target, variant="S2")
    assert alternatives[0]["target"] == 1
    assert abs(sum(row["weight"] for row in alternatives) - 1.0) < 1e-12


def test_s1_mirrored_contract_stays_balanced_with_interactions():
    runtime, target = enrich_question("russerrc", source_rows(), Item(), outer_fold=0, seed=42, meta_source=True)
    pairs = evidence_pair_rows(runtime, target)
    assert {row["target"] for row in pairs} == {0, 1}
    assert abs(sum(row["weight"] for row in pairs) - 1.0) < 1e-12
    assert all(name in pairs[0] for name in feature_columns("S1"))


def test_meta_folds_are_document_grouped_and_seed_specific():
    assert fold_for("same-doc", 42, 4) == fold_for("same-doc", 42, 4)
    assignments = {(seed, fold_for(f"doc-{i}", seed, 4)) for seed in (42,43,44) for i in range(20)}
    assert len(assignments) > 4


def test_scier_cells_share_canonical_lineage():
    assert canonical_document_id("scier_384", "scier:train:123") == canonical_document_id("scier_128", "scier:train:123")


def test_question_lineage_and_family_weighting():
    rows=[]
    for cell,family,lineage in (("scier_384","scier","L"),("scier_128","scier","L"),("russerrc","russerrc","R")):
        for i in range(2): rows.append({"dataset_cell":cell,"corpus_family":family,"question_id":cell,
                                        "question_lineage_id":lineage,"weight":0.5})
    normalize_training_weights(rows,pooled=True)
    totals={family:sum(r["weight"] for r in rows if r["corpus_family"]==family) for family in {r["corpus_family"] for r in rows}}
    assert totals["scier"] == pytest.approx(totals["russerrc"])
    assert sum(r["weight"] for r in rows if r["question_lineage_id"]=="L") == pytest.approx(1.0)


def test_safe_switch_uses_strict_threshold_and_safety_grid():
    base={"question_id":"q","candidate_f1":100.0,"baseline_f1":0.0,"candidate_tier":"sufficient_positive",
          "baseline_tier":"insufficient_evidence","candidate_hn":0,"baseline_hn":1,"effective_k":2,"chunk_id":1}
    choices=_question_choices([base],[0.5],0.5)
    assert choices[0]["switch"] is False
    tau, chosen, trials=select_switch([base],[0.9])
    assert tau in {0.5,0.55,0.6,0.65,0.7,0.75,0.8}
    assert chosen["token_f1"] == 100.0


def test_pooled_contract_does_not_expose_dataset_id():
    assert "dataset_cell" not in feature_columns("S4")
    assert "corpus_family" not in feature_columns("S4")
    assert "dataset_cell" not in GATE_NUMERIC


def test_launcher_resolves_project_root_and_venv():
    launcher = Path('experiments/qa_chunk_selection_v6/launch_background.sh').read_text()
    assert ')/../.." && pwd)' in launcher
    assert ')/../../.." && pwd)' not in launcher
    assert 'logs/qa_chunk_selection_v6' in launcher
    assert 'untie_venv/bin/python3' in launcher
