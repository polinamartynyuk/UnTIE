from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.keybert_topic_clustering.ce_contract import ce_contract
from experiments.keybert_topic_clustering.documents import (
    PaperDocument,
    parse_title_abstract_headings,
)
from experiments.keybert_topic_clustering.downstream import select_frequency_cluster
from experiments.keybert_topic_clustering.io_util import assert_isolated_output
from experiments.keybert_topic_clustering.representations import hybrid_matrix
from untie.domain import Answer, Sentence, TextChunk


def test_ce_contract_maps_to_extractive_qa_not_minilm() -> None:
    contract = ce_contract()
    assert contract["actual_class"] == "untie.models.ExtractiveQuestionAnswerer"
    assert "MiniLM" in str(contract["not_used_as_ce"])


def test_parses_scirex_title_and_abstract_from_body() -> None:
    title, abstract, headings = parse_title_abstract_headings(
        "document: Bi - Directional Attention Flow for Machine Comprehension "
        "section: Abstract Machine comprehension (MC) requires modeling the query. "
        "section: Introduction Later LDA topic modeling appears."
    )
    assert title.startswith("Bi - Directional Attention Flow")
    assert "Machine comprehension (MC)" in abstract
    assert "LDA topic modeling" not in abstract
    assert "Introduction" in headings


def test_parses_opening_sentence_when_abstract_marker_is_missing() -> None:
    title, abstract, _headings = parse_title_abstract_headings(
        "document: V - Net: Fully Convolutional Neural Networks for Volumetric "
        "Medical Image Segmentation Convolutional Neural Networks (CNNs) have "
        "been recently employed to solve problems. This paper focuses on volumes. "
        "Later we mention LDA topics."
    )
    assert "V - Net" in title
    assert "This paper focuses on volumes" in abstract
    assert "LDA topics" not in abstract


def test_hybrid_representation_is_weighted_and_l2_normalized() -> None:
    document = np.asarray([[1.0, 0.0]], dtype=np.float32)
    keybert = np.asarray([[0.0, 1.0]], dtype=np.float32)
    mixed = hybrid_matrix(document, keybert, 0.75)
    assert mixed.shape == (1, 2)
    assert pytest.approx(float(np.linalg.norm(mixed[0])), rel=1e-5) == 1.0
    assert mixed[0, 0] > mixed[0, 1]


def test_output_dir_refuses_production_hierarchical_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="refuses to write"):
        assert_isolated_output(
            root / "artifacts" / "hierarchical_topic_optimized" / "en" / "evaluation",
            root,
        )
    allowed = assert_isolated_output(
        root / "artifacts" / "keybert_topic_clustering" / "en",
        root,
    )
    assert "keybert_topic_clustering" in str(allowed)


def _chunk(index: int, text: str) -> TextChunk:
    return TextChunk(
        sentences=(Sentence(text=text, number=index, tokens=tuple(text.split())),),
        text=text,
        token_count=len(text.split()),
    )


def test_d0_selects_the_largest_answer_cluster() -> None:
    answers = [
        Answer("pose estimation", _chunk(0, "a"), 0.2),
        Answer("pose estimation", _chunk(1, "b"), 0.1),
        Answer("classification", _chunk(2, "c"), 0.9),
    ]
    similarity = np.asarray(
        [
            [1.0, 0.95, 0.1],
            [0.95, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    selected = select_frequency_cluster(answers, similarity, threshold=0.75)
    assert selected is not None
    assert selected.text == "pose estimation"


def test_launch_script_exists_and_points_at_isolated_output() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "keybert_topic_clustering"
        / "launch_background.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "run.py" in text
    assert "artifacts/keybert_topic_clustering" in text
    assert "hierarchical_topic_optimized" not in text.split("OUT_DIR")[1][:400]
    assert "nohup" in text
