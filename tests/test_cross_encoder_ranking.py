from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from untie.cross_encoder_ranking import (
    MixWeights,
    RankedDocumentPanel,
    collect_ranked_panel,
    cross_encoder_chunk_scores,
    fit_mix_weights,
    keyword_chunk_scores,
    mixed_chunk_scores,
    predict_from_panel,
    ranking_query,
)
from untie.domain import Answer, Sentence, TextChunk
from untie.ranking import WeightedKeyword
from untie.topics import TopicDocument


class Encoder:
    def encode(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("frost" in lowered or "pressure" in lowered),
                    float("lda" in lowered or "topic" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=float)


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for _query, passage in pairs:
            lowered = passage.lower()
            if "frost" in lowered:
                scores.append(2.5)
            elif "lda" in lowered:
                scores.append(-1.5)
            else:
                scores.append(0.0)
        return scores


class FakeAnswerer:
    def __call__(self, *, question: str, context: str) -> dict:
        del question
        if "frost pressure" in context:
            answer = "frost pressure"
        elif "LDA" in context:
            answer = "LDA"
        else:
            answer = context.split(".")[0]
        start = max(0, context.find(answer))
        return {
            "answer": answer,
            "score": 0.9,
            "start": start,
            "end": start + len(answer),
        }


class StubProcessor:
    def __init__(self, chunks: list[TextChunk]) -> None:
        self.chunks = chunks

    def process(self, text: str) -> list[TextChunk]:
        del text
        return list(self.chunks)


def _chunk(text: str, number: int) -> TextChunk:
    tokens = tuple(text.replace(".", "").split())
    return TextChunk((Sentence(text, number, tokens),), text, len(tokens))


def _panel(
    *,
    doc_id: str = "dev-1",
    split: str = "dev",
    gold: str = "frost pressure",
) -> RankedDocumentPanel:
    gold_chunk = _chunk("The paper solves frost pressure prediction.", 0)
    distractor = _chunk("Appendix discusses LDA topic modeling.", 1)
    return RankedDocumentPanel(
        doc_id=doc_id,
        split=split,
        question="Which task was solved?",
        query="Soil frost. Which task was solved?",
        references=(gold,),
        chunks=(gold_chunk, distractor),
        answers=(
            Answer(text="frost pressure", chunk=gold_chunk, confidence=0.9),
            Answer(text="LDA", chunk=distractor, confidence=0.9),
        ),
        cross_encoder_scores=(2.5, -1.5),
        keyword_scores=(0.0, 1.0),
    )


def _exact_char_f1(prediction: str | None, references) -> float:
    gold = str(references[0]) if references else ""
    return 1.0 if (prediction or "") == gold else 0.0


def test_ranking_query_prefers_title_and_question() -> None:
    document = TopicDocument(
        "d1",
        title="Soil frost pressure",
        abstract="Unused abstract.",
        split="test",
    )
    assert ranking_query(document, "Which task was solved?") == (
        "Soil frost pressure. Which task was solved?"
    )


def test_ranking_query_uses_body_heading_when_title_and_abstract_are_empty() -> None:
    document = TopicDocument(
        "d1",
        title="",
        abstract="",
        text=(
            "document: Bi - Directional Attention Flow for Machine Comprehension "
            "section: Abstract Machine comprehension requires modeling the query. "
            "Later we only mention LDA topic modeling."
        ),
        split="test",
    )
    query = ranking_query(document, "Which task was solved?")
    assert query.startswith("Bi - Directional Attention Flow for Machine Comprehension.")
    assert query.endswith("Which task was solved?")
    assert "LDA topic modeling" not in query
    assert "Topic:" not in query


def test_ranking_query_uses_first_body_sentence_without_abstract_marker() -> None:
    document = TopicDocument(
        "d1",
        title="",
        abstract="",
        text=(
            "document: V - Net: Fully Convolutional Neural Networks for Volumetric "
            "Medical Image Segmentation Convolutional Neural Networks (CNNs) have "
            "been recently employed to solve problems. Later we mention LDA topics."
        ),
        split="test",
    )
    query = ranking_query(document, "Which task was solved?")
    assert query.startswith("V - Net:")
    assert "Which task was solved?" in query
    assert "been recently employed to solve problems." in query
    assert "LDA topics" not in query


def test_ranking_query_clips_body_fallback_to_the_opening_words() -> None:
    words = " ".join(f"token{index}" for index in range(80))
    document = TopicDocument("d1", text=words, split="test")
    query = ranking_query(document, "Which task was solved?", max_identity_words=8)
    identity = query[: -len(" Which task was solved?")].rstrip(".")
    assert len(identity.split()) == 8
    assert identity.startswith("token0")


def test_cross_encoder_and_keyword_channels_keep_chunk_order() -> None:
    gold = _chunk("frost pressure in soil", 0)
    other = _chunk("LDA topic modeling appendix", 1)
    chunks = [gold, other]
    ce = cross_encoder_chunk_scores(FakeCrossEncoder(), "title. question", chunks)
    assert ce[0] > ce[1]
    kw = keyword_chunk_scores(
        chunks,
        [WeightedKeyword("lda", "lda", "lda", 1.0, 1.0)],
        weight_ratio=0.0,
    )
    assert len(kw) == 2
    assert kw[1] > kw[0]


def test_ce_only_mix_ignores_keyword_channel() -> None:
    mixed = mixed_chunk_scores((2.5, -1.5), (0.0, 9.0), MixWeights(1.0, 0.0))
    assert mixed[0] > mixed[1]


def test_predict_from_panel_follows_the_active_channel() -> None:
    panel = _panel()
    encoder = Encoder()
    ce_prediction = predict_from_panel(panel, MixWeights(1.0, 0.0), encoder)
    kw_prediction = predict_from_panel(panel, MixWeights(0.0, 1.0), encoder)
    assert ce_prediction == "frost pressure"
    assert kw_prediction == "LDA"


def test_fit_mix_weights_uses_dev_gold_and_refuses_test() -> None:
    encoder = Encoder()
    selected = fit_mix_weights(
        [_panel(doc_id="dev-1"), _panel(doc_id="dev-2")],
        encoder,
        _exact_char_f1,
    )
    assert selected.weights == MixWeights(1.0, 0.0)
    assert selected.mean_char_f1 == 1.0
    with pytest.raises(ValueError, match="refuses test documents"):
        fit_mix_weights([_panel(split="test")], encoder, _exact_char_f1)


def test_collect_ranked_panel_answers_every_chunk() -> None:
    gold = _chunk("The paper solves frost pressure prediction.", 0)
    distractor = _chunk("Appendix discusses LDA topic modeling.", 1)
    document = TopicDocument(
        "dev-1",
        text="unused",
        title="Soil frost",
        split="dev",
    )
    panel = collect_ranked_panel(
        document,
        "Which task was solved?",
        processor=StubProcessor([gold, distractor]),
        answerer=FakeAnswerer(),
        reranker=FakeCrossEncoder(),
        keywords=[WeightedKeyword("lda", "lda", "lda", 1.0, 1.0)],
        keyword_weight_ratio=0.0,
        references=("frost pressure",),
    )
    assert panel.chunks == (gold, distractor)
    assert [answer.text for answer in panel.answers] == ["frost pressure", "LDA"]
    assert panel.cross_encoder_scores[0] > panel.cross_encoder_scores[1]
    assert panel.keyword_scores[1] > panel.keyword_scores[0]
    assert len(panel.answers) == 2


def _load_eval_script(name: str):
    script = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_encoder_eval_refuses_previous_dirs_and_excludes_own_modes(
    tmp_path,
) -> None:
    module = _load_eval_script("12_Evaluate_cross_encoder.py")
    root = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
    )
    base_args = [
        "--dataset",
        str(tmp_path / "data.json"),
        "--static-model",
        str(tmp_path / "static.json"),
        "--output-dir",
        str(root / "evaluation"),
    ]
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(module.build_parser().parse_args(base_args))

    for relative in (
        "refit/evaluation",
        "refit_no_gold_contrast/evaluation",
        "soft_rerank/evaluation",
        "ce_topic_query/evaluation",
        "ce_topic_query_body/evaluation",
    ):
        args = module.build_parser().parse_args(
            [*base_args[:-1], str(root / relative)]
        )
        with pytest.raises(ValueError, match="refuses to overwrite"):
            module.run(args)

    reused_path = tmp_path / "prev.csv"
    reused_path.write_text(
        "doc_id,mode,prediction,char_f1\n"
        "a,baseline,x,0.5\n"
        "a,cross-encoder,y,0.4\n"
        "a,cross-encoder-static-mix,z,0.6\n"
        "a,hierarchical-soft-rerank,w,0.55\n",
        encoding="utf-8",
    )
    kept = module.load_reused_rows(
        reused_path,
        {"a"},
        exclude_modes={"cross-encoder", "cross-encoder-static-mix"},
    )
    assert {row["mode"] for row in kept} == {"baseline", "hierarchical-soft-rerank"}


def test_cross_encoder_eval_loads_dev_and_test_without_mixing_splits(
    tmp_path,
) -> None:
    module = _load_eval_script("12_Evaluate_cross_encoder.py")
    payload = [
        {
            "doc_id": "train-1",
            "split": "train",
            "title": "Train",
            "original_text": "train text",
            "tasks": ["train task"],
        },
        {
            "doc_id": "dev-1",
            "split": "dev",
            "title": "Dev",
            "original_text": "dev text",
            "tasks": ["dev task"],
        },
        {
            "doc_id": "test-1",
            "split": "test",
            "title": "Test",
            "original_text": "test text",
            "tasks": ["test task"],
        },
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dev, test, references = module.load_split_documents(path, seed=42)
    assert [item.doc_id for item in dev] == ["dev-1"]
    assert [item.doc_id for item in test] == ["test-1"]
    assert references["dev-1"] == ("dev task",)
    assert references["test-1"] == ("test task",)
