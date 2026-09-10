from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from untie.ce_topic_shortlist import (
    TopicCEPanel,
    collect_topic_ce_panel,
    fit_shortlist_k,
    predict_from_shortlist,
    predict_with_aggregator,
    ranked_topic_terms,
    shortlist_answers,
    topic_ranking_query,
)
from untie.cross_encoder_ranking import ranking_query
from untie.domain import Answer, Sentence, TextChunk
from untie.topics import TopicDocument, TopicTerm


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


class QueryAwareCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            query_l = query.lower()
            passage_l = passage.lower()
            if "network" in query_l:
                scores.append(2.0 if "lda" in passage_l else -1.0)
            else:
                scores.append(2.0 if "frost" in passage_l else -1.0)
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


def _panel(*, split: str = "dev", doc_id: str = "dev-1") -> TopicCEPanel:
    gold = _chunk("The paper solves frost pressure prediction.", 0)
    other_gold = _chunk("We study frost pressure in soil.", 1)
    distractor = _chunk("Appendix discusses LDA topic modeling.", 2)
    return TopicCEPanel(
        doc_id=doc_id,
        split=split,
        question="Which task was solved?",
        references=("frost pressure",),
        chunks=(gold, other_gold, distractor),
        answers=(
            Answer(text="frost pressure", chunk=gold, confidence=0.9),
            Answer(text="frost pressure", chunk=other_gold, confidence=0.8),
            Answer(text="LDA", chunk=distractor, confidence=0.9),
        ),
        title_query="Soil frost. Which task was solved?",
        topic_query="Soil frost. Which task was solved? Topic: networks.",
        title_ce_scores=(2.0, 1.5, -1.0),
        topic_ce_scores=(-1.0, -0.5, 2.0),
        leaf_id="leaf_1",
        topic_terms=("networks",),
    )


def _exact_char_f1(prediction: str | None, references) -> float:
    gold = str(references[0]) if references else ""
    return 1.0 if (prediction or "") == gold else 0.0


def test_topic_query_appends_leaf_terms_and_falls_back_without_them() -> None:
    document = TopicDocument("d1", title="Soil frost pressure", split="dev")
    base = ranking_query(document, "Which task was solved?")
    assert topic_ranking_query(document, "Which task was solved?", ()) == base
    assert "Topic: soil, frost." in topic_ranking_query(
        document, "Which task was solved?", ("soil", "frost", "soil")
    )


def test_topic_query_keeps_body_identity_before_leaf_terms() -> None:
    document = TopicDocument(
        "d1",
        title="",
        abstract="",
        text=(
            "document: Improved Language Modeling by Decoding the Past "
            "section: Abstract Highly regularized LSTMs. Plots come later."
        ),
        split="dev",
    )
    query = topic_ranking_query(
        document, "Which task was solved?", ("times faster", "hours")
    )
    assert query.startswith("Improved Language Modeling by Decoding the Past.")
    assert "Which task was solved?" in query
    assert query.endswith("Topic: times faster, hours.")
    assert "Plots come later" not in query


def test_ranked_topic_terms_orders_by_weight_and_deduplicates() -> None:
    terms = ranked_topic_terms(
        [
            TopicTerm("frost", 0.2, 0.5),
            TopicTerm("soil", 0.9, 0.5),
            TopicTerm("SOIL", 0.8, 0.5),
            TopicTerm("pressure", 0.4, 0.5),
        ],
        max_terms=2,
    )
    assert terms == ("soil", "pressure")


def test_shortlist_keeps_highest_ce_scores() -> None:
    panel = _panel()
    selected = shortlist_answers(panel.answers, panel.topic_ce_scores, k=1)
    assert [item.text for item in selected] == ["LDA"]
    assert len(shortlist_answers(panel.answers, panel.title_ce_scores, k=None)) == 3
    assert len(shortlist_answers(panel.answers, panel.title_ce_scores, k=8)) == 3


def test_aggregator_all_prefers_repeated_span_and_topic_shortlist_can_flip() -> None:
    panel = _panel()
    encoder = Encoder()
    all_pred = predict_with_aggregator(panel.answers, encoder)
    title_pred = predict_from_shortlist(
        panel, panel.title_ce_scores, encoder, k=2
    )
    topic_pred = predict_from_shortlist(
        panel, panel.topic_ce_scores, encoder, k=1
    )
    assert all_pred == "frost pressure"
    assert title_pred == "frost pressure"
    assert topic_pred == "LDA"


def test_fit_shortlist_k_uses_dev_gold_and_refuses_test() -> None:
    encoder = Encoder()
    selected = fit_shortlist_k(
        [_panel(doc_id="dev-1"), _panel(doc_id="dev-2")],
        encoder,
        _exact_char_f1,
        lambda panel: panel.title_ce_scores,
        grid=(1, None),
    )
    assert selected.k is None
    assert selected.mean_char_f1 == 1.0
    with pytest.raises(ValueError, match="refuses test documents"):
        fit_shortlist_k(
            [_panel(split="test")],
            encoder,
            _exact_char_f1,
            lambda panel: panel.title_ce_scores,
        )


def test_collect_panel_answers_every_chunk_and_changes_scores_with_topic_query() -> None:
    gold = _chunk("The paper solves frost pressure prediction.", 0)
    distractor = _chunk("Appendix discusses LDA topic modeling.", 1)
    document = TopicDocument(
        "dev-1",
        text="unused",
        title="Soil frost",
        split="dev",
    )
    panel = collect_topic_ce_panel(
        document,
        "Which task was solved?",
        processor=StubProcessor([gold, distractor]),
        answerer=FakeAnswerer(),
        reranker=QueryAwareCrossEncoder(),
        leaf_terms=("neural", "network"),
        leaf_id="leaf_1",
        references=("frost pressure",),
    )
    assert [answer.text for answer in panel.answers] == ["frost pressure", "LDA"]
    assert panel.title_ce_scores[0] > panel.title_ce_scores[1]
    assert panel.topic_ce_scores[1] > panel.topic_ce_scores[0]
    assert "Topic:" in panel.topic_query
    assert panel.leaf_id == "leaf_1"


def _load_eval_script(name: str):
    script = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topic_ce_eval_refuses_previous_dirs_and_excludes_own_modes(
    tmp_path,
) -> None:
    module = _load_eval_script("13_Evaluate_ce_topic_shortlist.py")
    root = (
        Path(__file__).parents[1]
        / "artifacts"
        / "hierarchical_topic_optimized"
        / "en"
    )
    base_args = [
        "--dataset",
        str(tmp_path / "data.json"),
        "--topic-model",
        str(tmp_path / "topic.json"),
        "--output-dir",
        str(root / "evaluation"),
    ]
    with pytest.raises(ValueError, match="refuses to overwrite"):
        module.run(module.build_parser().parse_args(base_args))

    for relative in (
        "refit/evaluation",
        "refit_no_gold_contrast/evaluation",
        "soft_rerank/evaluation",
        "cross_encoder/evaluation",
        "ce_topic_query/evaluation",
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
        "a,aggregator-all,y,0.4\n"
        "a,ce-title-shortlist,z,0.6\n"
        "a,cross-encoder,w,0.55\n",
        encoding="utf-8",
    )
    kept = module.load_reused_rows(
        reused_path,
        {"a"},
        exclude_modes={
            "aggregator-all",
            "ce-title-shortlist",
            "ce-topic-query-shortlist",
        },
    )
    assert {row["mode"] for row in kept} == {"baseline", "cross-encoder"}
