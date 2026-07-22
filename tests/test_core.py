from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from untie.attention import reconstruct_attention_words
from untie.chunking import ChunkBuilder
from untie.config import ModelProfile, PipelineConfig
from untie.domain import Answer, Question, Sentence, TextChunk
from untie.keywords import KeywordExtractor
from untie.pipelines import AnswerPipeline, AttentionRerankingPipeline, DocumentProcessor
from untie.qa import AnswerConsensus, AnswerFinder
from untie.ranking import WeightedKeyword, score_chunks
from untie.text import RussianSentenceSplitter, SentenceSplitter


class FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.replace(".", "").split()

    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        return [101, *range(len(self.tokenize(text))), 102]

    def convert_tokens_to_string(self, tokens) -> str:
        return " ".join(tokens)


class FakeEncoder:
    def encode(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vector = np.zeros(16, dtype=float)
            for word in text.lower().split():
                digest = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1
            vectors.append(vector)
        return np.asarray(vectors)


class FakeAnswerer:
    def __call__(self, *, question: str, context: str) -> dict:
        del question
        answer = "semantic segmentation" if "semantic segmentation" in context else context.split(".")[0]
        start = context.find(answer)
        return {
            "answer": answer,
            "score": 0.9,
            "start": max(start, 0),
            "end": max(start, 0) + len(answer),
        }


def test_sentence_splitter_preserves_decimal_and_title() -> None:
    result = SentenceSplitter().split("Dr. Smith reports 3.14 points. This works!")
    assert result == ["Dr. Smith reports 3.14 points.", "This works!"]


def test_russian_sentence_splitter_preserves_abbreviations_and_initials() -> None:
    result = RussianSentenceSplitter().split(
        "Проф. А. С. Иванов описал метод, т.е. основной алгоритм. Он работает!"
    )
    assert result == [
        "Проф. А. С. Иванов описал метод, т.е. основной алгоритм.",
        "Он работает!",
    ]


def test_attention_words_preserve_sentencepiece_and_roberta_behavior() -> None:
    sentencepiece = reconstruct_attention_words(
        ["▁анализ", "▁текст", "ов"],
        [0.4, 0.6, 0.8],
    )
    roberta = reconstruct_attention_words(
        ["Ġsemantic", "Ġsegment", "ation"],
        [0.3, 0.5, 0.7],
    )
    assert sentencepiece == {"анализ": [0.4], "текстов": [0.6, 0.8]}
    assert roberta == {"semantic": [0.3], "segmentation": [0.5, 0.7]}


def test_attention_words_support_wordpiece_boundaries() -> None:
    result = reconstruct_attention_words(
        ["анализ", "текст", "##ов"],
        [0.2, 0.4, 0.6],
        scheme="wordpiece",
    )
    assert result == {"анализ": [0.2], "текстов": [0.4, 0.6]}


def test_chunk_builder_respects_limit_and_overlap() -> None:
    builder = ChunkBuilder(FakeTokenizer(), max_tokens=6, overlap_tokens=2)
    sentences = builder.prepare_sentences(["one two three", "four five", "six seven"])
    chunks = builder.build(sentences)
    assert [chunk.token_count for chunk in chunks] == [5, 4]
    assert chunks[1].text.startswith("four five")


def test_keyword_filter_and_scoring() -> None:
    builder = ChunkBuilder(FakeTokenizer(), max_tokens=20, overlap_tokens=2)
    chunks = builder.build(
        builder.prepare_sentences(["semantic segmentation method.", "unrelated baseline."])
    )
    scored = score_chunks(
        chunks,
        [WeightedKeyword("segmentation", "segmentation", "segment", 0.8, 0.4)],
    )
    assert scored
    assert scored[0].matched_keywords == ("segmentation",)


def test_keyword_extractor_returns_ranked_candidates() -> None:
    keywords = KeywordExtractor(FakeEncoder()).extract(
        ["semantic segmentation segmentation network"],
        limit=2,
    )
    assert len(keywords) == 2
    assert "segmentation" in keywords


def test_pipeline_runs_without_model_loading() -> None:
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=20,
        overlap_tokens=2,
    )
    processor = DocumentProcessor(FakeTokenizer(), config)
    result = AnswerPipeline(processor, FakeAnswerer(), FakeEncoder(), config).run(
        "We study semantic segmentation. We propose a residual model.",
        ["Which task was solved?"],
    )
    assert result.final_answer is not None
    assert result.final_answer.text == "semantic segmentation"
    assert len(result.used_chunks) == 1


def test_fixture_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "sample_documents.json"
    documents = json.loads(fixture.read_text(encoding="utf-8"))
    assert {document["language"] for document in documents} == {"en", "ru"}
    assert all(document["question"] and document["reference_answer"] for document in documents)


def test_answer_finder_sorts_by_confidence() -> None:
    builder = ChunkBuilder(FakeTokenizer(), max_tokens=20, overlap_tokens=2)
    chunks = builder.build(builder.prepare_sentences(["semantic segmentation."]))
    question = AnswerFinder(FakeAnswerer()).find(Question("task?"), chunks)
    assert question.answers[0].confidence == 0.9


def test_attention_pipeline_runs_with_injected_extractor() -> None:
    config = PipelineConfig(
        profile=ModelProfile.english(),
        chunk_max_tokens=20,
        overlap_tokens=2,
        min_answer_threshold=0,
    )
    processor = DocumentProcessor(FakeTokenizer(), config)
    pipeline = AttentionRerankingPipeline(
        processor,
        FakeAnswerer(),
        FakeEncoder(),
        config,
        lambda question, context: [
            {
                "word": "semantic",
                "lemma": "semantic",
                "stem": "semantic",
                "weight": 0.8,
            }
        ],
    )
    result = pipeline.run(
        "We study semantic segmentation. We propose a residual model.",
        "Which task was solved?",
        "unrelated reference",
        aspect_name="semantic",
    )
    assert result.final_answer is not None
    assert result.metadata["keywords"] == ["semantic"]


def test_clustered_consensus_supports_legacy_strategy_matrix() -> None:
    sentence = Sentence("semantic segmentation", 0, ("semantic", "segmentation"))
    chunk = TextChunk((sentence,), sentence.text, sentence.token_count)
    answers = [
        Answer(
            "semantic segmentation",
            chunk,
            0.9,
            metadata={"chunk_score": 0.8},
        ),
        Answer(
            "segmentation task",
            chunk,
            0.8,
            metadata={"chunk_score": 0.7},
        ),
        Answer(
            "unrelated result",
            chunk,
            0.4,
            metadata={"chunk_score": 0.1},
        ),
    ]
    consensus = AnswerConsensus(FakeEncoder())
    for cluster_strategy in (
        "highest_avg_score",
        "weighted_score",
        "highest_cohesion",
    ):
        for answer_strategy in (
            "highest_chunk_score",
            "highest_similarity",
            "combined_score",
        ):
            selected = consensus.select_clustered(
                answers,
                similarity_threshold=0,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            assert selected is not None
            assert selected.metadata["selection_strategy"] == (
                f"{cluster_strategy}+{answer_strategy}"
            )
