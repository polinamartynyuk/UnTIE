from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .chunking import ChunkBuilder
from .config import PipelineConfig
from .domain import FinalAnswer, PipelineResult, Question, TextChunk
from .keywords import score_keyword_contrast
from .protocols import QuestionAnswerer, SentenceEncoder, WordTokenizer
from .qa import AnswerAggregator, AnswerConsensus, AnswerFinder, AnswerValidator, ScoredAnswerFinder
from .ranking import WeightedKeyword, filter_chunks, score_chunks
from .text import SentenceSplitter


@dataclass
class DocumentProcessor:
    tokenizer: WordTokenizer
    config: PipelineConfig
    splitter: SentenceSplitter = field(default_factory=SentenceSplitter)
    sentence_encoder: SentenceEncoder | None = None

    def process(self, text: str) -> list[TextChunk]:
        builder = ChunkBuilder(
            tokenizer=self.tokenizer,
            max_tokens=self.config.chunk_max_tokens,
            overlap_tokens=self.config.overlap_tokens,
            sentence_encoder=self.sentence_encoder,
        )
        sentences = builder.prepare_sentences(self.splitter.split(text))
        return builder.build(sentences)


@dataclass
class AnswerPipeline:
    processor: DocumentProcessor
    answerer: QuestionAnswerer
    encoder: SentenceEncoder
    config: PipelineConfig

    def run(
        self,
        text: str,
        questions: list[str],
        *,
        keywords: list[WeightedKeyword] | None = None,
    ) -> PipelineResult:
        chunks = self.processor.process(text)
        used_chunks = filter_chunks(chunks, keywords) if keywords is not None else chunks
        if not used_chunks:
            return PipelineResult(None, tuple(), tuple())

        finder = AnswerFinder(self.answerer)
        question_objects = [finder.find(Question(question), used_chunks) for question in questions]
        all_answers = [answer for question in question_objects for answer in question.answers]
        final = AnswerAggregator(
            self.encoder, self.config.answer_cluster_threshold
        ).aggregate(all_answers)
        return PipelineResult(final, tuple(question_objects), tuple(used_chunks))

    def validate(
        self, result: PipelineResult, reference_answer: str
    ) -> list[Any]:
        validator = AnswerValidator(
            self.encoder,
            self.config.strict_answer_threshold,
            self.config.min_answer_threshold,
        )
        answers = [answer for question in result.questions for answer in question.answers]
        return validator.validate(answers, reference_answer)


@dataclass
class AttentionRerankingPipeline:
    processor: DocumentProcessor
    answerer: QuestionAnswerer
    encoder: SentenceEncoder
    config: PipelineConfig
    attention_keywords: Callable[[str, str], list[dict[str, float | str]]]

    def run(
        self,
        text: str,
        question: str,
        reference_answer: str,
        *,
        aspect_name: str | None = None,
    ) -> PipelineResult:
        base = AnswerPipeline(self.processor, self.answerer, self.encoder, self.config)
        initial = base.run(text, [question])
        valid_answers = base.validate(initial, reference_answer)
        if not valid_answers:
            return initial

        raw_keywords: dict[str, dict[str, float | str]] = {}
        for answer in valid_answers:
            for keyword in self.attention_keywords(question, answer.chunk.text):
                word = str(keyword["word"])
                if word not in raw_keywords or float(keyword.get("weight", 0)) > float(
                    raw_keywords[word].get("weight", 0)
                ):
                    raw_keywords[word] = keyword

        contrasted = score_keyword_contrast(
            list(raw_keywords.values()),
            positive_reference=aspect_name or question,
            negative_reference=reference_answer,
            encoder=self.encoder,
        )
        weighted = [
            WeightedKeyword(
                word=word,
                lemma=str(item.get("lemma", word)).lower(),
                stem=str(item.get("stem", word)).lower(),
                attention_weight=float(item.get("weight", 1)),
                score_difference=float(item.get("score_diff", 1)),
            )
            for item in contrasted
            for word in (str(item["word"]),)
        ]
        chunks = self.processor.process(text)
        scored = score_chunks(chunks, weighted)
        reranked_question = ScoredAnswerFinder(self.answerer).find(Question(question), scored)
        consensus = AnswerConsensus(self.encoder).select(reranked_question.answers)
        final = (
            FinalAnswer(
                text=consensus.text,
                confidence=consensus.confidence,
                supporting_answers=(consensus,),
            )
            if consensus
            else initial.final_answer
        )
        return PipelineResult(
            final,
            (reranked_question,),
            tuple(item.chunk for item in scored),
            metadata={"keywords": [item.word for item in weighted]},
        )
