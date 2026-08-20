from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

from .chunking import ChunkBuilder
from .config import PipelineConfig
from .domain import FinalAnswer, PipelineResult, Question, TextChunk
from .keywords import score_keyword_contrast
from .protocols import QuestionAnswerer, SentenceEncoder, WordTokenizer
from .qa import AnswerAggregator, AnswerConsensus, AnswerFinder, AnswerValidator, ScoredAnswerFinder
from .ranking import WeightedKeyword, filter_chunks, score_chunks
from .text import SentenceSplitter
from .topics import (
    TopicArtifact,
    TopicDocument,
    TopicFallbackReason,
    compute_qa_ratios,
    resolve_topic_keyword_profile,
    resolve_topic_match_fallback,
    route_document,
    route_document_to_nearest_leaf,
    score_topic_aware_chunks,
)


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
        return self._run_chunks(chunks, questions, keywords=keywords)

    def _run_chunks(
        self,
        chunks: list[TextChunk],
        questions: list[str],
        *,
        keywords: list[WeightedKeyword] | None = None,
    ) -> PipelineResult:
        """Run QA on already processed chunks without changing the public API."""
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
class StaticKeywordRerankingPipeline:
    """Reference-free reranking with caller-provided, reusable keywords."""

    processor: DocumentProcessor
    answerer: QuestionAnswerer
    encoder: SentenceEncoder
    config: PipelineConfig
    keywords: list[WeightedKeyword] = field(default_factory=list)

    def run(
        self,
        text: str,
        questions: str | list[str],
        *,
        keywords: list[WeightedKeyword] | None = None,
        weight_ratio: float = 0.5,
        minimum_matches: int = 1,
        position_weight: float = 0.3,
        frequency_weight: float = 0.7,
        selection_strategy: Literal["consensus", "clustered"] = "clustered",
        similarity_threshold: float = 0.75,
        cluster_strategy: Literal[
            "highest_avg_score", "weighted_score", "highest_cohesion"
        ] = "weighted_score",
        answer_strategy: Literal[
            "highest_chunk_score", "highest_similarity", "combined_score"
        ] = "highest_chunk_score",
    ) -> PipelineResult:
        question_texts = [questions] if isinstance(questions, str) else list(questions)
        static_keywords = self.keywords if keywords is None else keywords
        chunks = self.processor.process(text)
        scored = score_chunks(
            chunks,
            static_keywords,
            weight_ratio=weight_ratio,
            minimum_matches=minimum_matches,
            position_weight=position_weight,
            frequency_weight=frequency_weight,
        )
        keyword_names = [keyword.word for keyword in static_keywords]

        if not scored:
            baseline = AnswerPipeline(
                self.processor, self.answerer, self.encoder, self.config
            )._run_chunks(chunks, question_texts)
            return PipelineResult(
                baseline.final_answer,
                baseline.questions,
                baseline.used_chunks,
                metadata={
                    "keywords": keyword_names,
                    "selection_strategy": "baseline",
                    "keyword_fallback": True,
                },
            )

        finder = ScoredAnswerFinder(self.answerer)
        question_objects = [
            finder.find(Question(question), scored) for question in question_texts
        ]
        answers = [answer for question in question_objects for answer in question.answers]
        consensus = AnswerConsensus(self.encoder)
        if selection_strategy == "consensus":
            selected = consensus.select(answers)
            selected_strategy = "consensus"
        else:
            selected = consensus.select_clustered(
                answers,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            selected_strategy = f"{cluster_strategy}+{answer_strategy}"

        if selected is not None:
            selected.metadata.setdefault("selection_strategy", selected_strategy)
            final = FinalAnswer(
                text=selected.text,
                confidence=selected.confidence,
                supporting_answers=(selected,),
            )
        else:
            final = None
        return PipelineResult(
            final,
            tuple(question_objects),
            tuple(item.chunk for item in scored),
            metadata={
                "keywords": keyword_names,
                "selection_strategy": selected_strategy,
                "keyword_fallback": False,
            },
        )


@dataclass
class HierarchicalTopicRerankingPipeline:
    """Optional topic-aware reranking before the existing extractive reader."""

    processor: DocumentProcessor
    answerer: QuestionAnswerer
    encoder: SentenceEncoder
    config: PipelineConfig
    topic_artifact: TopicArtifact
    aspect_keywords: list[WeightedKeyword] = field(default_factory=list)
    static_weight_ratio: float = 0.5
    enforce_encoder_compatibility: bool = True
    expected_encoder_revision: str | None = None

    def run(
        self,
        text: str,
        questions: str | list[str],
        *,
        document: TopicDocument | None = None,
        selection_strategy: Literal["consensus", "clustered"] = "clustered",
        similarity_threshold: float = 0.75,
        cluster_strategy: Literal[
            "highest_avg_score", "weighted_score", "highest_cohesion"
        ] = "weighted_score",
        answer_strategy: Literal[
            "highest_chunk_score", "highest_similarity", "combined_score"
        ] = "highest_chunk_score",
    ) -> PipelineResult:
        question_texts = [questions] if isinstance(questions, str) else list(questions)
        chunks = self.processor.process(text)
        if not chunks:
            return PipelineResult(
                None,
                tuple(),
                tuple(),
                metadata={
                    "topic_enabled": True,
                    "fallback_reason": TopicFallbackReason.NO_RERANKED_CHUNKS.value,
                },
            )

        topic_document = document or TopicDocument("", text=text)
        if topic_document.text != text:
            topic_document = replace(topic_document, text=text)
        has_node_profiles = any(
            node.keyword_profile is not None for node in self.topic_artifact.nodes
        )
        if has_node_profiles:
            route = route_document_to_nearest_leaf(
                topic_document,
                self.encoder,
                self.topic_artifact,
                expected_encoder_name=(
                    self.config.profile.sentence_model
                    if self.enforce_encoder_compatibility
                    else None
                ),
                expected_encoder_revision=self.expected_encoder_revision,
            )
            resolved = resolve_topic_keyword_profile(route, self.topic_artifact)
            if resolved is not None:
                keyword_node_id, profile = resolved
                scored = score_chunks(
                    chunks,
                    list(profile.keywords),
                    weight_ratio={
                        "only_score_diff": 0.0,
                        "only_weight": 1.0,
                        "equal_weight_score_diff": 0.5,
                    }.get(profile.score_chunk_strategy, 0.5),
                )
                if scored:
                    result = self._answer_scored(
                        scored,
                        question_texts,
                        selection_strategy=selection_strategy,
                        similarity_threshold=similarity_threshold,
                        cluster_strategy=profile.choose_cluster_strategy,
                        answer_strategy=profile.choose_answer_strategy,
                    )
                    return PipelineResult(
                        result.final_answer,
                        result.questions,
                        result.used_chunks,
                        metadata={
                            "topic_enabled": True,
                            "topic_keyword_mode": True,
                            "topic_route": _route_metadata(route),
                            "topic_keyword_node_id": keyword_node_id,
                            "topic_keywords": [
                                keyword.word for keyword in profile.keywords
                            ],
                            "selection_strategy": result.metadata[
                                "selection_strategy"
                            ],
                            "keyword_fallback": False,
                            "compute": compute_qa_ratios(result.used_chunks, chunks),
                            "chunk_scores": [
                                {
                                    "start_sentence": item.chunk.start_sentence,
                                    "score": item.score,
                                    "matched_keywords": " | ".join(
                                        item.matched_keywords
                                    ),
                                }
                                for item in scored
                            ],
                        },
                    )
            fallback = self._fallback(
                chunks,
                question_texts,
                selection_strategy=selection_strategy,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            return PipelineResult(
                fallback.final_answer,
                fallback.questions,
                fallback.used_chunks,
                {
                    **fallback.metadata,
                    "topic_enabled": True,
                    "topic_keyword_mode": True,
                    "topic_route": _route_metadata(route),
                    "topic_keyword_node_id": (
                        resolved[0] if resolved is not None else None
                    ),
                    "topic_keywords": (
                        [item.word for item in resolved[1].keywords]
                        if resolved is not None
                        else []
                    ),
                    "fallback_reason": (
                        TopicFallbackReason.NO_TOPIC_KEYWORD_MATCHES.value
                        if resolved is not None
                        else TopicFallbackReason.NO_TOPIC_KEYWORDS.value
                    ),
                    "compute": compute_qa_ratios(fallback.used_chunks, chunks),
                },
            )

        route = route_document(
            topic_document,
            self.encoder,
            self.topic_artifact,
            expected_encoder_name=(
                self.config.profile.sentence_model
                if self.enforce_encoder_compatibility
                else None
            ),
            expected_encoder_revision=self.expected_encoder_revision,
        )
        route = resolve_topic_match_fallback(chunks, route, self.topic_artifact)
        scored = score_topic_aware_chunks(
            chunks,
            self.aspect_keywords,
            route,
            self.topic_artifact,
        )
        has_topic_signal = bool(scored) and any(
            float(item.keyword_scores.get("topic_raw", 0.0)) > 0 for item in scored
        )

        if not scored or not has_topic_signal:
            fallback = self._fallback(
                chunks,
                question_texts,
                selection_strategy=selection_strategy,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            metadata = {
                **fallback.metadata,
                "topic_enabled": True,
                "topic_route": _route_metadata(route),
                "routing_fallback_reason": (
                    route.fallback_reason.value if route.fallback_reason else None
                ),
                "fallback_reason": (
                    TopicFallbackReason.NO_RERANKED_CHUNKS.value
                    if not scored
                    else TopicFallbackReason.NO_TOPIC_MATCHES.value
                ),
                "compute": compute_qa_ratios(fallback.used_chunks, chunks),
            }
            return PipelineResult(
                fallback.final_answer,
                fallback.questions,
                fallback.used_chunks,
                metadata,
            )

        result = self._answer_scored(
            scored,
            question_texts,
            selection_strategy=selection_strategy,
            similarity_threshold=similarity_threshold,
            cluster_strategy=cluster_strategy,
            answer_strategy=answer_strategy,
        )
        return PipelineResult(
            result.final_answer,
            result.questions,
            result.used_chunks,
            metadata={
                "topic_enabled": True,
                "topic_route": _route_metadata(route),
                "routing_fallback_reason": (
                    route.fallback_reason.value if route.fallback_reason else None
                ),
                "fallback_reason": (
                    route.fallback_reason.value if route.fallback_reason else None
                ),
                "selection_strategy": result.metadata["selection_strategy"],
                "keyword_fallback": False,
                "topic_filter_enabled": self.topic_artifact.mixing.filter_unmatched_chunks,
                "topic_filtered_chunk_count": len(scored),
                "compute": compute_qa_ratios(result.used_chunks, chunks),
                "chunk_scores": [
                    {
                        "start_sentence": item.chunk.start_sentence,
                        "score": item.score,
                        **item.keyword_scores,
                    }
                    for item in scored
                ],
            },
        )

    def _fallback(
        self,
        chunks: list[TextChunk],
        questions: list[str],
        *,
        selection_strategy: str,
        similarity_threshold: float,
        cluster_strategy: str,
        answer_strategy: str,
    ) -> PipelineResult:
        static_scored = score_chunks(
            chunks,
            self.aspect_keywords,
            weight_ratio=self.static_weight_ratio,
        )
        if static_scored:
            result = self._answer_scored(
                static_scored,
                questions,
                selection_strategy=selection_strategy,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            return PipelineResult(
                result.final_answer,
                result.questions,
                result.used_chunks,
                {
                    **result.metadata,
                    "keyword_fallback": True,
                    "fallback_stage": TopicFallbackReason.STATIC_KEYWORDS.value,
                },
            )
        baseline = AnswerPipeline(
            self.processor, self.answerer, self.encoder, self.config
        )._run_chunks(chunks, questions)
        return PipelineResult(
            baseline.final_answer,
            baseline.questions,
            baseline.used_chunks,
            {
                "selection_strategy": "baseline",
                "keyword_fallback": True,
                "fallback_stage": TopicFallbackReason.BASELINE.value,
            },
        )

    def _answer_scored(
        self,
        scored: list,
        questions: list[str],
        *,
        selection_strategy: str,
        similarity_threshold: float,
        cluster_strategy: str,
        answer_strategy: str,
    ) -> PipelineResult:
        finder = ScoredAnswerFinder(self.answerer)
        question_objects = [
            finder.find(Question(question), scored) for question in questions
        ]
        answers = [answer for question in question_objects for answer in question.answers]
        consensus = AnswerConsensus(self.encoder)
        if selection_strategy == "consensus":
            selected = consensus.select(answers)
            selected_strategy = "consensus"
        else:
            selected = consensus.select_clustered(
                answers,
                similarity_threshold=similarity_threshold,
                cluster_strategy=cluster_strategy,
                answer_strategy=answer_strategy,
            )
            selected_strategy = f"{cluster_strategy}+{answer_strategy}"
        final = (
            FinalAnswer(
                text=selected.text,
                confidence=selected.confidence,
                supporting_answers=(selected,),
            )
            if selected
            else None
        )
        return PipelineResult(
            final,
            tuple(question_objects),
            tuple(item.chunk for item in scored),
            {"selection_strategy": selected_strategy},
        )


def _route_metadata(route: Any) -> list[dict[str, Any]]:
    return [
        {
            "node_id": item.node_id,
            "similarity": item.similarity,
            "weight": item.weight,
            "depth": item.depth,
        }
        for item in route.topics
    ]


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
