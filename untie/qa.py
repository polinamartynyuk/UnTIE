from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .domain import Answer, FinalAnswer, Question, ScoredChunk, TextChunk
from .protocols import QuestionAnswerer, SentenceEncoder


@dataclass
class AnswerFinder:
    answerer: QuestionAnswerer

    def find(
        self,
        question: Question,
        chunks: Iterable[TextChunk],
        *,
        workers: int = 1,
    ) -> Question:
        chunk_list = list(chunks)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                answers = list(executor.map(lambda chunk: self._answer(question.text, chunk), chunk_list))
        else:
            answers = [self._answer(question.text, chunk) for chunk in chunk_list]
        question.answers = sorted(answers, key=lambda item: item.confidence, reverse=True)
        return question

    def _answer(self, question: str, chunk: TextChunk) -> Answer:
        result = self.answerer(question=question, context=chunk.text)
        return Answer(
            text=str(result["answer"]),
            chunk=chunk,
            confidence=float(result["score"]),
            start_pos=result.get("start"),
            end_pos=result.get("end"),
        )


@dataclass
class ScoredAnswerFinder:
    answerer: QuestionAnswerer

    def find(self, question: Question, chunks: Iterable[ScoredChunk]) -> Question:
        answers: list[Answer] = []
        for scored_chunk in chunks:
            result = self.answerer(question=question.text, context=scored_chunk.chunk.text)
            answer = Answer(
                text=str(result["answer"]),
                chunk=scored_chunk.chunk,
                confidence=float(result["score"]),
                start_pos=result.get("start"),
                end_pos=result.get("end"),
                metadata={
                    "chunk_score": scored_chunk.score,
                    "matched_keywords": scored_chunk.matched_keywords,
                    "keyword_scores": scored_chunk.keyword_scores,
                },
            )
            answer.metadata["combined_score"] = answer.confidence * scored_chunk.score
            answers.append(answer)
        question.answers = sorted(
            answers, key=lambda item: float(item.metadata["chunk_score"]), reverse=True
        )
        return question


@dataclass
class AnswerValidator:
    encoder: SentenceEncoder
    strict_threshold: float = 0.9
    min_threshold: float = 0.7

    def validate(
        self, answers: list[Answer], reference: str, *, top_k: int = 1
    ) -> list[Answer]:
        if not answers:
            return []
        embeddings = np.asarray(self.encoder.encode([answer.text for answer in answers]))
        reference_embedding = np.asarray(self.encoder.encode([reference]))
        similarities = cosine_similarity(reference_embedding, embeddings)[0]

        strict = np.flatnonzero(similarities >= self.strict_threshold)
        if strict.size:
            selected = strict[np.argsort(similarities[strict])[::-1]]
        else:
            eligible = np.flatnonzero(similarities >= self.min_threshold)
            selected = eligible[np.argsort(similarities[eligible])[::-1]][:top_k]
        result: list[Answer] = []
        for index in selected:
            answers[int(index)].similarity_score = float(similarities[index])
            result.append(answers[int(index)])
        return result


@dataclass
class AnswerAggregator:
    encoder: SentenceEncoder
    cluster_threshold: float = 0.5

    def aggregate(self, answers: list[Answer]) -> FinalAnswer | None:
        if not answers:
            return None
        embeddings = np.asarray(self.encoder.encode([answer.text for answer in answers]))
        similarities = cosine_similarity(embeddings)
        best_index = int(np.argmax(similarities.mean(axis=1)))
        best = answers[best_index]
        supporting = tuple(answer for answer in answers if answer.text == best.text)
        return FinalAnswer(
            text=best.text,
            confidence=float(np.mean([answer.confidence for answer in answers])),
            supporting_answers=supporting,
        )


@dataclass
class AnswerConsensus:
    encoder: SentenceEncoder

    def select(
        self,
        answers: list[Answer],
        *,
        chunk_weight: float = 0.6,
        similarity_weight: float = 0.4,
        min_similarity: float = 0.7,
    ) -> Answer | None:
        if not answers:
            return None
        if len(answers) == 1:
            return answers[0]

        embeddings = np.asarray(self.encoder.encode([answer.text for answer in answers]))
        similarity_matrix = cosine_similarity(embeddings)
        scores: list[float] = []
        for index, answer in enumerate(answers):
            chunk_score = float(answer.metadata.get("chunk_score", 0))
            base = (chunk_score + answer.confidence) / 2
            peers = [
                similarity_matrix[index, other]
                for other in range(len(answers))
                if other != index and similarity_matrix[index, other] >= min_similarity
            ]
            scores.append(
                base * chunk_weight
                + (float(np.mean(peers)) if peers else 0.0) * similarity_weight
            )
        best_index = int(np.argmax(scores))
        answers[best_index].metadata["consensus_score"] = scores[best_index]
        return answers[best_index]

    def select_clustered(
        self,
        answers: list[Answer],
        *,
        similarity_threshold: float = 0.75,
        cluster_strategy: Literal[
            "highest_avg_score", "weighted_score", "highest_cohesion"
        ] = "weighted_score",
        answer_strategy: Literal[
            "highest_chunk_score", "highest_similarity", "combined_score"
        ] = "highest_chunk_score",
    ) -> Answer | None:
        """Выбирает ответ по матрице стратегий из исходного эксперимента."""
        if not answers:
            return None
        if len(answers) == 1:
            return answers[0]

        embeddings = np.asarray(self.encoder.encode([answer.text for answer in answers]))
        similarity_matrix = cosine_similarity(embeddings)
        clusters = self._greedy_clusters(similarity_matrix, similarity_threshold)
        metrics = [
            self._cluster_metrics(cluster, answers, similarity_matrix)
            for cluster in clusters
        ]

        selectors = {
            "highest_avg_score": lambda item: item["average_chunk_score"],
            "weighted_score": lambda item: item["weighted_score"],
            "highest_cohesion": lambda item: item["cohesion"],
        }
        selected = max(metrics, key=selectors[cluster_strategy])
        indices = selected["indices"]

        def answer_score(index: int) -> float:
            chunk_score = float(answers[index].metadata.get("chunk_score", 0.0))
            peers = [other for other in indices if other != index]
            similarity = (
                float(np.mean([similarity_matrix[index, other] for other in peers]))
                if peers
                else 0.0
            )
            if answer_strategy == "highest_chunk_score":
                return chunk_score
            if answer_strategy == "highest_similarity":
                return similarity
            return 0.5 * chunk_score + 0.5 * similarity

        best_index = max(indices, key=answer_score)
        best = answers[best_index]
        best.metadata.update(
            {
                "cluster_size": len(indices),
                "total_clusters": len(clusters),
                "cluster_average_score": selected["average_chunk_score"],
                "cluster_cohesion": selected["cohesion"],
                "selection_strategy": f"{cluster_strategy}+{answer_strategy}",
            }
        )
        return best

    @staticmethod
    def _greedy_clusters(
        similarity_matrix: np.ndarray, threshold: float
    ) -> list[list[int]]:
        clusters: list[list[int]] = []
        visited: set[int] = set()
        for index in range(len(similarity_matrix)):
            if index in visited:
                continue
            cluster = [index]
            visited.add(index)
            for other in range(index + 1, len(similarity_matrix)):
                if other not in visited and similarity_matrix[index, other] >= threshold:
                    cluster.append(other)
                    visited.add(other)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _cluster_metrics(
        indices: list[int],
        answers: list[Answer],
        similarity_matrix: np.ndarray,
    ) -> dict[str, object]:
        chunk_scores = [
            float(answers[index].metadata.get("chunk_score", 0.0))
            for index in indices
        ]
        pairwise = [
            float(similarity_matrix[left, right])
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
        ]
        average = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        return {
            "indices": indices,
            "average_chunk_score": average,
            "weighted_score": len(indices) * average,
            "cohesion": float(np.mean(pairwise)) if pairwise else 0.0,
        }
