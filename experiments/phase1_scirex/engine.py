"""Run frozen QA and Phase 1 assignment on precomputed SciREX chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from untie.config import PipelineConfig
from untie.models import ModelFactory

from .assign import ChunkState, InstanceResult, assign_instance
from .config import Phase1Config
from .match import classify_match
from .tables import (
    as_int_list,
    as_list,
    eligible_chunks,
    eligible_instances,
    load_preprocessing,
)


AnswerFn = Callable[[str, str], Mapping[str, Any]]
EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def mention_chunk_ids(mentions: pd.DataFrame, document_id: str, aspect: str) -> set[int]:
    subset = mentions[(mentions["document_id"] == document_id) & (mentions["aspect"] == aspect)]
    ids: set[int] = set()
    for _, row in subset.iterrows():
        if "in_bibliography" in subset.columns and bool(row.get("in_bibliography")):
            continue
        ids.update(as_int_list(row.get("chunk_ids")))
    return ids


def source_chunk_index(chunks: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in chunks.iterrows():
        index[(str(row["document_id"]), int(row["chunk_id"]))] = {
            "chunk_text": str(row["chunk_text"]),
            "char_start": int(row["char_start"]),
            "char_end": int(row["char_end"]),
            "token_start": int(row["token_start"]),
            "token_end": int(row["token_end"]),
        }
    return index


def load_models(config: Phase1Config) -> tuple[AnswerFn, EncodeFn, dict[str, str]]:
    factory = ModelFactory(PipelineConfig(device=config.device))
    answerer = factory.answerer
    encoder = factory.sentence_encoder

    def answer(question: str, context: str) -> Mapping[str, Any]:
        return answerer(question=question, context=context)

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        matrix = np.asarray(encoder.encode(list(texts), show_progress_bar=False), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    meta = {
        "qa_model": str(factory.config.profile.qa_model),
        "embedding_model": str(factory.config.profile.sentence_model),
        "device": factory.device,
    }
    return answer, encode, meta


def build_chunk_states(
    chunk_rows: pd.DataFrame,
    *,
    gold_supporting: set[int],
    question: str,
    extractable: Sequence[str],
    match_method: str,
    max_extra_tokens: int,
    answer_fn: AnswerFn,
    question_embedding: np.ndarray | None,
    chunk_embeddings: Mapping[int, np.ndarray],
) -> list[ChunkState]:
    states: list[ChunkState] = []
    for _, row in chunk_rows.iterrows():
        chunk_id = int(row["chunk_id"])
        text = str(row["chunk_text"])
        predicted = answer_fn(question, text)
        prediction = str(predicted.get("answer") or "")
        matched = classify_match(
            prediction,
            extractable,
            method=match_method,
            max_extra_tokens=max_extra_tokens,
        )
        dense = 0.0
        embedding = chunk_embeddings.get(chunk_id)
        if question_embedding is not None and embedding is not None:
            dense = _cosine(question_embedding, embedding)
        states.append(
            ChunkState(
                chunk_id=chunk_id,
                text=text,
                gold_supporting=chunk_id in gold_supporting,
                qa_positive=matched.matched,
                predicted_answer=prediction,
                matched_gold_answer=matched.matched_gold,
                match_type=matched.match_type,
                extra_tokens=matched.extra_tokens,
                qa_confidence=float(predicted.get("score") or 0.0),
                answer_quality=matched.quality,
                start=predicted.get("start"),
                end=predicted.get("end"),
                start_logit=_optional_float(predicted.get("start_logit")),
                end_logit=_optional_float(predicted.get("end_logit")),
                char_start=int(row["char_start"]),
                char_end=int(row["char_end"]),
                token_start=int(row["token_start"]),
                token_end=int(row["token_end"]),
                section_type=str(row.get("section_type") or ""),
                section_title=str(row.get("section_title") or ""),
                eligible_for_qa=bool(row.get("eligible_for_qa", True)),
                dense_question_similarity=dense,
            )
        )
    return states


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def load_encoder(config: Phase1Config) -> tuple[EncodeFn, dict[str, str]]:
    factory = ModelFactory(PipelineConfig(device=config.device))
    encoder = factory.sentence_encoder

    def encode(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=float)
        matrix = np.asarray(encoder.encode(list(texts), show_progress_bar=False), dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    meta = {
        "qa_model": str(factory.config.profile.qa_model),
        "embedding_model": str(factory.config.profile.sentence_model),
        "device": factory.device,
        "qa_reused_from_predictions": True,
    }
    return encode, meta


def process_document_instances(
    *,
    document_id: str,
    instance_rows: pd.DataFrame,
    chunk_rows: pd.DataFrame,
    mentions: pd.DataFrame,
    config: Phase1Config,
    answer_fn: AnswerFn,
    encode_fn: EncodeFn,
) -> list[InstanceResult]:
    eligible = chunk_rows
    if eligible.empty:
        return []
    chunk_matrix = encode_fn([str(text) for text in eligible["chunk_text"].tolist()])
    embeddings = {
        int(chunk_id): chunk_matrix[index]
        for index, chunk_id in enumerate(eligible["chunk_id"].tolist())
    }
    results: list[InstanceResult] = []
    for _, instance in instance_rows.iterrows():
        aspect = str(instance["aspect"])
        question = str(instance["question"])
        extractable = as_list(instance["extractable_gold_answers"])
        question_vec = encode_fn([question])[0]
        states = build_chunk_states(
            eligible,
            gold_supporting=set(as_int_list(instance.get("gold_supporting_chunk_ids"))),
            question=question,
            extractable=extractable,
            match_method=config.positive_match_method,
            max_extra_tokens=config.containment_max_extra_tokens,
            answer_fn=answer_fn,
            question_embedding=question_vec,
            chunk_embeddings=embeddings,
        )
        results.append(
            assign_instance(
                document_id=document_id,
                aspect=aspect,
                split=str(instance.get("split") or config.split),
                question=question,
                status=str(instance["status"]),
                extractable_gold_answers=extractable,
                unextractable_gold_answers=as_list(instance.get("unextractable_gold_answers")),
                chunks=states,
                embeddings_by_id=embeddings,
                config=config,
                gold_mention_chunk_ids=mention_chunk_ids(mentions, document_id, aspect),
            )
        )
    return results


def select_instances(tables: dict[str, pd.DataFrame], config: Phase1Config) -> pd.DataFrame:
    instances = eligible_instances(tables["aspect_instances"], split=config.split)
    if config.debug:
        from .tables import select_debug_document_ids

        doc_ids = select_debug_document_ids(instances, n_docs=config.debug_docs, seed=config.seed)
        instances = instances[instances["document_id"].isin(doc_ids)]
    if config.document_ids:
        from .tables import filter_instances_by_document_ids

        instances = filter_instances_by_document_ids(instances, config.document_ids)
    return instances.reset_index(drop=True)


def prediction_lookup(pred_rows: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for _, row in pred_rows.iterrows():
        key = (str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"]))
        index[key] = row.to_dict()
    return index


def process_document_from_predictions(
    *,
    document_id: str,
    instance_rows: pd.DataFrame,
    chunk_rows: pd.DataFrame,
    mentions: pd.DataFrame,
    predictions: Mapping[tuple[str, str, int], Mapping[str, Any]],
    config: Phase1Config,
    encode_fn: EncodeFn,
) -> list[InstanceResult]:
    """Re-match and re-assign using stored QA spans. Does not call the QA model."""
    eligible = chunk_rows
    if eligible.empty:
        return []
    chunk_matrix = encode_fn([str(text) for text in eligible["chunk_text"].tolist()])
    embeddings = {
        int(chunk_id): chunk_matrix[index]
        for index, chunk_id in enumerate(eligible["chunk_id"].tolist())
    }
    results: list[InstanceResult] = []
    for _, instance in instance_rows.iterrows():
        aspect = str(instance["aspect"])
        question = str(instance["question"])
        extractable = as_list(instance["extractable_gold_answers"])
        gold_supporting = set(as_int_list(instance.get("gold_supporting_chunk_ids")))
        question_vec = encode_fn([question])[0]
        states: list[ChunkState] = []
        for _, row in eligible.iterrows():
            chunk_id = int(row["chunk_id"])
            stored = predictions.get((document_id, aspect, chunk_id))
            if stored is None:
                raise KeyError(
                    f"missing stored QA prediction for {document_id}::{aspect} chunk {chunk_id}"
                )
            prediction = str(stored.get("qa_predicted_answer") or "")
            matched = classify_match(
                prediction,
                extractable,
                method=config.positive_match_method,
                max_extra_tokens=config.containment_max_extra_tokens,
            )
            embedding = embeddings.get(chunk_id)
            dense = 0.0
            if embedding is not None:
                dense = _cosine(question_vec, embedding)
            start = stored.get("answer_start")
            end = stored.get("answer_end")
            states.append(
                ChunkState(
                    chunk_id=chunk_id,
                    text=str(row["chunk_text"]),
                    gold_supporting=chunk_id in gold_supporting,
                    qa_positive=matched.matched,
                    predicted_answer=prediction,
                    matched_gold_answer=matched.matched_gold,
                    qa_confidence=float(stored.get("qa_confidence") or 0.0),
                    answer_quality=matched.quality,
                    start=None if pd.isna(start) else start,
                    end=None if pd.isna(end) else end,
                    start_logit=_optional_float(stored.get("start_logit")),
                    end_logit=_optional_float(stored.get("end_logit")),
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    token_start=int(row["token_start"]),
                    token_end=int(row["token_end"]),
                    section_type=str(row.get("section_type") or ""),
                    section_title=str(row.get("section_title") or ""),
                    eligible_for_qa=bool(row.get("eligible_for_qa", True)),
                    dense_question_similarity=dense,
                    match_type=matched.match_type,
                    extra_tokens=matched.extra_tokens,
                )
            )
        results.append(
            assign_instance(
                document_id=document_id,
                aspect=aspect,
                split=str(instance.get("split") or config.split),
                question=question,
                status=str(instance["status"]),
                extractable_gold_answers=extractable,
                unextractable_gold_answers=as_list(instance.get("unextractable_gold_answers")),
                chunks=states,
                embeddings_by_id=embeddings,
                config=config,
                gold_mention_chunk_ids=mention_chunk_ids(mentions, document_id, aspect),
            )
        )
    return results

