"""Run frozen QA on every question variant. Gold lists stay those from preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.phase1_scirex.engine import load_encoder, load_models, mention_chunk_ids, source_chunk_index
from experiments.phase1_scirex.match import classify_match
from experiments.phase1_scirex.tables import (
    as_int_list,
    as_list,
    eligible_chunks,
    eligible_instances,
    load_preprocessing,
    select_debug_document_ids,
)

from .assign import ChunkState, InstanceResult, QuestionPrediction, assign_instance
from .config import MultiQuestionConfig
from .questions import questions_for


AnswerFn = Callable[[str, str], Mapping[str, Any]]
EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def _json_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def load_canonical_predictions(path: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_parquet(path)
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"]))
        index[key] = row.to_dict()
    return index


class PredictionCache:
    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (
                    str(item["document_id"]),
                    str(item["aspect"]),
                    int(item["chunk_id"]),
                    int(item["question_id"]),
                )
                self.rows[key] = item

    def get(self, document_id: str, aspect: str, chunk_id: int, question_id: int) -> dict[str, Any] | None:
        return self.rows.get((document_id, aspect, chunk_id, question_id))

    def put(self, item: dict[str, Any]) -> None:
        key = (
            str(item["document_id"]),
            str(item["aspect"]),
            int(item["chunk_id"]),
            int(item["question_id"]),
        )
        self.rows[key] = item
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _prediction_from_canonical(stored: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(stored.get("qa_predicted_answer") or ""),
        "score": float(stored.get("qa_confidence") or 0.0),
        "start": stored.get("answer_start"),
        "end": stored.get("answer_end"),
        "start_logit": stored.get("start_logit"),
        "end_logit": stored.get("end_logit"),
    }


def _answer_question(
    *,
    document_id: str,
    aspect: str,
    chunk_id: int,
    question_id: int,
    question: str,
    text: str,
    answer_fn: AnswerFn | None,
    cache: PredictionCache,
    canonical: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    cached = cache.get(document_id, aspect, chunk_id, question_id)
    if cached is not None:
        return cached
    if question_id == 0 and (document_id, aspect, chunk_id) in canonical:
        predicted = _prediction_from_canonical(canonical[(document_id, aspect, chunk_id)])
    else:
        if answer_fn is None:
            raise RuntimeError(
                f"missing QA cache for {document_id}::{aspect} chunk {chunk_id} q{question_id}"
            )
        predicted = dict(answer_fn(question, text))
    item = {
        "document_id": document_id,
        "aspect": aspect,
        "chunk_id": int(chunk_id),
        "question_id": int(question_id),
        "question": question,
        "qa_predicted_answer": str(predicted.get("answer") or ""),
        "qa_confidence": float(predicted.get("score") or 0.0),
        "answer_start": _json_int(predicted.get("start")),
        "answer_end": _json_int(predicted.get("end")),
        "start_logit": _optional_float(predicted.get("start_logit")),
        "end_logit": _optional_float(predicted.get("end_logit")),
    }
    cache.put(item)
    return item


def process_document(
    *,
    document_id: str,
    instance_rows: pd.DataFrame,
    chunk_rows: pd.DataFrame,
    mentions: pd.DataFrame,
    config: MultiQuestionConfig,
    encode_fn: EncodeFn,
    answer_fn: AnswerFn | None,
    cache: PredictionCache,
    canonical: dict[tuple[str, str, int], dict[str, Any]],
    threshold: float,
) -> list[InstanceResult]:
    if chunk_rows.empty:
        return []
    chunk_matrix = encode_fn([str(text) for text in chunk_rows["chunk_text"].tolist()])
    embeddings = {
        int(chunk_id): chunk_matrix[index]
        for index, chunk_id in enumerate(chunk_rows["chunk_id"].tolist())
    }
    results: list[InstanceResult] = []
    for _, instance in instance_rows.iterrows():
        aspect = str(instance["aspect"])
        questions = questions_for(aspect, config.n_questions)
        extractable = as_list(instance["extractable_gold_answers"])
        gold_supporting = set(as_int_list(instance.get("gold_supporting_chunk_ids")))
        question_vec = encode_fn([questions[0]])[0]
        states: list[ChunkState] = []
        for _, row in chunk_rows.iterrows():
            chunk_id = int(row["chunk_id"])
            text = str(row["chunk_text"])
            preds: list[QuestionPrediction] = []
            for question_id, question in enumerate(questions):
                raw = _answer_question(
                    document_id=document_id,
                    aspect=aspect,
                    chunk_id=chunk_id,
                    question_id=question_id,
                    question=question,
                    text=text,
                    answer_fn=answer_fn,
                    cache=cache,
                    canonical=canonical,
                )
                matched = classify_match(
                    str(raw["qa_predicted_answer"]),
                    extractable,
                    method=config.positive_match_method,
                    max_extra_tokens=config.containment_max_extra_tokens,
                )
                preds.append(
                    QuestionPrediction(
                        question_id=question_id,
                        question=question,
                        predicted_answer=str(raw["qa_predicted_answer"]),
                        matched_gold_answer=matched.matched_gold,
                        qa_correct=matched.matched,
                        qa_confidence=float(raw["qa_confidence"] or 0.0),
                        answer_quality=matched.quality,
                        match_type=matched.match_type,
                        extra_tokens=matched.extra_tokens,
                        start=_json_int(raw.get("answer_start")),
                        end=_json_int(raw.get("answer_end")),
                        start_logit=_optional_float(raw.get("start_logit")),
                        end_logit=_optional_float(raw.get("end_logit")),
                    )
                )
            embedding = embeddings.get(chunk_id)
            dense = 0.0
            if embedding is not None:
                dense = _cosine(question_vec, embedding)
            states.append(
                ChunkState(
                    chunk_id=chunk_id,
                    text=text,
                    gold_supporting=chunk_id in gold_supporting,
                    predictions=preds,
                    n_correct=0,
                    n_questions=len(questions),
                    qa_consistency=0.0,
                    consistency_class="robust_negative",
                    any_correct=False,
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
        results.append(
            assign_instance(
                document_id=document_id,
                aspect=aspect,
                split=str(instance.get("split") or config.split),
                questions=questions,
                status=str(instance["status"]),
                extractable_gold_answers=extractable,
                unextractable_gold_answers=as_list(instance.get("unextractable_gold_answers")),
                chunks=states,
                embeddings_by_id=embeddings,
                config=config,
                threshold=threshold,
                gold_mention_chunk_ids=mention_chunk_ids(mentions, document_id, aspect),
            )
        )
    return results


def select_instances(tables: dict[str, pd.DataFrame], config: MultiQuestionConfig) -> pd.DataFrame:
    instances = eligible_instances(tables["aspect_instances"], split=config.split)
    if config.debug:
        doc_ids = select_debug_document_ids(instances, n_docs=config.debug_docs, seed=config.seed)
        instances = instances[instances["document_id"].isin(doc_ids)]
    if config.document_ids:
        from experiments.phase1_scirex.tables import filter_instances_by_document_ids

        instances = filter_instances_by_document_ids(instances, config.document_ids)
    return instances.reset_index(drop=True)


def preprocessing_gold_index(instances: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for _, row in instances.iterrows():
        index[(str(row["document_id"]), str(row["aspect"]))] = as_list(row["extractable_gold_answers"])
    return index


def usable_pair_rate(results: Sequence[InstanceResult]) -> float:
    if not results:
        return 0.0
    n_ok = sum(1 for item in results if item.robust_positive_ids and item.hard_negative_ids)
    return n_ok / float(len(results))


def embeddings_from_chunks(chunks: Sequence[ChunkState], encode_fn: EncodeFn) -> dict[int, np.ndarray]:
    if not chunks:
        return {}
    matrix = encode_fn([item.text for item in chunks])
    return {item.chunk_id: matrix[index] for index, item in enumerate(chunks)}


def select_threshold_on_dev(
    results: Sequence[InstanceResult],
    *,
    encode_fn: EncodeFn,
    config: MultiQuestionConfig,
    mentions: pd.DataFrame,
) -> dict[str, Any]:
    from .assign import relabel_with_threshold

    candidates = config.threshold_candidates()
    rows: list[dict[str, Any]] = []
    best_threshold = None
    best_pair_rate = -1.0
    for threshold in candidates:
        relabeled: list[InstanceResult] = []
        for item in results:
            embeddings = embeddings_from_chunks(item.chunks, encode_fn)
            relabeled.append(
                relabel_with_threshold(
                    item,
                    embeddings_by_id=embeddings,
                    config=config,
                    threshold=float(threshold),
                    gold_mention_chunk_ids=mention_chunk_ids(
                        mentions, item.document_id, item.aspect
                    ),
                )
            )
        n_robust = sum(len(item.robust_positive_ids) for item in relabeled)
        n_sensitive = sum(len(item.question_sensitive_ids) for item in relabeled)
        n_no_pos = sum(item.instance_status == "no_robust_positive" for item in relabeled)
        pair_rate = usable_pair_rate(relabeled)
        rows.append(
            {
                "threshold": float(threshold),
                "usable_pair_rate": pair_rate,
                "robust_positive_chunks": n_robust,
                "question_sensitive_chunks": n_sensitive,
                "no_robust_positive_instances": n_no_pos,
                "instances": len(relabeled),
            }
        )
        if pair_rate > best_pair_rate + 1e-12 or (
            abs(pair_rate - best_pair_rate) <= 1e-12
            and (best_threshold is None or threshold > best_threshold)
        ):
            best_pair_rate = pair_rate
            best_threshold = float(threshold)
    if best_threshold is None:
        raise RuntimeError("DEV threshold selection produced no candidate")
    final = [
        relabel_with_threshold(
            item,
            embeddings_by_id=embeddings_from_chunks(item.chunks, encode_fn),
            config=config,
            threshold=best_threshold,
            gold_mention_chunk_ids=mention_chunk_ids(mentions, item.document_id, item.aspect),
        )
        for item in results
    ]
    return {
        "chosen_threshold": best_threshold,
        "criterion": "maximize DEV usable_pair_rate; ties take the higher threshold",
        "candidates": rows,
        "results": final,
    }


# Re-export helpers used by the CLI.
load_preprocessing = load_preprocessing
eligible_chunks = eligible_chunks
source_chunk_index = source_chunk_index
load_models = load_models
load_encoder = load_encoder
mention_chunk_ids = mention_chunk_ids
