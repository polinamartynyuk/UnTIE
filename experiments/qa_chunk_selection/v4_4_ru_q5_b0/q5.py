from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.phase1_multiquestion.assign import ChunkState, QuestionPrediction, assign_instance
from experiments.phase1_multiquestion.engine import PredictionCache
from experiments.phase1_scirex.engine import mention_chunk_ids
from experiments.phase1_scirex.match import classify_match
from experiments.phase1_scirex.tables import as_int_list, as_list

from .questions import QUESTIONS


def _cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def process_document(document_id: str, instances: pd.DataFrame, chunks: pd.DataFrame,
                     mentions: pd.DataFrame, config, answer_fn, encode_fn,
                     cache: PredictionCache, threshold: float):
    matrix = encode_fn(chunks["chunk_text"].astype(str).tolist())
    embeddings = {int(cid): matrix[i] for i, cid in enumerate(chunks["chunk_id"].tolist())}
    results = []
    for _, instance in instances.iterrows():
        aspect = str(instance["aspect"])
        questions = QUESTIONS[aspect]
        extractable = as_list(instance["extractable_gold_answers"])
        support = set(as_int_list(instance.get("gold_supporting_chunk_ids")))
        qvec = encode_fn([questions[0]])[0]
        states = []
        for _, row in chunks.iterrows():
            cid, text = int(row["chunk_id"]), str(row["chunk_text"])
            preds = []
            for qn, question in enumerate(questions):
                cached = cache.get(document_id, aspect, cid, qn)
                if cached is None:
                    raw = dict(answer_fn(question, text))
                    cached = {"document_id": document_id, "aspect": aspect, "chunk_id": cid,
                              "question_id": qn, "question": question,
                              "qa_predicted_answer": str(raw.get("answer") or ""),
                              "qa_confidence": float(raw.get("score") or 0.0),
                              "answer_start": raw.get("start"), "answer_end": raw.get("end"),
                              "start_logit": raw.get("start_logit"), "end_logit": raw.get("end_logit")}
                    cache.put(cached)
                match = classify_match(str(cached["qa_predicted_answer"]), extractable,
                                       method=config.positive_match_method,
                                       max_extra_tokens=config.containment_max_extra_tokens)
                preds.append(QuestionPrediction(qn, question, str(cached["qa_predicted_answer"]),
                    match.matched_gold, match.matched, float(cached.get("qa_confidence") or 0.0),
                    match.quality, match.match_type, match.extra_tokens,
                    cached.get("answer_start"), cached.get("answer_end"),
                    cached.get("start_logit"), cached.get("end_logit")))
            states.append(ChunkState(cid, text, cid in support, preds, 0, 5, 0.0,
                "robust_negative", False, int(row["char_start"]), int(row["char_end"]),
                int(row["token_start"]), int(row["token_end"]),
                str(row.get("section_type") or ""), str(row.get("section_title") or ""),
                bool(row.get("eligible_for_qa", True)), dense_question_similarity=_cosine(qvec, embeddings[cid])))
        results.append(assign_instance(document_id=document_id, aspect=aspect,
            split=str(instance["split"]), questions=questions, status=str(instance["status"]),
            extractable_gold_answers=extractable,
            unextractable_gold_answers=as_list(instance.get("unextractable_gold_answers")),
            chunks=states, embeddings_by_id=embeddings, config=config, threshold=threshold,
            gold_mention_chunk_ids=mention_chunk_ids(mentions, document_id, aspect)))
    return results

