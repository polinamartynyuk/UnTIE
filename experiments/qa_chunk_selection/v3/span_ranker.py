"""Span-level candidate reranking. Attribution-aware labels; gold string from TM is not positive."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.answerability import TfidfAnswerabilityScorer, TrainPair
from experiments.qa_chunk_selection.metrics import exact_match, qa_row
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.qa import answer_chunk
from experiments.qa_chunk_selection.v2.schemas import RuntimeInstance, chunk_to_runtime, rank_ids, to_runtime

from .roles import known_target_status, v3_target_positives, verified_target_mismatch


def span_input(question: str, chunk_text: str, answer: str) -> str:
    return f"[QUESTION] {question}\n[CHUNK] {chunk_text}\n[CANDIDATE ANSWER] {answer}"


def generate_candidates(
    gold: InstanceRecord,
    ranked_ids: Sequence[int],
    *,
    k: int,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
) -> list[dict[str, Any]]:
    runtime = to_runtime(gold)
    chosen = list(ranked_ids[: max(int(k), 1)])
    rows = []
    for rank, chunk_id in enumerate(chosen, start=1):
        chunk = gold.chunk_by_id(int(chunk_id))
        pred = answer_chunk(
            answer_fn,
            gold.question,
            chunk_to_runtime(chunk),
            cache=cache,
            model_id=model_id,
            inference_hash=inference_hash,
        )
        rows.append(
            {
                "chunk_id": int(chunk_id),
                "rank": rank,
                "qa_answer": pred.get("qa_answer") or "",
                "qa_span_score": float(pred.get("qa_span_score") or 0.0),
                "qa_chunk_start_char": pred.get("qa_chunk_start_char"),
                "qa_chunk_end_char": pred.get("qa_chunk_end_char"),
                "source_start_char": pred.get("source_start_char"),
                "source_end_char": pred.get("source_end_char"),
                "chunk_text": chunk.text,
            }
        )
    return rows


def label_candidate(
    gold: InstanceRecord,
    candidate: Mapping[str, Any],
    overlay: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    chunk = gold.chunk_by_id(int(candidate["chunk_id"]))
    raw = qa_row(candidate.get("qa_answer"), gold.gold_answers, skip_bertscore=True)
    p_v3 = {c.chunk_id for c in v3_target_positives(gold, overlay)}
    tm = {c.chunk_id for c in verified_target_mismatch(gold, overlay)}
    status = known_target_status(gold, overlay, chunk)
    em = float(raw.get("exact_match") or 0.0)
    f1 = float(raw.get("token_f1") or 0.0)
    in_target = chunk.chunk_id in p_v3
    is_tm = chunk.chunk_id in tm or status == "target_mismatch"
    span_correct = int(in_target and em >= 1.0)
    wrong_target = int(is_tm)
    attr_f1 = f1 if in_target else (0.0 if is_tm or status == "known_non_target" else f1)
    train_include = status not in {None, "partial_support", "unresolved"} or in_target or chunk.label in {"positive", "hard_negative", "easy_negative"}
    if status in {"partial_support", "unresolved"}:
        train_include = False
    return {
        **dict(candidate),
        "raw_EM": em,
        "raw_TokenF1": f1,
        "source_target_status": status,
        "span_correct_target": span_correct,
        "wrong_target": wrong_target,
        "attribution_aware_TokenF1": attr_f1,
        "train_include": train_include,
        "in_p_v3_target": in_target,
    }


def collect_span_pairs(
    records: Sequence[InstanceRecord],
    overlay,
    *,
    score_fn,
    k: int,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
) -> list[TrainPair]:
    pairs: list[TrainPair] = []
    for gold in records:
        runtime = to_runtime(gold)
        ranked = rank_ids(runtime.chunks, score_fn(runtime))
        for cand in generate_candidates(
            gold, ranked, k=k, answer_fn=answer_fn, cache=cache, model_id=model_id, inference_hash=inference_hash
        ):
            labeled = label_candidate(gold, cand, overlay)
            if not labeled["train_include"]:
                continue
            pairs.append(
                TrainPair(
                    question=gold.question,
                    text=span_input(gold.question, labeled["chunk_text"], labeled["qa_answer"]),
                    label=int(labeled["span_correct_target"]),
                    document_id=gold.document_id,
                    question_id=gold.question_id,
                    chunk_id=int(labeled["chunk_id"]),
                    kind="span_correct" if labeled["span_correct_target"] else "span_incorrect",
                )
            )
    return pairs


def fit_span_ranker(pairs: Sequence[TrainPair]):
    if not pairs or len({p.label for p in pairs}) < 2:
        class _Const:
            def predict(self, items):
                return [0.5] * len(items)

        return _Const()
    return TfidfAnswerabilityScorer().fit(pairs)


def rerank_spans(question: str, candidates: Sequence[Mapping[str, Any]], ranker) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(question, span_input(question, row["chunk_text"], row.get("qa_answer") or "")) for row in candidates]
    scores = list(ranker.predict(pairs))
    ordered = sorted(zip(candidates, scores), key=lambda item: (-float(item[1]), int(item[0]["chunk_id"])))
    out = []
    for rank, (row, score) in enumerate(ordered, start=1):
        item = dict(row)
        item["span_ranker_score"] = float(score)
        item["span_rank"] = rank
        out.append(item)
    return out


def attribution_aware_oracle(labeled: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    raw_best = max((float(row.get("raw_TokenF1") or 0.0) for row in labeled), default=0.0)
    attr_best = max(
        (float(row.get("attribution_aware_TokenF1") or 0.0) for row in labeled if row.get("in_p_v3_target")),
        default=0.0,
    )
    return {"raw_best_string_f1": raw_best, "attribution_aware_best_f1": attr_best}
