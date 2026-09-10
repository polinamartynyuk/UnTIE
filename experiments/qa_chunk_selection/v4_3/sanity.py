"""TRAIN-only semantic sanity. Gold/source labels allowed here only."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.metrics import token_f1
from experiments.qa_chunk_selection.schema import InstanceRecord

from .features import cosine
from .text import is_valid_answer


def pairwise_gt(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    n = 0
    wins = 0
    for a in left:
        for b in right:
            n += 1
            if float(a) > float(b):
                wins += 1
    return float(wins / n) if n else None


def sanity_pairs(
    items: Sequence[InstanceRecord],
    union: Mapping[str, set[int]],
    qa_store: Mapping[tuple[str, int], Mapping[str, Any]],
    embeddings: Mapping[tuple[str, int], Any],
    *,
    reader_easy_threshold: float,
    gold_f1: Callable[[InstanceRecord, int], float],
) -> dict[str, Any]:
    pos_pos: list[float] = []
    pos_hn: list[float] = []
    n_questions = 0
    for item in items:
        cids = sorted(union.get(item.question_id) or [])
        easy_pos = []
        hn = []
        for cid in cids:
            chunk = next((row for row in item.chunks if int(row.chunk_id) == int(cid)), None)
            if chunk is None:
                continue
            qa = qa_store.get((item.question_id, int(cid))) or {}
            answer = str(qa.get("qa_answer") or "")
            if not is_valid_answer(answer):
                continue
            vec = embeddings.get((item.question_id, int(cid)))
            if vec is None:
                continue
            if chunk.label == "positive" and float(gold_f1(item, int(cid))) >= float(reader_easy_threshold):
                easy_pos.append((int(cid), vec, answer))
            if chunk.label == "hard_negative":
                hn.append((int(cid), vec, answer))
        if len(easy_pos) >= 2:
            n_questions += 1
            for i, (cid_a, vec_a, _a) in enumerate(easy_pos):
                for cid_b, vec_b, _b in easy_pos[i + 1 :]:
                    pos_pos.append(cosine(vec_a, vec_b))
        if easy_pos and hn:
            for _cid_a, vec_a, _a in easy_pos:
                for _cid_b, vec_b, _b in hn:
                    pos_hn.append(cosine(vec_a, vec_b))
    def _stats(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = list(values)
        arr.sort()
        return {
            "n": float(len(arr)),
            "mean": float(sum(arr) / len(arr)),
            "p10": float(arr[max(int(0.1 * (len(arr) - 1)), 0)]),
            "p50": float(arr[int(0.5 * (len(arr) - 1))]),
            "p90": float(arr[int(0.9 * (len(arr) - 1))]),
        }

    return {
        "n_train_questions_with_two_easy_pos": n_questions,
        "pos_pos": _stats(pos_pos),
        "pos_hn": _stats(pos_hn),
        "pairwise_pospos_gt_posHN": pairwise_gt(pos_pos, pos_hn),
        "diagnostic_only": True,
    }


def gold_token_f1(item: InstanceRecord, qa_store: Mapping[tuple[str, int], Mapping[str, Any]], cid: int) -> float:
    qa = qa_store.get((item.question_id, int(cid))) or {}
    return float(token_f1(str(qa.get("qa_answer") or ""), list(item.gold_answers)))
