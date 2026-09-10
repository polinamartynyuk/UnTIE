"""R3 reader-aware reranking. QA span score is a confidence proxy only."""

from __future__ import annotations

from typing import Any, Sequence

from experiments.qa_chunk_selection.ranking import min_max

from .qa import answer_chunk
from .schemas import RuntimeInstance, rank_ids
from .signals import fuse_ranks, relevance_scores


def qa_proxy_scores(
    instance: RuntimeInstance,
    chunk_ids: Sequence[int],
    *,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for chunk_id in chunk_ids:
        chunk = instance.chunk_by_id(int(chunk_id))
        out[int(chunk_id)] = answer_chunk(
            answer_fn,
            instance.question,
            chunk,
            cache=cache,
            model_id=model_id,
            inference_hash=inference_hash,
        )
    return out


def r3_scores(
    instance: RuntimeInstance,
    *,
    mode: str,
    k_pre: int,
    weights: tuple[float, float, float],
    fusion: str,
    b_scores: Sequence[float],
    a_scores: Sequence[float] | None,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
    rrf_k: int = 60,
) -> tuple[list[float], dict[str, Any]]:
    b_rank = rank_ids(instance.chunks, b_scores)
    w_a, w_b, w_q = weights
    if mode == "full":
        candidate_ids = [chunk.chunk_id for chunk in instance.chunks]
    else:
        candidate_ids = b_rank[: max(int(k_pre), 1)]
    if abs(float(w_q)) > 1e-12:
        qa_map = qa_proxy_scores(
            instance,
            candidate_ids,
            answer_fn=answer_fn,
            cache=cache,
            model_id=model_id,
            inference_hash=inference_hash,
        )
    else:
        qa_map = {}
    qa_channel = []
    for chunk in instance.chunks:
        if chunk.chunk_id in qa_map:
            qa_channel.append(float(qa_map[chunk.chunk_id].get("qa_confidence_proxy") or 0.0))
        else:
            qa_channel.append(float("-inf"))
    if fusion == "rank":
        rankings = [rank_ids(instance.chunks, b_scores)]
        if w_a:
            rankings.append(rank_ids(instance.chunks, a_scores or [0.0] * len(instance.chunks)))
        qa_finite = [value if value != float("-inf") else -1e9 for value in qa_channel]
        if w_q:
            rankings.append(rank_ids(instance.chunks, qa_finite))
        fused = fuse_ranks(rankings, rrf_k=rrf_k)
        scores = [float(fused.get(chunk.chunk_id, 0.0)) for chunk in instance.chunks]
        if mode == "topk":
            allowed = set(candidate_ids)
            scores = [score if chunk.chunk_id in allowed else score - 1e6 for chunk, score in zip(instance.chunks, scores)]
    else:
        channels = []
        used_w = []
        if w_a:
            channels.append(a_scores or [0.0] * len(instance.chunks))
            used_w.append(w_a)
        channels.append(b_scores)
        used_w.append(w_b)
        if w_q:
            present = [value for value in qa_channel if value != float("-inf")]
            floor = min(present) if present else 0.0
            filled = [floor if value == float("-inf") else value for value in qa_channel]
            channels.append(filled)
            used_w.append(w_q)
        scores = [0.0] * len(instance.chunks)
        for channel, weight in zip(channels, used_w):
            scaled = min_max(channel)
            for index, value in enumerate(scaled):
                scores[index] += float(weight) * float(value)
        if mode == "topk":
            allowed = set(candidate_ids)
            scores = [score if chunk.chunk_id in allowed else score - 1e6 for chunk, score in zip(instance.chunks, scores)]
    meta = {
        "r3_mode": mode,
        "k_pre": int(k_pre),
        "weights": list(weights),
        "fusion": fusion,
        "qa_forwards_logical": len(candidate_ids) if abs(float(w_q)) > 1e-12 else 0,
        "channels": "A/B2/QA-confidence-proxy",
    }
    return scores, meta


def r3_weight_grid() -> list[tuple[str, tuple[float, float, float]]]:
    return [
        ("R3_0", (0.0, 1.0, 0.0)),
        ("R3_1", (0.3, 0.7, 0.0)),
        ("R3_2", (0.0, 0.6, 0.4)),
        ("R3_3", (0.2, 0.5, 0.3)),
    ]
