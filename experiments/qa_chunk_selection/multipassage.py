"""Frozen extractive QA plus globally competing start/end logits for experiment C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from untie.models import ExtractiveQuestionAnswerer

from .schema import ChunkRecord, GoldSpan


@dataclass(frozen=True)
class PassageLogits:
    chunk_id: int
    text: str
    start_logits: tuple[float, ...]
    end_logits: tuple[float, ...]
    offsets: tuple[tuple[int, int], ...]
    token_is_context: tuple[bool, ...]


@dataclass(frozen=True)
class SpanCandidate:
    chunk_id: int
    start_char: int
    end_char: int
    text: str
    start_logit: float
    end_logit: float

    @property
    def logit_sum(self) -> float:
        return float(self.start_logit + self.end_logit)


def _context_indices(sequence_ids: Sequence[int | None]) -> list[int]:
    return [index for index, seq in enumerate(sequence_ids) if seq == 1]


def encode_passage(
    answerer: ExtractiveQuestionAnswerer,
    question: str,
    chunk: ChunkRecord,
) -> PassageLogits:
    import torch

    encoded = answerer.tokenizer(
        question,
        chunk.text,
        return_tensors="pt",
        truncation="only_second",
        max_length=answerer.max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0]
    try:
        sequence_ids = encoded.sequence_ids(0)
    except (AttributeError, ValueError) as error:
        raise RuntimeError("extractive QA requires a fast tokenizer") from error
    model_inputs = {key: value.to(answerer.device) for key, value in encoded.items()}
    answerer.model.eval()
    with torch.no_grad():
        outputs = answerer.model(**model_inputs)
    start_logits = outputs.start_logits[0].detach().cpu().tolist()
    end_logits = outputs.end_logits[0].detach().cpu().tolist()
    offset_pairs = []
    for start, end in offsets.tolist():
        offset_pairs.append((int(start), int(end)))
    context_mask = tuple(seq == 1 for seq in sequence_ids)
    return PassageLogits(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        start_logits=tuple(float(item) for item in start_logits),
        end_logits=tuple(float(item) for item in end_logits),
        offsets=tuple(offset_pairs),
        token_is_context=context_mask,
    )


def independent_best_span(
    passage: PassageLogits,
    *,
    max_answer_tokens: int = 64,
) -> SpanCandidate | None:
    context = [index for index, flag in enumerate(passage.token_is_context) if flag]
    if not context:
        return None
    context_set = set(context)
    best_start = context[0]
    best_end = best_start
    best_score = float("-inf")
    for start in context:
        last_end = min(start + max_answer_tokens, context[-1] + 1)
        for end in range(start, last_end):
            if end not in context_set:
                continue
            score = passage.start_logits[start] + passage.end_logits[end]
            if score > best_score:
                best_score = float(score)
                best_start, best_end = start, end
    start_char = int(passage.offsets[best_start][0])
    end_char = int(passage.offsets[best_end][1])
    return SpanCandidate(
        chunk_id=passage.chunk_id,
        start_char=start_char,
        end_char=end_char,
        text=passage.text[start_char:end_char],
        start_logit=float(passage.start_logits[best_start]),
        end_logit=float(passage.end_logits[best_end]),
    )


def global_best_span(
    passages: Sequence[PassageLogits],
    *,
    max_answer_tokens: int = 64,
) -> SpanCandidate | None:
    """Compete with raw start/end logits across passages of one question.

    Independent per-passage softmax confidence is intentionally not used.
    """
    best: SpanCandidate | None = None
    best_score = float("-inf")
    for passage in passages:
        candidate = independent_best_span(passage, max_answer_tokens=max_answer_tokens)
        if candidate is None:
            continue
        if candidate.logit_sum > best_score:
            best = candidate
            best_score = candidate.logit_sum
    return best


def global_start_end_probs(
    passages: Sequence[PassageLogits],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    start_keys: list[tuple[int, int]] = []
    start_logits: list[float] = []
    end_keys: list[tuple[int, int]] = []
    end_logits: list[float] = []
    for passage in passages:
        for index, flag in enumerate(passage.token_is_context):
            if not flag:
                continue
            start_keys.append((passage.chunk_id, index))
            start_logits.append(passage.start_logits[index])
            end_keys.append((passage.chunk_id, index))
            end_logits.append(passage.end_logits[index])
    start_prob = _softmax_map(start_keys, start_logits)
    end_prob = _softmax_map(end_keys, end_logits)
    return start_prob, end_prob


def _softmax_map(keys: Sequence[tuple[int, int]], logits: Sequence[float]) -> dict[tuple[int, int], float]:
    if not logits:
        return {}
    arr = np.asarray(logits, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    denom = float(np.sum(exp)) or 1.0
    return {key: float(value) / denom for key, value in zip(keys, exp)}


def independent_softmax_confidence(passage: PassageLogits, max_answer_tokens: int = 64) -> float:
    """Per-passage softmax confidence. Not comparable across passages."""
    candidate = independent_best_span(passage, max_answer_tokens=max_answer_tokens)
    if candidate is None:
        return 0.0
    context = [index for index, flag in enumerate(passage.token_is_context) if flag]
    start = np.asarray([passage.start_logits[index] for index in context], dtype=float)
    end = np.asarray([passage.end_logits[index] for index in context], dtype=float)
    start = np.exp(start - np.max(start))
    end = np.exp(end - np.max(end))
    start = start / start.sum()
    end = end / end.sum()
    # Map best char span back to the context-local argmax used above.
    local_start = int(np.argmax([passage.start_logits[index] for index in context]))
    local_end = int(np.argmax([passage.end_logits[index] for index in context]))
    del candidate
    return float(start[local_start] * end[local_end])


def gold_token_indices(passage: PassageLogits, span: GoldSpan) -> tuple[int, int] | None:
    start_idx = None
    end_idx = None
    for index, flag in enumerate(passage.token_is_context):
        if not flag:
            continue
        token_start, token_end = passage.offsets[index]
        if token_end <= token_start:
            continue
        if start_idx is None and token_start <= span.start_char < token_end:
            start_idx = index
        if token_start < span.end_char <= token_end or (
            token_start <= span.start_char and span.end_char <= token_end
        ):
            end_idx = index
    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None
    return start_idx, end_idx


def marginal_gold_loss(
    passages: Sequence[PassageLogits],
    gold_spans: Sequence[GoldSpan],
) -> float:
    """-log sum_{g in G} P_s(g) P_e(g) with a global softmax over passages."""
    start_prob, end_prob = global_start_end_probs(passages)
    by_chunk = {item.chunk_id: item for item in passages}
    mass = 0.0
    for span in gold_spans:
        passage = by_chunk.get(span.chunk_id)
        if passage is None:
            continue
        indices = gold_token_indices(passage, span)
        if indices is None:
            continue
        start_idx, end_idx = indices
        mass += start_prob.get((span.chunk_id, start_idx), 0.0) * end_prob.get(
            (span.chunk_id, end_idx), 0.0
        )
    return float(-np.log(max(mass, 1e-12)))
