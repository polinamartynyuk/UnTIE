"""Attention signatures from a frozen extractive reader. Gold-span features are diagnostic-only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.structure import split_sentences
from experiments.qa_chunk_selection.v2.schemas import RuntimeChunk, source_offsets

EPS = 1e-8


@dataclass
class TokenMasks:
    question: np.ndarray
    context: np.ndarray
    special: np.ndarray
    sequence_ids: tuple[Any, ...]
    offsets: tuple[tuple[int, int], ...]


def token_masks_from_encoding(encoded, *, example: int = 0) -> TokenMasks:
    try:
        sequence_ids = encoded.sequence_ids(example)
    except (AttributeError, ValueError):
        sequence_ids = _fallback_sequence_ids(encoded, example)
    special = encoded.get("special_tokens_mask")
    if special is not None:
        spec = np.array(special[example].tolist() if hasattr(special[example], "tolist") else special[example], dtype=bool)
    else:
        spec = np.array([sid is None for sid in sequence_ids], dtype=bool)
    q = np.array([sid == 0 for sid in sequence_ids], dtype=bool)
    c = np.array([sid == 1 for sid in sequence_ids], dtype=bool)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        pairs = ((0, 0),) * len(sequence_ids)
    else:
        raw = offsets[example]
        pairs = tuple((int(a), int(b)) for a, b in (raw.tolist() if hasattr(raw, "tolist") else raw))
    return TokenMasks(question=q, context=c, special=spec, sequence_ids=tuple(sequence_ids), offsets=pairs)


def _fallback_sequence_ids(encoded, example: int) -> list[Any]:
    input_ids = encoded["input_ids"][example].tolist()
    # Fallback only: first sequence until a second SEP-like run. Covered by unit tests.
    ids = list(input_ids)
    seq: list[Any] = []
    seen_none = 0
    mode = 0
    for _index, _token in enumerate(ids):
        # Without tokenizer we cannot know special ids; mark unknown as None around likely pads.
        seq.append(mode if seen_none < 2 else 1)
    return seq


def span_token_mask(masks: TokenMasks, start_char: int, end_char: int) -> np.ndarray:
    flags = np.zeros_like(masks.context, dtype=bool)
    for index, ((start, end), is_ctx) in enumerate(zip(masks.offsets, masks.context)):
        if not is_ctx or end <= start:
            continue
        if start < int(end_char) and end > int(start_char):
            flags[index] = True
    return flags


def compact_signature(
    attention: np.ndarray,
    masks: TokenMasks,
    span_mask: np.ndarray,
    *,
    gold_dependent: bool = False,
) -> dict[str, Any]:
    """attention: [layers, heads, T, T] query-from, key-to."""
    layers, heads, t_q, t_k = attention.shape
    q = masks.question
    c = masks.context
    s = span_mask & c
    n_q = max(int(q.sum()), 1)
    n_s = max(int(s.sum()), 1)
    n_c = max(int(c.sum()), 1)
    q_idx = np.where(q)[0]
    s_idx = np.where(s)[0]
    c_idx = np.where(c)[0]
    features: dict[str, Any] = {
        "n_layers": int(layers),
        "n_heads": int(heads),
        "question_length": int(q.sum()),
        "context_length": int(c.sum()),
        "predicted_span_length": int(s.sum()),
        "gold_dependent": bool(gold_dependent),
        "diagnostic_only": bool(gold_dependent),
    }
    q_to_s = np.zeros((layers, heads), dtype=float)
    s_to_q = np.zeros((layers, heads), dtype=float)
    q_to_s_ratio = np.zeros((layers, heads), dtype=float)
    entropy = np.zeros((layers, heads), dtype=float)
    entropy_norm = np.zeros((layers, heads), dtype=float)
    top1_mass = np.zeros((layers, heads), dtype=float)
    for layer in range(layers):
        for head in range(heads):
            A = attention[layer, head]
            mass_qs = float(A[np.ix_(q_idx, s_idx)].sum()) if len(s_idx) else 0.0
            mass_qc = float(A[np.ix_(q_idx, c_idx)].sum()) if len(c_idx) else 0.0
            mass_sq = float(A[np.ix_(s_idx, q_idx)].sum()) if len(s_idx) else 0.0
            q_to_s[layer, head] = mass_qs / float(n_q * n_s)
            s_to_q[layer, head] = mass_sq / float(n_s * n_q)
            q_to_s_ratio[layer, head] = mass_qs / (mass_qc + EPS)
            col = A[np.ix_(q_idx, c_idx)].sum(axis=0) if len(c_idx) else np.zeros(1)
            p = col / (col.sum() + EPS)
            h = float(-(p * np.log(p + EPS)).sum())
            entropy[layer, head] = h
            entropy_norm[layer, head] = h / math.log(max(n_c, 2))
            top1_mass[layer, head] = float(p.max()) if p.size else 0.0
    features["q_to_s_mean"] = _layer_summaries(q_to_s)
    features["s_to_q_mean"] = _layer_summaries(s_to_q)
    features["q_to_s_ratio"] = _layer_summaries(q_to_s_ratio)
    features["entropy_norm"] = _layer_summaries(entropy_norm)
    features["top1_mass"] = _layer_summaries(top1_mass)
    features["head_dispersion_q_to_s"] = {
        "mean": float(q_to_s.mean()),
        "std": float(q_to_s.std()),
        "max": float(q_to_s.max()),
        "top4_mean": float(np.sort(q_to_s.ravel())[-min(4, q_to_s.size) :].mean()),
    }
    features["q_to_s_raw_mean"] = float(q_to_s.mean() * n_q * n_s)
    features["q_to_s_matrix"] = q_to_s.tolist()
    features["vector"] = _vectorize(features)
    return features


def _layer_summaries(matrix: np.ndarray) -> dict[str, float]:
    n = matrix.shape[0]
    early = matrix[: max(n // 3, 1)]
    mid = matrix[n // 3 : max(2 * n // 3, n // 3 + 1)]
    late = matrix[max(2 * n // 3, 1) :]
    per_layer = matrix.mean(axis=1)
    return {
        "early": float(early.mean()),
        "middle": float(mid.mean()) if mid.size else float(early.mean()),
        "late": float(late.mean()) if late.size else float(per_layer[-1]),
        "late_minus_early": float((late.mean() if late.size else per_layer[-1]) - early.mean()),
        "max": float(matrix.max()),
        "argmax_layer": int(per_layer.argmax()),
        "all_head_mean": float(matrix.mean()),
        "last_layer_mean": float(matrix[-1].mean()),
        "middle_layer_mean": float(mid.mean()) if mid.size else float(matrix[n // 2].mean()),
    }


def _vectorize(features: Mapping[str, Any]) -> list[float]:
    keys = ("q_to_s_mean", "s_to_q_mean", "q_to_s_ratio", "entropy_norm", "top1_mass")
    fields = ("early", "middle", "late", "late_minus_early", "max", "all_head_mean", "last_layer_mean", "middle_layer_mean")
    values = [
        float(features["question_length"]),
        float(features["context_length"]),
        float(features["predicted_span_length"]),
        float(features["head_dispersion_q_to_s"]["mean"]),
        float(features["head_dispersion_q_to_s"]["std"]),
        float(features["head_dispersion_q_to_s"]["max"]),
    ]
    for key in keys:
        for field in fields:
            values.append(float(features[key][field]))
    return values


def competing_region_margin(
    attention: np.ndarray,
    masks: TokenMasks,
    span_mask: np.ndarray,
    context_text: str,
) -> dict[str, float]:
    sentences = split_sentences(context_text) or [(0, len(context_text), context_text)]
    q_idx = np.where(masks.question)[0]
    region_scores = []
    for start, end, _text in sentences:
        region = span_token_mask(masks, start, end)
        idx = np.where(region)[0]
        if not len(idx):
            region_scores.append(0.0)
            continue
        vals = []
        for layer in range(attention.shape[0]):
            for head in range(attention.shape[1]):
                vals.append(float(attention[layer, head][np.ix_(q_idx, idx)].mean()) if len(q_idx) else 0.0)
        region_scores.append(float(np.mean(vals) if vals else 0.0))
    span_score = 0.0
    s_idx = np.where(span_mask & masks.context)[0]
    if len(s_idx) and len(q_idx):
        vals = []
        for layer in range(attention.shape[0]):
            for head in range(attention.shape[1]):
                vals.append(float(attention[layer, head][np.ix_(q_idx, s_idx)].mean()))
        span_score = float(np.mean(vals) if vals else 0.0)
    strongest = max(region_scores) if region_scores else 0.0
    ordered = sorted(region_scores, reverse=True)
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "span_region_score": span_score,
        "strongest_competitor": strongest,
        "margin": float(span_score - strongest),
        "top1_top2_ratio": float(strongest / (top2 + EPS)),
        "n_regions": float(len(region_scores)),
    }


def synthetic_attention(n_layers: int, n_heads: int, n_tokens: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_layers, n_heads, n_tokens, n_tokens))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    return raw.astype(float)


def extract_reader_attention(
    answerer,
    question: str,
    context: str,
    *,
    predicted_start: int,
    predicted_end: int,
) -> dict[str, Any]:
    import torch

    encoded = answerer.tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=getattr(answerer, "max_length", 512),
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    masks = token_masks_from_encoding(encoded)
    offsets = encoded.pop("offset_mapping")
    special = encoded.pop("special_tokens_mask", None)
    del offsets, special
    model_inputs = {key: value.to(answerer.device) for key, value in encoded.items()}
    answerer.model.eval()
    with torch.no_grad():
        off = answerer.model(**model_inputs, output_attentions=False)
        on = answerer.model(**model_inputs, output_attentions=True)
    start_off = off.start_logits[0].detach().cpu().numpy()
    end_off = off.end_logits[0].detach().cpu().numpy()
    start_on = on.start_logits[0].detach().cpu().numpy()
    end_on = on.end_logits[0].detach().cpu().numpy()
    attn = np.stack([layer[0].detach().cpu().numpy() for layer in on.attentions], axis=0)
    span = span_token_mask(masks, predicted_start, predicted_end)
    sig = compact_signature(attn, masks, span, gold_dependent=False)
    sig["logit_regression"] = {
        "start_max_abs": float(np.max(np.abs(start_off - start_on))),
        "end_max_abs": float(np.max(np.abs(end_off - end_on))),
        "identical_argmax_start": bool(int(start_off.argmax()) == int(start_on.argmax())),
        "identical_argmax_end": bool(int(end_off.argmax()) == int(end_on.argmax())),
    }
    return {"signature": sig, "masks": masks, "attention_shape": list(attn.shape)}


def control_features(
    *,
    question_length: int,
    context_length: int,
    span_length: int,
    span_relative_position: float,
    chunk_rank: int,
    qa_span_score: float,
) -> list[float]:
    return [
        float(question_length),
        float(context_length),
        float(span_length),
        float(span_relative_position),
        float(chunk_rank),
        float(qa_span_score),
    ]


def fit_scaler(vectors: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    matrix = np.array(list(vectors), dtype=float)
    if matrix.size == 0:
        return {"mean": [], "std": []}
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_scaler(vector: Sequence[float], scaler: Mapping[str, Sequence[float]]) -> list[float]:
    mean = np.array(scaler.get("mean") or [], dtype=float)
    std = np.array(scaler.get("std") or [], dtype=float)
    arr = np.array(vector, dtype=float)
    if mean.size == 0:
        return [float(x) for x in arr]
    n = min(len(arr), len(mean), len(std))
    return ((arr[:n] - mean[:n]) / std[:n]).tolist()


def prototype_score(vector: Sequence[float], prototypes: Mapping[str, Sequence[float]]) -> dict[str, float]:
    z = np.array(vector, dtype=float)
    out = {}
    for name, proto in prototypes.items():
        p = np.array(proto, dtype=float)
        n = min(len(z), len(p))
        if n == 0:
            out[f"cosine_{name}"] = 0.0
            continue
        a, b = z[:n], p[:n]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + EPS)
        out[f"cosine_{name}"] = float(np.dot(a, b) / denom)
    out["distance_to_success"] = 1.0 - out.get("cosine_success", 0.0)
    out["distance_to_mismatch"] = 1.0 - out.get("cosine_mismatch", 0.0)
    out["success_minus_mismatch"] = out.get("cosine_success", 0.0) - out.get("cosine_mismatch", 0.0)
    return out


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    from experiments.qa_chunk_selection.metrics import binary_roc_auc

    return float(binary_roc_auc(labels, scores))


def smoke_signature(question: str, context: str, *, seed: int = 0) -> dict[str, Any]:
    q_tokens = max(1, len(str(question).split()))
    c_tokens = max(2, len(str(context).split()))
    n = min(24, max(8, q_tokens + min(c_tokens, 12)))
    q_len = min(q_tokens, n // 3)
    s_len = min(2, max(1, n - q_len - 2))
    A = synthetic_attention(4, 4, n, seed=int(seed) % (10**6))
    q = np.zeros(n, dtype=bool)
    c = np.zeros(n, dtype=bool)
    spec = np.zeros(n, dtype=bool)
    q[:q_len] = True
    spec[q_len] = True
    c[q_len + 1 :] = True
    offsets = tuple((i, i + 1) for i in range(n))
    masks = TokenMasks(question=q, context=c, special=spec, sequence_ids=tuple(0 if q[i] else (1 if c[i] else None) for i in range(n)), offsets=offsets)
    span = np.zeros(n, dtype=bool)
    start = q_len + 1
    span[start : start + s_len] = True
    return compact_signature(A, masks, span, gold_dependent=False)
