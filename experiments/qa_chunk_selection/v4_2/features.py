"""Shortlist z-scores, length-normalized attention, shortlist-IDF. No gold."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import Z_EPS

_WORD = re.compile(r"\w+", re.UNICODE)


def zscore_values(values: Sequence[float], *, eps: float = Z_EPS) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=0))
    if sigma < float(eps):
        return [0.0] * int(arr.size)
    return [float((item - mu) / (sigma + float(eps))) for item in arr]


def zscore_map(scores: Mapping[int, float], *, eps: float = Z_EPS) -> dict[int, float]:
    keys = [int(cid) for cid in scores]
    zipped = zscore_values([float(scores[cid]) for cid in keys], eps=eps)
    return {cid: zipped[index] for index, cid in enumerate(keys)}


def content_masks(question: np.ndarray, context: np.ndarray, special: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = np.asarray(special, dtype=bool)
    q = np.asarray(question, dtype=bool) & ~spec
    c = np.asarray(context, dtype=bool) & ~spec
    return q, c


def compact_head_features(
    attention: np.ndarray,
    *,
    q_idx: np.ndarray,
    c_idx: np.ndarray,
    s_idx: np.ndarray,
) -> dict[str, Any]:
    """attention: [layers, heads, T, T] from-token, to-token."""
    layers, heads, _t_q, _t_k = attention.shape
    n_q = int(len(q_idx))
    n_s = int(len(s_idx))
    n_c = int(len(c_idx))
    mass_qs = np.zeros((layers, heads), dtype=float)
    dens_qs = np.zeros((layers, heads), dtype=float)
    enr_qs = np.zeros((layers, heads), dtype=float)
    mass_sq = np.zeros((layers, heads), dtype=float)
    dens_sq = np.zeros((layers, heads), dtype=float)
    entropy = np.zeros((layers, heads), dtype=float)
    token_mean: list[list[list[float]]] = [[[] for _ in range(heads)] for _ in range(layers)]
    uniform = (float(n_s) / float(n_c)) if n_s and n_c else 0.0
    for layer in range(layers):
        for head in range(heads):
            a = attention[layer, head]
            if n_q and n_s:
                qs_mat = a[np.ix_(q_idx, s_idx)]
                qs = float(qs_mat.sum())
                token_mean[layer][head] = qs_mat.mean(axis=0).astype(float).tolist()
                sq = float(a[np.ix_(s_idx, q_idx)].sum())
            else:
                qs = 0.0
                sq = 0.0
            mass_qs[layer, head] = qs
            dens_qs[layer, head] = qs / float(n_q * n_s) if n_q and n_s else 0.0
            enr_qs[layer, head] = (qs / float(n_q)) / (uniform + Z_EPS) if n_q else 0.0
            mass_sq[layer, head] = sq
            dens_sq[layer, head] = sq / float(n_s * n_q) if n_q and n_s else 0.0
            if n_q and n_c:
                col = a[np.ix_(q_idx, c_idx)].sum(axis=0)
                p = col / (col.sum() + Z_EPS)
                entropy[layer, head] = float(-(p * np.log(p + Z_EPS)).sum())
    return {
        "n_layers": int(layers),
        "n_heads": int(heads),
        "question_length_tokens": n_q,
        "context_length_tokens": n_c,
        "span_length_tokens": n_s,
        "valid_span": bool(n_s > 0 and n_q > 0),
        "q_to_span_mass_raw": mass_qs.tolist(),
        "q_to_span_density": dens_qs.tolist(),
        "q_to_span_enrichment": enr_qs.tolist(),
        "q_to_span_token_mean": token_mean,
        "span_to_q_mass_raw": mass_sq.tolist(),
        "span_to_q_density": dens_sq.tolist(),
        "entropy_q_to_context": entropy.tolist(),
    }


def matrix_at(payload: Mapping[str, Any], field: str, layer: int, head: int) -> float:
    matrix = payload.get(field) or []
    try:
        return float(matrix[int(layer)][int(head)])
    except (IndexError, TypeError, ValueError):
        return 0.0


def word_spans(text: str) -> list[tuple[str, int, int]]:
    rows = []
    for match in _WORD.finditer(str(text or "")):
        rows.append((match.group(0).casefold(), match.start(), match.end()))
    return rows


def token_word_key(text: str, start: int, end: int) -> str:
    if end <= start:
        return ""
    piece = str(text)[int(start) : int(end)]
    for word, w_start, w_end in word_spans(text):
        if start < w_end and end > w_start:
            return word
    return piece.casefold().strip()


def chunk_word_set(text: str) -> set[str]:
    return {word for word, _a, _b in word_spans(text)}


def idf_q_to_span_enrichment(payload: Mapping[str, Any], idf: Mapping[str, float]) -> list[list[float]]:
    """M_IDF = mean_q sum_s A(q,s)*idf(word(s)), then same |S|/|C| enrichment as ATT1."""
    n_s = int(payload.get("span_length_tokens") or 0)
    n_c = int(payload.get("context_length_tokens") or 0)
    n_layers = int(payload.get("n_layers") or 0)
    n_heads = int(payload.get("n_heads") or 0)
    keys = list(payload.get("span_word_keys") or [])
    token_mean = payload.get("q_to_span_token_mean") or []
    uniform = (float(n_s) / float(n_c)) if n_s and n_c else 0.0
    out: list[list[float]] = []
    for layer in range(n_layers):
        row: list[float] = []
        for head in range(n_heads):
            masses = []
            if layer < len(token_mean) and head < len(token_mean[layer]):
                masses = token_mean[layer][head] or []
            total = 0.0
            for offset, mass in enumerate(masses):
                word = str(keys[offset]) if offset < len(keys) else ""
                total += float(mass) * float(idf.get(word, 1.0))
            row.append(total / (uniform + Z_EPS) if n_s else 0.0)
        out.append(row)
    return out


def apply_shortlist_idf(payload: Mapping[str, Any], idf: Mapping[str, float]) -> dict[str, Any]:
    clone = dict(payload)
    clone["q_to_span_enrichment"] = idf_q_to_span_enrichment(payload, idf)
    clone["idf_adjusted"] = True
    return clone


def shortlist_idf(chunk_texts: Mapping[int, str], *, k: int) -> dict[str, float]:
    n = max(int(k), 1)
    df: dict[str, int] = {}
    for text in chunk_texts.values():
        for word in chunk_word_set(text):
            df[word] = df.get(word, 0) + 1
    return {word: math.log((n + 1) / (count + 1)) + 1.0 for word, count in df.items()}


def idf_weighted_mass(
    attention: np.ndarray,
    *,
    q_idx: np.ndarray,
    s_idx: np.ndarray,
    span_word_keys: Sequence[str],
    idf: Mapping[str, float],
    layer: int,
    head: int,
) -> float:
    if len(q_idx) == 0 or len(s_idx) == 0:
        return 0.0
    a = attention[int(layer), int(head)]
    total = 0.0
    n_q = float(len(q_idx))
    for offset, s_token in enumerate(s_idx):
        key = span_word_keys[offset] if offset < len(span_word_keys) else ""
        weight = float(idf.get(key, 1.0))
        total += float(a[np.ix_(q_idx, np.array([int(s_token)]))].sum()) * weight
    return total / n_q
