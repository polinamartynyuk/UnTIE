"""Compact selected-head features from frozen extractive QA. Raw tensors are not saved."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.qa_chunk_selection.v3.attention import span_token_mask, synthetic_attention, token_masks_from_encoding

from .contract import ATTENTION_FEATURE_VERSION, Z_EPS
from .features import compact_head_features, content_masks, token_word_key


def _decode_token_span(offsets, start_idx: int, end_idx: int, question: str, context: str, sequence_ids) -> str:
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    if start_idx < 0 or end_idx >= len(offsets):
        return ""
    sid = sequence_ids[int(start_idx)] if start_idx < len(sequence_ids) else None
    text = question if sid == 0 else context if sid == 1 else ""
    if not text:
        return ""
    char_start = int(offsets[int(start_idx)][0])
    char_end = int(offsets[int(end_idx)][1])
    if char_end <= char_start:
        return ""
    return text[char_start:char_end]


def _enable_eager(model) -> str:
    backend = str(getattr(getattr(model, "config", None), "_attn_implementation", "") or "")
    try:
        if hasattr(model, "config"):
            model.config._attn_implementation = "eager"
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "eager"
        return "eager"
    except Exception:
        return backend or "unknown"


def extract_compact_attention(
    answerer,
    question: str,
    context: str,
    *,
    predicted_start: int,
    predicted_end: int,
    predicted_answer: str = "",
    eager: bool = False,
) -> dict[str, Any]:
    import torch

    if eager:
        _enable_eager(answerer.model)
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
    encoded.pop("special_tokens_mask", None)
    model_inputs = {key: value.to(answerer.device) for key, value in encoded.items()}
    answerer.model.eval()
    with torch.no_grad():
        off = answerer.model(**model_inputs, output_attentions=False)
        on = answerer.model(**model_inputs, output_attentions=True)
    start_off = off.start_logits[0].detach().cpu().numpy()
    end_off = off.end_logits[0].detach().cpu().numpy()
    start_on = on.start_logits[0].detach().cpu().numpy()
    end_on = on.end_logits[0].detach().cpu().numpy()
    if not getattr(on, "attentions", None):
        _enable_eager(answerer.model)
        with torch.no_grad():
            on = answerer.model(**model_inputs, output_attentions=True)
    if not getattr(on, "attentions", None):
        raise RuntimeError("QA backend returned no attentions even after eager fallback")
    attn = np.stack([layer[0].detach().cpu().numpy() for layer in on.attentions], axis=0)
    del on, off
    q, c = content_masks(masks.question, masks.context, masks.special)
    span = span_token_mask(masks, predicted_start, predicted_end) & c
    q_idx = np.where(q)[0]
    c_idx = np.where(c)[0]
    s_idx = np.where(span)[0]
    payload = compact_head_features(attn, q_idx=q_idx, c_idx=c_idx, s_idx=s_idx)
    span_words = []
    offset_pairs = masks.offsets
    for index in s_idx:
        start, end = offset_pairs[int(index)]
        span_words.append(token_word_key(context, start, end))
    payload["span_word_keys"] = span_words
    payload["valid_span"] = bool(len(s_idx) > 0 and str(predicted_answer or "").strip())
    answer_off = _decode_token_span(offset_pairs, int(start_off.argmax()), int(end_off.argmax()), question, context, masks.sequence_ids)
    answer_on = _decode_token_span(offset_pairs, int(start_on.argmax()), int(end_on.argmax()), question, context, masks.sequence_ids)
    payload["logit_regression"] = {
        "start_max_abs": float(np.max(np.abs(start_off - start_on))),
        "end_max_abs": float(np.max(np.abs(end_off - end_on))),
        "identical_argmax_start": bool(int(start_off.argmax()) == int(start_on.argmax())),
        "identical_argmax_end": bool(int(end_off.argmax()) == int(end_on.argmax())),
        "identical_answer_text": answer_off == answer_on,
        "predicted_start_off": int(start_off.argmax()),
        "predicted_start_on": int(start_on.argmax()),
        "predicted_end_off": int(end_off.argmax()),
        "predicted_end_on": int(end_on.argmax()),
        "predicted_answer_off": answer_off,
        "predicted_answer_on": answer_on,
    }
    payload["feature_version"] = ATTENTION_FEATURE_VERSION
    payload["attention_shape"] = list(attn.shape)
    del attn
    return payload


def synthetic_compact(
    question: str,
    context: str,
    *,
    predicted_start: int,
    predicted_end: int,
    seed: int = 0,
    n_layers: int = 4,
    n_heads: int = 4,
) -> dict[str, Any]:
    q_tokens = max(1, len(str(question).split()))
    c_tokens = max(2, len(str(context).split()))
    n = min(24, max(8, q_tokens + min(c_tokens, 12)))
    q_len = min(q_tokens, n // 3)
    attn = synthetic_attention(n_layers, n_heads, n, seed=int(seed) % (10**6))
    q = np.zeros(n, dtype=bool)
    c = np.zeros(n, dtype=bool)
    spec = np.zeros(n, dtype=bool)
    q[:q_len] = True
    spec[q_len] = True
    c[q_len + 1 :] = True
    s = np.zeros(n, dtype=bool)
    start = min(n - 1, q_len + 1 + max(0, predicted_start) % max(n - q_len - 1, 1))
    width = max(1, min(3, (predicted_end - predicted_start) if predicted_end > predicted_start else 1))
    s[start : min(n, start + width)] = True
    s &= c
    payload = compact_head_features(attn, q_idx=np.where(q)[0], c_idx=np.where(c)[0], s_idx=np.where(s)[0])
    payload["span_word_keys"] = [token_word_key(context, 0, min(4, len(context)))] * int(s.sum())
    payload["valid_span"] = bool(int(s.sum()) > 0)
    payload["logit_regression"] = {
        "start_max_abs": 0.0,
        "end_max_abs": 0.0,
        "identical_argmax_start": True,
        "identical_argmax_end": True,
        "identical_answer_text": True,
    }
    payload["feature_version"] = ATTENTION_FEATURE_VERSION
    payload["attention_shape"] = list(attn.shape)
    payload["synthetic"] = True
    return payload


def logits_close(payload: dict[str, Any], *, atol: float = 1e-4) -> bool:
    row = payload.get("logit_regression") or {}
    return (
        float(row.get("start_max_abs") or 0.0) <= atol
        and float(row.get("end_max_abs") or 0.0) <= atol
        and bool(row.get("identical_argmax_start"))
        and bool(row.get("identical_argmax_end"))
        and bool(row.get("identical_answer_text", True))
    )


def span_length_bias_probe(*, seed: int = 0) -> dict[str, Any]:
    """Density/enrichment must not grow linearly with |S| under context-uniform attention."""
    n, layers, heads = 16, 2, 2
    q_idx = np.arange(0, 4)
    c_idx = np.arange(5, 16)
    attn = np.zeros((layers, heads, n, n), dtype=float)
    attn[:, :, :, c_idx] = 1.0 / float(len(c_idx))
    short = compact_head_features(attn, q_idx=q_idx, c_idx=c_idx, s_idx=np.arange(5, 7))
    long = compact_head_features(attn, q_idx=q_idx, c_idx=c_idx, s_idx=np.arange(5, 13))
    dens_short = float(np.mean(short["q_to_span_density"]))
    dens_long = float(np.mean(long["q_to_span_density"]))
    mass_short = float(np.mean(short["q_to_span_mass_raw"]))
    mass_long = float(np.mean(long["q_to_span_mass_raw"]))
    enr_short = float(np.mean(short["q_to_span_enrichment"]))
    enr_long = float(np.mean(long["q_to_span_enrichment"]))
    return {
        "density_short": dens_short,
        "density_long": dens_long,
        "mass_short": mass_short,
        "mass_long": mass_long,
        "enrichment_short": enr_short,
        "enrichment_long": enr_long,
        "mass_grows_with_span": mass_long > mass_short + Z_EPS,
        "density_not_proportional_to_span": abs(dens_long - dens_short) < abs(mass_long - mass_short),
        "enrichment_near_uniform": abs(enr_short - 1.0) < 0.15 and abs(enr_long - 1.0) < 0.15,
        "ok": abs(dens_long - dens_short) < abs(mass_long - mass_short) and abs(enr_short - 1.0) < 0.15,
        "seed": seed,
    }
