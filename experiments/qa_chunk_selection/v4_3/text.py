"""Answer/query/evidence text for V4.3 semantic features. No gold."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from experiments.qa_chunk_selection.v4.answers import normalize_predicted_answer

from .contract import CONTEXT_TOKEN_WINDOW

_WS = re.compile(r"\s+", re.UNICODE)
_WORDISH = re.compile(r"[\w]", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def semantic_normalize(text: str) -> str:
    raw = unicodedata.normalize("NFC", str(text or ""))
    raw = raw.strip()
    return _WS.sub(" ", raw)


def lexical_normalize(text: str) -> str:
    return normalize_predicted_answer(str(text or ""))


def is_valid_answer(text: str) -> bool:
    norm = semantic_normalize(text)
    if not norm:
        return False
    return bool(_WORDISH.search(norm))


def cosine(left, right, *, eps: float = 1e-8) -> float:
    import numpy as np

    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < float(eps):
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_normalize(vector, *, eps: float = 1e-8):
    import numpy as np

    arr = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < float(eps):
        return arr
    return arr / (norm + float(eps))


def _sentences(text: str) -> list[tuple[int, int, str]]:
    raw = str(text or "")
    if not raw:
        return []
    parts = _SENT_SPLIT.split(raw)
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for part in parts:
        if not part:
            continue
        start = raw.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        out.append((start, end, part))
        cursor = end
    if not out:
        out.append((0, len(raw), raw))
    return out


def local_evidence_sentence(
    chunk_text: str,
    *,
    start_char: int,
    end_char: int,
    predicted_answer: str = "",
    token_window: int = CONTEXT_TOKEN_WINDOW,
) -> dict[str, Any]:
    text = str(chunk_text or "")
    answer = str(predicted_answer or "")
    start = int(start_char or 0)
    end = int(end_char or 0)
    if end < start:
        start, end = end, start
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    if start == end and answer:
        found = text.find(answer)
        if found >= 0:
            start, end = found, found + len(answer)
    contained = []
    overlapping = []
    for sent_start, sent_end, sent in _sentences(text):
        if sent_start <= start and end <= sent_end and end > start:
            contained.append((sent_start, sent_end, sent))
        elif not (end <= sent_start or start >= sent_end):
            overlapping.append((sent_start, sent_end, sent))
    chosen = contained[0] if contained else (overlapping[0] if overlapping else None)
    if chosen is not None:
        sent_start, sent_end, sent = chosen
        return {
            "text": sent.strip() or sent,
            "mode": "sentence",
            "span_in_sentence": bool(sent_start <= start and end <= sent_end) or (bool(answer) and answer in sent),
            "start_char": int(sent_start),
            "end_char": int(sent_end),
        }
    tokens = text.split()
    if not tokens:
        return {"text": "", "mode": "empty", "span_in_sentence": False, "start_char": 0, "end_char": 0}
    prefix = text[:start]
    span = text[start:end] or answer
    start_tok = len(prefix.split())
    n_span = max(len(span.split()), 1)
    lo = max(0, start_tok - int(token_window))
    hi = min(len(tokens), start_tok + n_span + int(token_window))
    window = " ".join(tokens[lo:hi])
    return {
        "text": window,
        "mode": "token_window",
        "span_in_sentence": bool(span and span in window),
        "start_char": start,
        "end_char": end,
        "token_window": int(token_window),
    }


def query_answer_text(question: str, answer: str) -> str:
    q = semantic_normalize(question)
    a = semantic_normalize(answer)
    return f"{q} {a}".strip()


def answer_record(raw: str) -> dict[str, Any]:
    text = str(raw or "")
    return {
        "raw": text,
        "semantic_norm": semantic_normalize(text),
        "lexical_norm": lexical_normalize(text),
        "valid": is_valid_answer(text),
    }


def qa_answer_text(qa: Mapping[str, Any] | None) -> str:
    return str((qa or {}).get("qa_answer") or "")
