"""Reuse V4.3 sentence extraction, enforcing V4.4's containing-span fallback."""
import re
from experiments.qa_chunk_selection.v4_3.text import local_evidence_sentence as _sentence


def local_evidence_sentence(chunk_text, *, start_char, end_char, predicted_answer="", token_window=64):
    result = _sentence(chunk_text, start_char=start_char, end_char=end_char,
                       predicted_answer=predicted_answer, token_window=token_window)
    if result["mode"] != "sentence" or result["span_in_sentence"]:
        return result
    # V4.3 also returns a merely overlapping sentence. For a cross-sentence span
    # V4.4 explicitly requires the complete predicted span plus its token window.
    start = max(0, min(int(start_char or 0), len(chunk_text)))
    end = max(start, min(int(end_char or 0), len(chunk_text)))
    tokens = list(re.finditer(r"\S+", chunk_text))
    overlap = [i for i,t in enumerate(tokens) if t.start() < end and t.end() > start]
    if not overlap:
        return {"text": "", "mode": "empty", "span_in_sentence": False, "start_char": start, "end_char": end}
    lo, hi = max(0, overlap[0]-token_window), min(len(tokens)-1, overlap[-1]+token_window)
    left, right = tokens[lo].start(), tokens[hi].end()
    return {"text": chunk_text[left:right], "mode": "token_window", "span_in_sentence": False,
            "span_in_window": True, "start_char": left, "end_char": right, "token_window": token_window}
