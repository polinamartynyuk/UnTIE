"""Strict pairing around the established V4.1 evaluation implementation."""
from experiments.qa_chunk_selection.v4_1.evaluate import (
    paired_delta as _paired_delta, row_key, summarize_rows, transitions,
)


def paired_delta(baseline, candidate, **kwargs):
    left, right = [row_key(r) for r in baseline], [row_key(r) for r in candidate]
    if len(set(left)) != len(left) or len(set(right)) != len(right) or set(left) != set(right):
        raise ValueError("Incomplete or duplicate paired development rows")
    docs = {row_key(r): r["document_id"] for r in baseline}
    if any(docs[row_key(r)] != r["document_id"] for r in candidate):
        raise ValueError("Paired document_id mismatch")
    result = _paired_delta(baseline, candidate, **kwargs)
    if result["n_rows"] != len(candidate):
        raise ValueError("Missing endpoint in paired rows")
    return {**result, "interval_type": "development CI; repeated-use selection bias remains"}


def identity(left, right):
    a, b = {row_key(r): r for r in left}, {row_key(r): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Reference identity coverage mismatch")
    bad = []
    for key, row in b.items():
        previous = a[key]
        if (row["selected_chunk_id"] != previous["selected_chunk_id"]
                or row["qa_answer"] != previous["qa_answer"]
                or abs(row["token_f1"] - previous["token_f1"]) > 1e-6):
            bad.append(list(key))
    return {"ok": not bad, "n": len(b), "mismatches": bad}
