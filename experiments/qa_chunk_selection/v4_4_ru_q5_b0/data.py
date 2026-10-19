from __future__ import annotations

from dataclasses import replace

from experiments.qa_chunk_selection.schema import ChunkRecord, InstanceRecord

LABELS = {
    "robust_positive": "positive", "question_sensitive": "ambiguous",
    "gold_supporting_robust_negative": "ambiguous", "false_negative_candidate": "ambiguous",
    "confirmed_false_negative": "ambiguous", "hard_negative": "hard_negative",
    "easy_negative": "easy_negative",
}


def records_from_results(results, base_records):
    base = {item.question_id: item for item in base_records}
    out = []
    for result in results:
        prior = base[result.qa_instance_id]
        states = {item.chunk_id: item for item in result.chunks}
        chunks = []
        for chunk in prior.chunks:
            state = states[chunk.chunk_id]
            chunks.append(replace(chunk, label=LABELS[state.primary_class], primary_class=state.primary_class))
        out.append(replace(prior, question=result.canonical_question, chunks=tuple(chunks),
                           instance_status=result.instance_status))
    return tuple(out)


def validate(records, expected_qids=None):
    errors = []
    qids = [r.question_id for r in records]
    if len(qids) != len(set(qids)):
        errors.append("duplicate question_id")
    if expected_qids is not None and set(qids) != set(expected_qids):
        errors.append("Q5/base question_id coverage mismatch")
    for row in records:
        if len(row.chunks) == 0:
            errors.append(f"no chunks: {row.question_id}")
        for chunk in row.chunks:
            if chunk.label == "hard_negative" and (chunk.gold_supporting or chunk.primary_class != "hard_negative"):
                errors.append(f"polluted HN: {row.question_id}:{chunk.chunk_id}")
    return errors


def validate_cv(records, cv):
    errors = []
    docs = {r.document_id for r in records}
    if any(not r.positives for r in records):
        errors.append("CV contains unanswerable instance")
    if docs != set(cv["assignment"]):
        errors.append("CV document coverage mismatch")
    for run in cv["runs"]:
        roles = [set(run[k]) for k in ("train", "calibration", "eval")]
        if any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3)):
            errors.append(f"overlapping folds in run {run['run']}")
    return errors

