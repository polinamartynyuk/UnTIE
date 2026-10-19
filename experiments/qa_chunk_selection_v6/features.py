"""Build leakage-separated V6 runtime features and gold-only training targets."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re

import numpy as np

from experiments.qa_chunk_selection.v4.answers import pair_token_f1

from .contract import ASPECTS, ASPECT_MAP, FAMILIES, LANGUAGES

_SPACE = re.compile(r"\s+")


def canonical_aspect(value: str) -> str:
    return ASPECT_MAP.get(str(value).strip().lower(), "OTHER")


def canonical_document_id(cell: str, document_id: str) -> str:
    value = str(document_id)
    return f"scier::{value.removeprefix('scier:')}" if cell.startswith("scier_") else f"{FAMILIES[cell]}::{value}"


def normalized_document_hash(item) -> str:
    chunks = sorted(item.chunks, key=lambda x: (int(x.char_start), int(x.chunk_id)))
    length = max((int(chunk.char_end) for chunk in chunks), default=0)
    document = [None] * length
    for chunk in chunks:
        start = int(chunk.char_start)
        for offset, char in enumerate(str(chunk.text)):
            position = start + offset
            if position >= length:
                break
            if document[position] is not None and document[position] != char:
                raise ValueError(f"inconsistent overlapping chunks for {item.document_id} at char {position}")
            document[position] = char
    text = "".join(char if char is not None else " " for char in document)
    normalized = _SPACE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row["question_id"])].append(row)
    return {key: sorted(value, key=lambda r: (int(r["b0_rank"]), int(r["chunk_id"]))) for key, value in out.items()}


def _zscores(values: list[float]) -> tuple[list[float], bool]:
    x = np.asarray(values, dtype=float)
    scale = float(x.std())
    if scale < 1e-6:
        return [0.0] * len(values), True
    return [float(v) for v in ((x - float(x.mean())) / scale)], False


def _lexical(answers: list[str]) -> list[float]:
    values = []
    for i, answer in enumerate(answers):
        other = [pair_token_f1(answer, value) / 100.0 for j, value in enumerate(answers) if i != j]
        values.append(float(np.mean(other)) if other else 0.0)
    return values


def _f1(answer: str, gold_answers) -> float:
    return max((pair_token_f1(answer, gold) for gold in gold_answers), default=0.0)


def enrich_question(cell: str, rows: list[dict], item, *, outer_fold: int, seed: int,
                    meta_source: bool) -> tuple[list[dict], dict]:
    """Return candidate runtime rows and a separate question target object."""
    rows = sorted(rows, key=lambda r: int(r["b0_rank"]))
    if not rows:
        raise ValueError("empty question shortlist")
    top = rows[0]
    scores = [float(r["b0_score_raw"]) for r in rows]
    b0_z, flat = _zscores(scores)
    qarel_raw = [float(r["q_answer_ranker_raw"]) for r in rows]
    valid = [int(bool(str(r.get("qa_answer") or "").strip())) for r in rows]
    finite = [v for v, ok in zip(qarel_raw, valid) if ok and math.isfinite(v)]
    missing = min(finite) if finite else 0.0
    qarel_raw = [v if ok and math.isfinite(v) else missing for v, ok in zip(qarel_raw, valid)]
    qarel_z, _ = _zscores(qarel_raw)
    source_z = [float(r.get("source_support_norm", 0.0)) for r in rows]
    lex_z, _ = _zscores(_lexical([str(r.get("qa_answer") or "") for r in rows]))
    k = len(rows)
    scale = max(float(np.std(scores)), 1e-6)
    gap12 = float((scores[0] - scores[1]) / scale) if k >= 2 else 0.0
    gap13 = float((scores[0] - scores[2]) / scale) if k >= 3 else 0.0
    z = np.asarray(b0_z)
    exp = np.exp(z - z.max()); probabilities = exp / exp.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum() / math.log(k)) if k > 1 else 0.0
    aspect = canonical_aspect(str(top["aspect_id"]))
    canonical_doc = canonical_document_id(cell, str(item.document_id))
    doc_hash = normalized_document_hash(item)
    lineage = hashlib.sha256(f"{FAMILIES[cell]}|{canonical_doc}|{aspect}".encode()).hexdigest()
    f1s = [_f1(str(r.get("qa_answer") or ""), item.gold_answers) for r in rows]
    runtime_rows = []
    for index, source in enumerate(rows):
        runtime = {
            "dataset_cell": cell, "corpus_family": FAMILIES[cell], "language": LANGUAGES[cell],
            "question_id": str(source["question_id"]), "document_id": str(item.document_id),
            "canonical_document_id": canonical_doc, "document_sha256": doc_hash,
            "question_lineage_id": lineage, "canonical_aspect": aspect,
            "outer_fold": int(outer_fold), "seed": int(seed),
            "upstream_inner_fold": source.get("inner_validation_fold") if meta_source else None,
            "chunk_id": int(source["chunk_id"]), "b0_rank": int(source["b0_rank"]),
            "b0_score_raw": float(source["b0_score_raw"]), "b0_score_norm": float(b0_z[index]),
            "qarel_raw": float(qarel_raw[index]), "qarel_norm": float(qarel_z[index]),
            "qa_answer": str(source.get("qa_answer") or ""), "answer_valid": valid[index],
            "source_norm": float(source_z[index]), "lex_norm": float(lex_z[index]),
            "sem_norm": 0.0, "qaconf_norm": 0.0,
            "source_available": 1, "lex_available": 1, "sem_available": 0, "qa_conf_available": 0,
            "effective_k": k, "candidate_count": int(top.get("candidate_count", k)),
            "gap12_norm": gap12, "gap13_norm": gap13, "top1_z": float(b0_z[0]),
            "b0_entropy_norm": entropy, "flat_score_vector": int(flat), "has_rank3": int(k >= 3),
            "upstream_feature_origin": "inner_oof" if meta_source else "outer_train_cal_refit",
        }
        runtime_rows.append(runtime)
    target = {
        "question_id": str(top["question_id"]), "baseline_f1": float(f1s[0]),
        "candidate_f1": {str(r["chunk_id"]): float(f1) for r, f1 in zip(rows, f1s)},
        "label_tier": {str(r["chunk_id"]): str(r.get("label_tier", "unresolved_for_training")) for r in rows},
        "hard_negative": {str(r["chunk_id"]): int(str(item.chunk_by_id(int(r["chunk_id"])).label) == "hard_negative") for r in rows},
        "gate_target": int(any(value > f1s[0] for value in f1s[1:])),
    }
    return runtime_rows, target


def alternative_rows(runtime_rows: list[dict], target: dict, *, variant: str) -> list[dict]:
    if len(runtime_rows) <= 1:
        return []
    top = runtime_rows[0]
    out = []
    for alt in runtime_rows[1:]:
        aspect = str(alt["canonical_aspect"])
        row = {
            "question_id": alt["question_id"], "document_id": alt["document_id"],
            "canonical_document_id": alt["canonical_document_id"], "document_sha256": alt["document_sha256"],
            "question_lineage_id": alt["question_lineage_id"], "dataset_cell": alt["dataset_cell"],
            "corpus_family": alt["corpus_family"], "language": alt["language"], "canonical_aspect": aspect,
            "outer_fold": alt["outer_fold"], "seed": alt["seed"], "chunk_id": alt["chunk_id"],
            "b0_chunk_id": top["chunk_id"], "effective_k": alt["effective_k"],
            "candidate_count": alt["candidate_count"], "gap12_norm": alt["gap12_norm"],
            "gap13_norm": alt["gap13_norm"], "top1_z": alt["top1_z"],
            "b0_entropy_norm": alt["b0_entropy_norm"], "flat_score_vector": alt["flat_score_vector"],
            "has_rank3": alt["has_rank3"], "delta_answer_valid": alt["answer_valid"] - top["answer_valid"],
            "delta_b0": alt["b0_score_norm"] - top["b0_score_norm"],
            "delta_qarel": alt["qarel_norm"] - top["qarel_norm"],
            "delta_source": alt["source_norm"] - top["source_norm"],
            "delta_lex": alt["lex_norm"] - top["lex_norm"],
            "delta_sem": alt["sem_norm"] - top["sem_norm"],
            "delta_qaconf": alt["qaconf_norm"] - top["qaconf_norm"],
            "source_available": alt["source_available"], "lex_available": alt["lex_available"],
            "sem_available": alt["sem_available"], "qa_conf_available": alt["qa_conf_available"],
            "candidate_f1": float(target["candidate_f1"][str(alt["chunk_id"])]),
            "baseline_f1": float(target["baseline_f1"]),
            "candidate_tier": target["label_tier"][str(alt["chunk_id"])],
            "baseline_tier": target["label_tier"][str(top["chunk_id"])],
            "candidate_hn": target["hard_negative"][str(alt["chunk_id"])],
            "baseline_hn": target["hard_negative"][str(top["chunk_id"])],
            "target": int(float(target["candidate_f1"][str(alt["chunk_id"])]) > float(target["baseline_f1"])),
            "utility": float(target["candidate_f1"][str(alt["chunk_id"])]) - float(target["baseline_f1"]),
            "weight": 1.0 / (len(runtime_rows) - 1),
            "qa_answer": alt["qa_answer"], "baseline_answer": top["qa_answer"],
            "upstream_feature_origin": alt["upstream_feature_origin"],
        }
        row.update({
            "qarel_x_k": row["delta_qarel"] * row["effective_k"],
            "qarel_x_margin": row["delta_qarel"] * row["gap12_norm"],
            "b0_x_margin": row["delta_b0"] * row["gap12_norm"],
            "qarel_x_entropy": row["delta_qarel"] * row["b0_entropy_norm"],
            "b0_x_qarel": row["delta_b0"] * row["delta_qarel"],
            "source_x_k": row["delta_source"] * row["effective_k"],
            "source_x_margin": row["delta_source"] * row["gap12_norm"],
            "lex_x_k": row["delta_lex"] * row["effective_k"],
            "lex_x_margin": row["delta_lex"] * row["gap12_norm"],
        })
        for name in ASPECTS:
            row[f"qarel_x_aspect__{name}"] = row["delta_qarel"] * int(aspect == name)
            row[f"source_x_aspect__{name}"] = row["delta_source"] * int(aspect == name)
        out.append(row)
    return out


def evidence_pair_rows(runtime_rows: list[dict], target: dict) -> list[dict]:
    """Exact V5 mirrored evidence pair contract, augmented only by fixed interactions."""
    sufficient = [r for r in runtime_rows if target["label_tier"][str(r["chunk_id"])] == "sufficient_positive"]
    insufficient = [r for r in runtime_rows if target["label_tier"][str(r["chunk_id"])] == "insufficient_evidence"]
    if not sufficient or not insufficient:
        return []
    pairs = []
    unit = 1.0 / (len(sufficient) * len(insufficient) * 2.0)
    for pos in sufficient:
        for neg in insufficient:
            for left, right, label in ((pos, neg, 1), (neg, pos, 0)):
                fake_target = dict(target)
                fake_target["baseline_f1"] = 0.0
                fake_target["candidate_f1"] = {str(left["chunk_id"]): float(label), str(right["chunk_id"]): 0.0}
                row = alternative_rows([right, left], fake_target, variant="S1")[0]
                row.update(target=int(label), weight=unit, candidate_tier="sufficient_positive" if label else "insufficient_evidence")
                pairs.append(row)
    return pairs


def gate_row(runtime_rows: list[dict], target: dict) -> dict:
    first = runtime_rows[0]
    keys = ("question_id", "document_id", "canonical_document_id", "document_sha256", "question_lineage_id",
            "dataset_cell", "corpus_family", "language", "canonical_aspect", "outer_fold", "seed",
            "effective_k", "candidate_count", "gap12_norm", "gap13_norm", "has_rank3", "top1_z",
            "b0_entropy_norm", "flat_score_vector")
    return {key: first[key] for key in keys} | {"target": int(target["gate_target"]), "weight": 1.0}


GOLD_FORBIDDEN = ("gold", "target", "candidate_f1", "baseline_f1", "label_tier", "hard_negative", "oracle_rank", "utility_target")


def assert_runtime_gold_free(runtime: dict) -> None:
    for key in runtime:
        lowered = key.lower()
        if any(token in lowered for token in GOLD_FORBIDDEN):
            raise AssertionError(f"gold-dependent field leaked into runtime: {key}")

