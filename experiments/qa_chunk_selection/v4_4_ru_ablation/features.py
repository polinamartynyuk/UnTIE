"""Generalises the V4.4 confidence core to K=3 while preserving K=4/5 behaviour."""

from itertools import combinations
import math

from experiments.qa_chunk_selection.v4.answers import answers_match, majority_answer
from experiments.qa_chunk_selection.v4_2.features import zscore_map
from experiments.qa_chunk_selection.v4_3.features import similarity_matrix, sim_at, sem_top2
from experiments.qa_chunk_selection.v4_3.text import is_valid_answer


def adaptive_semantic_pack(answers, embeddings, top_ids):
    valid = [int(cid) for cid in top_ids if is_valid_answer(answers.get(int(cid), ""))]
    matrix = similarity_matrix(embeddings, valid) if len(valid) >= 2 else {}
    top2 = {cid: sem_top2(matrix, cid, valid) for cid in valid} if len(valid) >= 3 else {}
    rel = zscore_map({cid: value for cid, value in top2.items() if value is not None})
    core_size = 3 if len(valid) >= 4 else (2 if len(valid) == 3 else 0)
    best = None
    if core_size:
        for core in combinations(sorted(valid), core_size):
            inside = [sim_at(matrix, a, b) for i, a in enumerate(core) for b in core[i + 1:]]
            outside = [cid for cid in valid if cid not in core]
            cross = [sim_at(matrix, a, b) for a in core for b in outside]
            cohesion = sum(inside) / len(inside)
            between = sum(cross) / len(cross)
            candidate = (cohesion - between, cohesion, tuple(core))
            if best is None or candidate[:2] > best[:2] or (candidate[:2] == best[:2] and candidate[2] < best[2]):
                best = candidate
    return {"valid_ids": valid, "n_valid": len(valid),
            "sem_rel": {str(cid): float(rel.get(cid, 0.0)) for cid in top_ids},
            "best_core": list(best[2]) if best else [],
            "consensus_conf": float(best[0]) if best else None,
            "cohesion": float(best[1]) if best else None,
            "core_size": core_size,
            "confidence_definition": "best coherent triple-vs-outside for K>=4; pair-vs-outside for K=3"}


def routing_available(pack, selected_id):
    value = pack.get("consensus_conf")
    return bool(value is not None and math.isfinite(float(value)) and
                int(selected_id) in {int(x) for x in pack.get("best_core", [])})


def lemma_features(answers, top_ids, lemmatizer):
    values = [answers.get(int(cid), "") for cid in top_ids]
    keys = [lemmatizer.key(x) for x in values]
    counts = {key: keys.count(key) for key in keys if key}
    majority = max(counts, key=lambda x: (counts[x], -len(x))) if counts else ""
    out = {}
    for cid, answer, key in zip(top_ids, values, keys):
        other = [value for oid, value in zip(top_ids, values) if int(oid) != int(cid)]
        scores = [lemmatizer.f1(answer, value) for value in other]
        out[int(cid)] = {"lemma_majority": float(bool(key and key == majority)),
                         "mean_lemma_f1": sum(scores) / len(scores) if scores else 0.0}
    return out

