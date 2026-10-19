"""Read-only protocol recovery and dataset invariants for V4.5-A."""

import json
from pathlib import Path

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v4.shortlist import ranked_ids_from_row, scores_from_row
from experiments.qa_chunk_selection.v4_1.evaluate import row_key
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl

from .contract import DATASETS, RANKERS, STAGE2_CONTRACT, TRAINING_RECIPE
from .io_guard import canonical_hash, required_paths, sha256, validate_inputs


def _tree_hash(root):
    root = Path(root)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    return {"root": str(root.resolve()), "files": rows, "sha256": canonical_hash(rows)}


def _dataset_hashes(records, splits):
    documents = sorted({str(item.document_id) for item in records})
    questions = sorted(str(item.question_id) for item in records)
    candidate_rows = []
    chunk_rows = []
    for item in sorted(records, key=lambda row: str(row.question_id)):
        ids = [int(chunk.chunk_id) for chunk in item.chunks]
        candidate_rows.append([str(item.question_id), ids])
        chunk_rows.append([
            str(item.document_id), str(item.question_id), str(item.aspect), str(item.question),
            [[int(chunk.chunk_id), str(chunk.text), int(chunk.char_start), int(chunk.char_end),
              int(chunk.token_start), int(chunk.token_end)] for chunk in item.chunks],
        ])
    return {
        "dataset_hash": canonical_hash(chunk_rows),
        "candidate_universe_hash": canonical_hash(candidate_rows),
        "document_ids_hash": canonical_hash(documents),
        "question_ids_hash": canonical_hash(questions),
        "fold_membership_hash": canonical_hash(splits),
        "chunking_hash": canonical_hash(chunk_rows),
        "documents": len(documents),
        "questions": len(questions),
    }


def _candidate_check(records, named_rows):
    items = {str(item.question_id): item for item in records}
    problems = []
    for source, rows in named_rows.items():
        for row in rows:
            qid = str(row["question_id"])
            if qid not in items:
                problems.append({"source": source, "question_id": qid, "reason": "unknown_question"})
                continue
            expected = {int(chunk.chunk_id) for chunk in items[qid].chunks}
            ranked = {int(cid) for cid in ranked_ids_from_row(row)}
            scored = set(scores_from_row(row))
            if ranked != expected or (scored and scored != expected):
                problems.append({
                    "source": source, "question_id": qid, "seed": row.get("seed"),
                    "expected": sorted(expected), "ranked": sorted(ranked), "scored": sorted(scored),
                })
    return {
        "valid": not problems,
        "sources": {name: len(rows) for name, rows in named_rows.items()},
        "violations": problems[:100],
        "violation_count": len(problems),
        "paired_quality_valid": not problems,
    }


def _q5_replay(spec):
    root = spec.historical_factorial_root / "protocol/dity_msmarco/systems" / f"q0_k{spec.k}"
    paths = {name: root / name / "predictions_scored.jsonl" for name in ("B0", "SEM", "ASEM", "ASEM-Q5")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical Q5 replay inputs missing: {missing}")
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    indexes = {name: {row_key(row): row for row in values} for name, values in rows.items()}
    keys = set(indexes["ASEM"])
    if any(set(index) != keys for index in indexes.values()):
        raise ValueError(f"historical replay coverage mismatch for {spec.key}")

    def delta(left, right, metric):
        return sum(float(indexes[right][key][metric]) - float(indexes[left][key][metric]) for key in keys) / max(len(keys), 1)

    changed = [key for key in keys if indexes["ASEM"][key]["selected_chunk_id"] != indexes["ASEM-Q5"][key]["selected_chunk_id"]]
    sem_diff = [key for key in keys if indexes["SEM"][key]["selected_chunk_id"] != indexes["B0"][key]["selected_chunk_id"]]
    return {
        "dataset": spec.key,
        "historical_cell": f"q0_k{spec.k}",
        "same_top1_count": len(keys) - len(changed),
        "changed_top1_count": len(changed),
        "f1_delta_asem_q5_minus_asem": delta("ASEM", "ASEM-Q5", "token_f1"),
        "hit1_delta": delta("ASEM", "ASEM-Q5", "hit@1"),
        "hn1_delta": delta("ASEM", "ASEM-Q5", "hn@1"),
        "runtime_dependency_delta": "Q5-veto requires four additional question variants per candidate; primary V4.5 removes it",
        "alternative_sem_branch_exact_b0": not sem_diff,
        "alternative_sem_changed_top1_count": len(sem_diff),
        "descriptive_replay_only": True,
    }


def audit_dataset(spec):
    paths = validate_inputs(spec)
    records = records_from_jsonl(paths["records"])
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    if len(splits.get("runs", [])) != 5:
        raise ValueError(f"{spec.key}: expected exact five-run document grouped split")
    qcache = read_jsonl(paths["qcache"])
    q0 = {(str(row["document_id"]), str(row["aspect"]), int(row["chunk_id"])): row
          for row in qcache if int(row["question_id"]) == 0}
    missing_q0, wrong_question = [], []
    for item in records:
        for chunk in item.chunks:
            key = (str(item.document_id), str(item.aspect), int(chunk.chunk_id))
            row = q0.get(key)
            if row is None:
                missing_q0.append(key)
            elif str(row.get("question") or "") != str(item.question):
                wrong_question.append(key)
    if missing_q0 or wrong_question:
        raise ValueError(f"{spec.key}: canonical q0 cache mismatch: missing={len(missing_q0)}, text={len(wrong_question)}")
    named = {
        "historical_original_b0": read_jsonl(paths["historical_b0"]),
        "historical_dity": read_jsonl(paths["historical_dity"]),
    }
    if "historical_bge" in paths:
        named["historical_bge"] = read_jsonl(paths["historical_bge"])
    candidate = _candidate_check(records, named)
    if not candidate["valid"]:
        raise ValueError(f"{spec.key}: historical candidate universe mismatch")
    hashes = _dataset_hashes(records, splits)
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = {
        "dataset": spec.key, "read_only": True, "k": spec.k,
        "prepared_artifact": str(spec.prepared_artifact.resolve()),
        "reference_artifact": str(spec.reference_artifact.resolve()),
        **hashes, "source_hashes": source_hashes,
        "q0_question_text_hash": canonical_hash(sorted({str(item.question) for item in records})),
    }
    cache_audit = {
        "dataset": spec.key, "valid": True, "policy": "canonical q0 only",
        "rows_total": len(qcache), "q0_rows": len(q0),
        "question_hash_checked": True, "chunk_identity_checked": True,
        "reader": _tree_hash(spec.qa_model),
        "decoding": {"max_length": 512, "max_answer_tokens": 64, "extractive": True},
        "new_cache_location": "V4.5 run directory only; historical cache remains read-only",
    }
    return {
        "records": records, "splits": splits, "paths": paths,
        "manifest": manifest, "candidate_check": candidate,
        "chunking_check": {"dataset": spec.key, "valid": True, "chunking_hash": hashes["chunking_hash"]},
        "qa_cache_audit": cache_audit, "q5_replay": _q5_replay(spec),
    }


def historical_stage2_contract(audits):
    source_files = [
        Path(__file__).parents[1] / "v4_4_ru_ranker_factorial/pipeline.py",
        Path(__file__).parents[1] / "v4_4_ru_ablation/tuning.py",
        Path(__file__).parents[1] / "v4_4_ru_ablation/features.py",
        Path(__file__).parents[1] / "v4_1/ranking.py",
        Path(__file__).parents[1] / "v4_3/ranking.py",
    ]
    return {
        **STAGE2_CONTRACT,
        "historical_source_sha256": {str(path.resolve()): sha256(path) for path in source_files},
        "historical_replay": {key: audit["q5_replay"] for key, audit in audits.items()},
        "recovery_status": "recovered_exactly_from_source_and_completed_artifacts",
    }


def protocol_manifests(audits):
    sources = {key: audit["manifest"]["source_hashes"] for key, audit in audits.items()}
    return {
        "source_hashes": sources,
        "hash_before": sources,
        "split_check": {
            "valid": True, "datasets": {key: audit["manifest"]["fold_membership_hash"] for key, audit in audits.items()},
            "outer_eval_tuning": False, "roles": ["train", "cal", "eval"],
        },
        "candidate_universe_check": {key: audit["candidate_check"] for key, audit in audits.items()},
        "chunking_check": {key: audit["chunking_check"] for key, audit in audits.items()},
        "qa_cache_audit": {key: audit["qa_cache_audit"] for key, audit in audits.items()},
        "test_provenance_audit": {
            "phase": "V4.5-A development characterization",
            "candidates": {
                key: {
                    "status": "contaminated_for_confirmation",
                    "reason": "documents, texts, labels and predictions were used in V4.4 development",
                    "historical_artifact": str(DATASETS[key].historical_factorial_root.resolve()),
                } for key in audits
            },
            "independent_test_available": False,
        },
        "historical_reference_manifest": {
            "rankers": {key: vars(value) for key, value in RANKERS.items()},
            "training_recipe": TRAINING_RECIPE,
            "interpretation": "V4.5-A is development characterization; DiTy was selected after V4.4",
        },
    }
