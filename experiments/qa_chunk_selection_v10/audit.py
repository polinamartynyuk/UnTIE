from __future__ import annotations

import json
from pathlib import Path
import subprocess

from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.audit import baseline_manifest, candidate_and_lineage_audit
from .contract import (
    EVIDENCE_MAPPING, H2_COLUMNS, INNER_EVIDENCE_MIN_QUESTIONS,
    INNER_EVIDENCE_MIN_ROWS, REFIT_EVIDENCE_MIN_QUESTIONS, REFIT_EVIDENCE_MIN_ROWS,
)
from .features import inner_manifest
from .io_guard import read_jsonl, sha256, write_json, write_text


def git_snapshot(destination: Path):
    for name, command in {
        "status.txt": ["git", "status", "--porcelain"],
        "head.txt": ["git", "rev-parse", "HEAD"],
        "diff.patch": ["git", "diff", "--no-ext-diff"],
    }.items():
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        write_text(destination / name, result.stdout)


def historical_hashes(config):
    frozen = json.loads((Path(config.v9_run) / "audit/historical_hashes_after.json").read_text())
    values = {}
    for key, recorded in frozen.items():
        path = Path(recorded["path"])
        if not path.is_file(): raise FileNotFoundError(f"historical file missing: {path}")
        current = {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size}
        if current["sha256"] != recorded["sha256"] or current["size"] != recorded["size"]:
            raise RuntimeError(f"historical V1-V8 hash mismatch before V10: {path}")
        values[f"v1-v8:{key}"] = current
    roots = [Path(config.v9_run), Path(__file__).resolve().parents[1] / "qa_chunk_selection_v9"]
    files = []
    for root in roots:
        files.extend(x for x in root.rglob("*") if x.is_file())
    for path in sorted(set(x.resolve() for x in files)):
        values[f"v9:{path}"] = {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
    return dict(sorted(values.items()))


def evidence_support_feasibility(config):
    from collections import defaultdict

    report = {}
    for cell in config.datasets:
        report[cell] = {}
        for outer_fold in config.outer_folds:
            reference_manifest = None; seed_reports = {}
            for seed in config.seeds:
                source = config.v9_run / "cache/canonical_features" / cell / f"outer_fold_{outer_fold}" / f"seed_{seed}" / "meta_hrr_targets.jsonl"
                rows = read_jsonl(source); by_question = defaultdict(list)
                for row in rows: by_question[str(row["question_id"])].append(row)
                questions = []
                for question_id, values in sorted(by_question.items()):
                    first = values[0]
                    candidates = [{"evaluation": {"evidence_class": "EXCLUDE"}}]
                    for row in values:
                        label = EVIDENCE_MAPPING.get(row["candidate_tier"], "UNKNOWN")
                        if label == "UNKNOWN": raise RuntimeError(f"UNKNOWN_EVIDENCE_LABEL: {row["candidate_tier"]!r}")
                        candidates.append({"evaluation": {"evidence_class": label}})
                    questions.append({"question_id": question_id, "canonical_document_id": first["canonical_document_id"], "candidates": candidates})
                if reference_manifest is None:
                    if seed != 42: raise RuntimeError("seed-42 feasibility reference must be processed first")
                    reference_manifest = inner_manifest(questions)
                manifest = reference_manifest
                if {q["canonical_document_id"] for q in questions} != set(manifest["assignment"]):
                    raise RuntimeError(f"INNER_DOCUMENT_UNIVERSE_DIFFERS_ACROSS_SEEDS: {cell}/f{outer_fold}/s{seed}")
                used = [row for row in rows if EVIDENCE_MAPPING.get(row["candidate_tier"]) != "EXCLUDE"]
                def support(values):
                    result = {}
                    for target, tier in ((0, "sufficient_positive"), (1, "insufficient_evidence")):
                        selected = [row for row in values if row["candidate_tier"] == tier]
                        result[str(target)] = {"rows": len(selected), "questions": len({row["question_id"] for row in selected})}
                    return result
                full = support(used)
                if any(value["rows"] < REFIT_EVIDENCE_MIN_ROWS or value["questions"] < REFIT_EVIDENCE_MIN_QUESTIONS for value in full.values()):
                    raise RuntimeError(f"EVIDENCE_REFIT_PREFLIGHT_BLOCKED: {cell}/f{outer_fold}/s{seed}: {full}")
                inner = {}
                for valid_fold in range(manifest["folds"]):
                    train = [row for row in used if manifest["assignment"][row["canonical_document_id"]] != valid_fold]
                    current = support(train); inner[str(valid_fold)] = current
                    if any(value["rows"] < INNER_EVIDENCE_MIN_ROWS or value["questions"] < INNER_EVIDENCE_MIN_QUESTIONS for value in current.values()):
                        raise RuntimeError(f"EVIDENCE_INNER_PREFLIGHT_BLOCKED: {cell}/f{outer_fold}/s{seed}/i{valid_fold}: {current}")
                seed_reports[str(seed)] = {"source": str(source), "split_sha256": manifest["sha256"], "full_refit": full, "inner_train": inner}
            report[cell][str(outer_fold)] = seed_reports
    return report


def run_audit(output: Path, config):
    hashes = historical_hashes(config)
    write_json(output / "historical_hashes_before.json", hashes)
    write_json(output / "evidence_support_feasibility.json", evidence_support_feasibility(config))
    write_json(output / "history_write_guard.json", {"active": True, "historical_versions": "V1-V9", "v10_namespace_only": True})
    candidates, lineage, overlap = candidate_and_lineage_audit(config.datasets)
    write_json(output / "candidate_universe.json", candidates)
    write_json(output / "scier_lineage.json", lineage)
    write_json(output / "cross_corpus_overlap.json", overlap)
    write_json(output / "canonical_baselines.json", baseline_manifest(config.v5_run, config.datasets))
    write_json(output / "outer_split_hashes.json", {cell: sha256(V5_CELLS[cell].splits) for cell in config.datasets})
    write_json(output / "h2_feature_schema.json", {
        "columns": list(H2_COLUMNS), "ordered": True,
        "source": "experiments.qa_chunk_selection_v8.contract.H2_COLUMNS",
        "learned_h2_outputs_excluded": True,
    })
    mapping = {cell: {str(key): value for key, value in EVIDENCE_MAPPING.items()} for cell in config.datasets}
    write_json(output / "protocol_amendment_v10_0_1.json", {
        "reason": "SciER384 outer TRAIN/CAL can contain only 11 UNSAFE_EVIDENCE rows, making >=10 rows in every 4-fold INNER_TRAIN mathematically impossible",
        "inner_split": "StratifiedGroupKFold by question_has_unsafe_evidence_alternative; grouped by canonical_document_id; seed=42",
        "inner_evidence_minimum": {"rows_per_class": 5, "questions_per_class": 5},
        "full_outer_refit_minimum": {"rows_per_class": 10, "questions_per_class": 5},
        "constant_predictor_or_manual_prior": False,
        "historical_run_reuse": False,
    })
    write_json(output / "evidence_label_mapping.json", {
        "mapping_by_cell": mapping,
        "safe": ["sufficient_positive"], "unsafe": ["insufficient_evidence"],
        "excluded": ["unresolved_for_training", None, ""], "string_guessing": False,
    })
    for cell in config.datasets:
        source = config.v5_run / "inputs/model_manifests" / f"{cell}_models.json"
        write_json(output / "model_manifests" / f"{cell}.json", {"source": str(source.resolve()), "sha256": sha256(source), "payload": json.loads(source.read_text())})
    return hashes

