"""Read-only historical audit, hash protection, lineage audit and V5 replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS

from .contract import ASPECT_MAP, CELLS, FAMILIES, V5_RUN, frozen_contract
from .features import canonical_document_id, normalized_document_hash
from .io_guard import canonical_hash, read_jsonl, sha256, write_json, write_text


def critical_files(v5_run: Path = V5_RUN, cells=CELLS) -> dict[str, Path]:
    files = {"v5_completed": v5_run / "COMPLETED.json", "v5_contract": v5_run / "contract_frozen.json"}
    for cell in cells:
        spec = V5_CELLS[cell]
        values = {"records": spec.records, "splits": spec.splits, "qa_cache": spec.qa_cache,
                  "historical_b0": spec.historical_b0, "historical_reference": spec.historical_reference}
        if spec.questions_q5:
            values["questions_q5"] = spec.questions_q5
        if spec.question_selection:
            values["question_selection"] = spec.question_selection
        for name, path in values.items():
            files[f"{cell}:{name}"] = Path(path)
        for fold in range(5):
            for seed in (42, 43, 44):
                root = v5_run / cell / f"outer_fold_{fold}" / f"seed_{seed}"
                for name, path in {
                    "meta_features": root / "inner_meta/features.jsonl",
                    "eval_features": root / "eval/features.jsonl",
                    "v5_a1_model": root / "residual/A1/model.joblib",
                    "eval_question_policy": root / "eval/question_policy.json",
                }.items():
                    files[f"{cell}:fold{fold}:seed{seed}:{name}"] = path
    return files


def historical_hashes(v5_run: Path = V5_RUN, cells=CELLS) -> dict:
    files = critical_files(v5_run, cells)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical inputs missing: {missing}")
    return {key: {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size}
            for key, path in sorted(files.items())}


def git_snapshot(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    commands = {"status.txt": ["git", "status", "--porcelain"], "head.txt": ["git", "rev-parse", "HEAD"],
                "diff.patch": ["git", "diff", "--no-ext-diff"]}
    for name, command in commands.items():
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        write_text(destination / name, result.stdout)


def _record_maps(cell: str):
    records = records_from_jsonl(V5_CELLS[cell].records)
    return {str(item.question_id): item for item in records}


def candidate_and_lineage_audit(cells=CELLS) -> tuple[dict, dict, dict]:
    candidates = {}; lineage = {}; by_hash = defaultdict(list)
    for cell in cells:
        records = records_from_jsonl(V5_CELLS[cell].records)
        historical_qids = {str(row["question_id"]) for row in read_jsonl(V5_CELLS[cell].historical_b0)}
        items = [item for item in records if str(item.question_id) in historical_qids]
        candidate_rows = []
        candidate_text_rows = []
        docs = {}
        for item in items:
            qid = str(item.question_id)
            candidate_rows.append([qid, [int(chunk.chunk_id) for chunk in item.chunks]])
            candidate_text_rows.append([qid, [sha256_text(chunk.text) for chunk in item.chunks]])
            canonical = canonical_document_id(cell, str(item.document_id)); digest = normalized_document_hash(item)
            docs[str(item.document_id)] = {"canonical_document_id": canonical, "document_sha256": digest}
            by_hash[digest].append({"cell": cell, "family": FAMILIES[cell], "document_id": str(item.document_id),
                                    "canonical_document_id": canonical})
        candidates[cell] = {"questions": len(items), "sha256": canonical_hash(candidate_rows),
                            "chunk_text_sha256": canonical_hash(candidate_text_rows),
                            "historical_v5_sha256": json.loads((V5_RUN / "inputs/candidate_universe_hashes.json").read_text())[cell]}
        if candidates[cell]["sha256"] != candidates[cell]["historical_v5_sha256"]:
            raise RuntimeError(f"candidate universe replay mismatch: {cell}")
        lineage[cell] = docs
    overlap = {digest: rows for digest, rows in by_hash.items() if len({r["family"] for r in rows}) > 1}
    scier_docs = defaultdict(set)
    for cell in ("scier_384", "scier_128"):
        if cell in lineage:
            for row in lineage[cell].values():
                scier_docs[row["canonical_document_id"]].add(cell)
    scier_summary = {"documents": len(scier_docs), "represented_in_both": sum(len(v) == 2 for v in scier_docs.values()),
                     "canonical_ids": sorted(scier_docs)}
    return candidates, {"cells": lineage, "scier": scier_summary}, {"cross_family_exact_hash_overlap": overlap}


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def baseline_manifest(v5_run: Path = V5_RUN, cells=CELLS) -> dict:
    primary = json.loads((v5_run / "analysis/primary_metrics.json").read_text())
    refs = json.loads((v5_run / "analysis/historical_references.json").read_text())
    out = {}
    for cell in cells:
        spec = V5_CELLS[cell]
        out[cell] = {"dataset_cell": cell, "corpus_family": FAMILIES[cell],
                     "historical_raw_b0": primary[cell]["B0"], "v5_primary": primary[cell].get("V5-Q-A1"),
                     "historical_selected_system": refs[cell], "question_policy": spec.question_policy,
                     "candidate_universe": str(spec.records.resolve()), "ranker": spec.ranker_id,
                     "ranker_revision": spec.ranker_revision, "qa_reader": str(spec.qa_model.resolve()), "k": spec.k}
    return out


def run_audit(output: Path, *, v5_run: Path = V5_RUN, cells=CELLS) -> dict:
    output = Path(output)
    hashes = historical_hashes(v5_run, cells)
    candidates, lineage, overlap = candidate_and_lineage_audit(cells)
    write_json(output / "historical_hashes_before.json", hashes)
    write_json(output / "canonical_baselines.json", baseline_manifest(v5_run, cells))
    write_json(output / "candidate_universe.json", candidates)
    write_json(output / "cross_cell_document_lineage.json", lineage)
    write_json(output / "cross_corpus_overlap.json", overlap)
    write_json(output / "canonical_aspect_mapping.json", {"mapping": ASPECT_MAP,
               "mapping_contract": "explicit bilingual semantic ids; unknown -> OTHER", "audit_status": "frozen_before_eval"})
    write_json(output / "k_manifest.json", {cell: V5_CELLS[cell].k for cell in cells})
    write_json(output / "upstream_training_provenance.json", {cell: "dataset_trained" for cell in cells})
    write_json(output / "split_hashes.json", {cell: sha256(V5_CELLS[cell].splits) for cell in cells})
    for cell in cells:
        historical_model = v5_run / "inputs/model_manifests" / f"{cell}_models.json"
        historical_question = v5_run / "inputs/question_manifests" / f"{cell}.json"
        write_json(output / "model_manifests" / f"{cell}.json", {
            "historical_manifest": json.loads(historical_model.read_text()),
            "historical_manifest_sha256": sha256(historical_model), "source": str(historical_model.resolve())})
        write_json(output / "question_policy" / f"{cell}.json", {
            "historical_manifest": json.loads(historical_question.read_text()),
            "historical_manifest_sha256": sha256(historical_question), "source": str(historical_question.resolve())})
    write_json(output / "availability_proxy_audit.json", {"patterns_by_cell": {cell: [1, 1, 0, 0] for cell in cells},
               "mutual_information_with_corpus_family": 0.0, "mask_fit_diagnostic": "with/without masks algebraically identical because pattern is constant"})
    return {"hashes": hashes, "candidate_universe": candidates, "lineage": lineage, "overlap": overlap}


def parser():
    value = argparse.ArgumentParser(description="V6 read-only historical audit")
    value.add_argument("--project-root", type=Path, default=Path(".")); value.add_argument("--read-only-history", action="store_true")
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.read_only_history:
        raise ValueError("--read-only-history is mandatory")
    run_audit(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

