"""Read-only provenance and history audit for V9."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS as V5_CELLS
from experiments.qa_chunk_selection_v6.audit import baseline_manifest, candidate_and_lineage_audit
from experiments.qa_chunk_selection_v8.audit import historical_hashes as v8_upstream_hashes
from .contract import (
    AGREEMENT_GROUP,
    CELLS,
    CONFIG_COLUMNS,
    H2_COLUMNS,
    PAIRLEX_GROUP,
    PAIRLEX_INCREMENTAL_GROUP,
)
from .io_guard import sha256, write_json, write_text


def git_snapshot(destination: Path) -> None:
    destination = Path(destination)
    commands = {
        "status.txt": ["git", "status", "--porcelain"],
        "head.txt": ["git", "rev-parse", "HEAD"],
        "diff.patch": ["git", "diff", "--no-ext-diff"],
    }
    for name, command in commands.items():
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        write_text(destination / name, completed.stdout)


def _v8_critical_files(v8_run: Path) -> list[Path]:
    v8_run = Path(v8_run)
    files = [
        v8_run / "COMPLETED.json",
        v8_run / "manifest.json",
        v8_run / "contracts/branch_hrr_frozen.yaml",
        v8_run / "analysis/global_summary.json",
        v8_run / "branch_hrr/analysis/primary.json",
        v8_run / "branch_hrr/analysis/paired.json",
    ]
    files += sorted(v8_run.glob("branch_hrr/models/local/*/outer_fold_*/seed_*/H[23]/*"))
    files += sorted(v8_run.glob("branch_hrr/predictions/local/*/*/outer_fold_*/seed_*/H[0-3].jsonl"))
    source = Path(__file__).resolve().parents[1] / "qa_chunk_selection_v8"
    files += sorted(x for x in source.glob("*.py") if x.is_file())
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def historical_hashes(config) -> dict:
    values = {
        f"v1-v7:{key}": value
        for key, value in v8_upstream_hashes(
            config.v4_4_run,
            config.v4_4_external_run,
            config.v4_5_run,
            config.v5_run,
            config.v6_run,
            config.v7_run,
            config.datasets,
        ).items()
    }
    files = _v8_critical_files(config.v8_run)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V8 critical files missing: {missing}")
    for path in files:
        resolved = path.resolve()
        values[f"v8:{resolved}"] = {
            "path": str(resolved),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
    return dict(sorted(values.items()))


def run_audit(output: Path, config) -> dict:
    output = Path(output)
    hashes = historical_hashes(config)
    write_json(output / "historical_hashes_before.json", hashes)
    write_json(output / "history_write_guard.json", {
        "active": True,
        "historical_versions": "V1-V8",
        "historical_roots_read_only": True,
        "v9_namespace_only": True,
    })
    write_json(output / "fulllex_group.json", {
        "status": "EXACT_V8_H2",
        "columns": [x for x in H2_COLUMNS if "fulllex" in x],
        "source": "experiments.qa_chunk_selection_v8.features.hrr_rows",
        "normalization": "within-question z-score, exact V8",
    })
    write_json(output / "agreement_group.json", {
        "status": "EXACT_V8_H3_MINUS_H2",
        "columns": list(AGREEMENT_GROUP),
        "formula": "candidate agreement_raw - B0 agreement_raw plus frozen H3 interactions",
        "source": "experiments.qa_chunk_selection_v8.features.hrr_rows",
        "normalization": "raw difference, exact V8",
        "new_cross_group_interactions": False,
    })
    write_json(output / "pairlex_group.json", {
        "status": "EXACT_V7_GROUP_ALREADY_PRESENT_IN_EXACT_V8_H2",
        "columns": list(PAIRLEX_GROUP),
        "incremental_columns_over_h2": list(PAIRLEX_INCREMENTAL_GROUP),
        "reference_answer": "predicted answer of B0 top1",
        "formula": "TokenF1(A_i,A_B0)/100 with exact V7 empty-answer semantics",
        "empty_answer_policy": {"both_empty": 1.0, "one_empty": 0.0},
        "value_range": [0.0, 1.0],
        "progressive_admission_queue_or_routing_included": False,
        "scientific_implication": "CP=C0 and CAP=CA under exact set-union semantics",
    })
    write_json(output / "candidate_configurations.json", {
        "columns": {key: list(value) for key, value in CONFIG_COLUMNS.items()},
        "cap_is_exact_union": True,
        "agreement_pairlex_cross_interaction_added": False,
        "cp_identity_with_c0": CONFIG_COLUMNS["CP"] == CONFIG_COLUMNS["C0"],
        "cap_identity_with_ca": CONFIG_COLUMNS["CAP"] == CONFIG_COLUMNS["CA"],
    })
    candidates, lineage, overlap = candidate_and_lineage_audit(config.datasets)
    write_json(output / "candidate_universe.json", candidates)
    write_json(output / "scier_lineage.json", lineage)
    write_json(output / "cross_corpus_overlap.json", overlap)
    write_json(output / "canonical_baselines.json", baseline_manifest(config.v5_run, config.datasets))
    write_json(output / "split_hashes.json", {cell: sha256(V5_CELLS[cell].splits) for cell in config.datasets})
    for cell in config.datasets:
        source = config.v5_run / "inputs/model_manifests" / f"{cell}_models.json"
        write_json(output / "model_manifests" / f"{cell}.json", {
            "source": str(source.resolve()), "sha256": sha256(source),
            "payload": json.loads(source.read_text()),
        })
    return hashes

