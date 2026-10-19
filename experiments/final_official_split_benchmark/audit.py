"""Pre-development provenance, exposure, split, and capability audits."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .config import Config
from .contract import CELL_SPECS, MODEL_REGISTRY, REAL_SYSTEMS
from .data import check_scier_linkage, load_test_runtime, normalized_text, split_manifest, text_hash
from .io_guard import canonical_hash, read_json, sha256, write_json

HISTORICAL_ROOTS = (
    "qa_chunk_selection_v2", "qa_chunk_selection_v3", "qa_chunk_selection_v4",
    "qa_chunk_selection_v4_1", "qa_chunk_selection_v4_2", "qa_chunk_selection_v4_3",
    "qa_chunk_selection_v4_4", "qa_chunk_selection_v4_4_external_20260913_213335",
    "qa_chunk_selection_v4_4_ru_20260912_191024", "qa_chunk_selection_v4_4_ru_ablation_20260912_230234",
    "qa_chunk_selection_v4_5_ru", "qa_chunk_selection_v5", "qa_chunk_selection_v6_universal",
    "qa_chunk_selection_v7_rapr", "qa_chunk_selection_v8_dual", "qa_chunk_selection_v9_h2_acs",
    "qa_chunk_selection_v10_srr",
)


def git_snapshot(path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout.strip()
    report = {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--short")}
    write_json(path, report)
    return report


def _guard_files() -> list[Path]:
    result = []
    for name in HISTORICAL_ROOTS:
        root = PROJECT_ROOT / "artifacts" / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = str(path.relative_to(root)).lower()
            if path.suffix.lower() in {".json", ".jsonl", ".md", ".yaml", ".yml"} and any(token in lower for token in ("manifest", "contract", "result", "analysis", "prediction", "completed", "report")):
                result.append(path)
    return sorted(result)


def historical_hashes() -> dict:
    values = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in _guard_files()}
    return {"files": len(values), "tree_sha256": canonical_hash(values), "entries": values}


def _historical_eval_files() -> list[Path]:
    return [p for p in _guard_files() if any(x in str(p).lower() for x in ("prediction", "evaluation", "analysis")) and p.suffix.lower() in {".json", ".jsonl"}]


def exposure_audit(config: Config) -> dict:
    if config.smoke:
        return {cell: {"status": "NOT_APPLICABLE_SMOKE", "matches": []} for cell in config.cells}
    needles = {}
    for cell in config.cells:
        runtime = load_test_runtime(cell)
        needles[cell] = {"qid": {x.question_id for x in runtime}, "doc": {x.document_id for x in runtime}, "hash": {text_hash(x) for x in runtime}}
    matches = {cell: [] for cell in config.cells}
    for path in _historical_eval_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for cell, values in needles.items():
            q = next((x for x in values["qid"] if x in text), None)
            d = next((x for x in values["doc"] if x in text), None)
            h = next((x for x in values["hash"] if x in text), None)
            if not h and any(key in text for key in ('"chunk_text"', '"document_text"', '"context"')):
                try:
                    payloads = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
                    def walk(value, key=""):
                        if isinstance(value, dict):
                            for child_key, child in value.items():
                                yield from walk(child, str(child_key))
                        elif isinstance(value, list):
                            for child in value: yield from walk(child, key)
                        elif isinstance(value, str) and key in {"chunk_text", "document_text", "context", "text"}:
                            yield hashlib.sha256(normalized_text(value).encode()).hexdigest()
                    observed = {value for payload in payloads for value in walk(payload)}
                    h = next(iter(values["hash"] & observed), None)
                except (ValueError, TypeError):
                    pass
            if q or d or h:
                matches[cell].append({"path": str(path.relative_to(PROJECT_ROOT)), "question_id_example": q, "document_id_example": d, "normalized_text_hash_example": h})
    result = {}
    for cell in config.cells:
        official = CELL_SPECS[cell].split_mode != "NO_OFFICIAL_TEST"
        status = "EXPOSED_TO_RESEARCH_PROGRAM" if matches[cell] else ("UNTOUCHED" if official else "NOT_APPLICABLE_NO_OFFICIAL_TEST")
        result[cell] = {"status": status, "matches": matches[cell][:50], "searched_historical_evaluation_files": len(_historical_eval_files()), "definition": "prior V1-V10 prediction/evaluation/analysis occurrence by question ID, document ID, stored document hash, or normalized text hash"}
    return result


def _snapshot(model_id: str) -> tuple[str | None, str | None]:
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None, None
    candidates = sorted(x for x in base.iterdir() if x.is_dir())
    if not candidates:
        return None, None
    chosen = candidates[-1]
    return chosen.name, str(chosen.resolve())


def model_provenance(config: Config) -> dict:
    result = {}
    for key, value in MODEL_REGISTRY.items():
        row = dict(value)
        if "path" in row:
            path = Path(row["path"])
            row.update(available=path.is_dir(), resolved_path=str(path.resolve()))
            if path.is_dir():
                files = {str(x.relative_to(path)): sha256(x) for x in sorted(path.rglob("*")) if x.is_file()}
                row["tree_sha256"] = canonical_hash(files)
        elif "model_id" in row:
            revision, path = _snapshot(row["model_id"])
            row.update(available=bool(path), resolved_revision=revision, resolved_path=path)
            if not path and config.allow_downloads:
                try:
                    from huggingface_hub import model_info
                    row["resolved_revision"] = str(model_info(row["model_id"]).sha)
                    row["availability_note"] = "exact remote revision resolved at audit; weights may download at first use"
                except Exception as exc:
                    row["resolution_error"] = f"{type(exc).__name__}: {exc}"
        result[key] = row
    return result


def capability_matrix(config: Config, provenance: dict) -> dict:
    rows = []
    for cell in config.cells:
        language = CELL_SPECS[cell].language
        for system in REAL_SYSTEMS:
            applicable = not (system == "B5-EN" and language != "en")
            status, reason = ("RUN", "predeclared") if applicable else ("NOT_APPLICABLE_LANGUAGE", "ModernBERT comparator is EN-only")
            needed = []
            if system in {"B4a", "B4b"}: needed = ["BGE"]
            if system == "B5-EN": needed = ["MODERNBERT_EN"]
            if system == "G1": needed = ["QWEN"]
            if applicable and any(not provenance[x].get("available") and not provenance[x].get("resolved_revision") for x in needed):
                status, reason = "BLOCKED_MISSING_MODEL", "required frozen model is absent and downloads are disabled"
            rows.append({"cell": cell, "system": system, "status": status, "reason": reason, "required_models": needed})
    for cell in config.cells:
        language = CELL_SPECS[cell].language
        rows.append({"cell": cell, "system": "B5-RU", "status": "NOT_RUN_PREDECLARED" if language == "ru" else "NOT_APPLICABLE_LANGUAGE",
                     "reason": MODEL_REGISTRY["B5_RU"]["reason"], "required_models": []})
        rows.append({"cell": cell, "system": "B5-EN-LONG", "status": "NOT_RUN_PREDECLARED" if language == "en" else "NOT_APPLICABLE_LANGUAGE",
                     "reason": MODEL_REGISTRY["B5_EN_LONG"]["reason"], "required_models": []})
    return {"rows": rows, "blocked": [x for x in rows if x["status"].startswith("BLOCKED")]}


def h2_replay_guard(run_dir: Path) -> dict:
    root = PROJECT_ROOT / "artifacts/qa_chunk_selection_v10_srr"
    candidates = []
    for marker in root.glob("run_*/audit/replay_COMPLETED.json"):
        try:
            value = json.loads(marker.read_text())
            detail = marker.parent / "s1_h2_replay.json"
            if value.get("status") == "PASS" and detail.is_file(): candidates.append((marker, detail))
        except (OSError, ValueError):
            continue
    if not candidates: raise RuntimeError("STOP_H2_DEFINITION_UNRESOLVED: no passing V10 exact V8↔V9 H2 replay")
    marker, detail = sorted(candidates)[-1]; rows = json.loads(detail.read_text())
    expected = {cell for cell in CELL_SPECS}; observed = {key.split("/", 1)[0] for key in rows}
    failed = [key for key, value in rows.items() if value.get("status") != "PASS"]
    if observed != expected or failed: raise RuntimeError(f"STOP_H2_DEFINITION_UNRESOLVED: cells={observed}, failed={failed[:5]}")
    report = {"status": "PASS", "source_marker": str(marker.relative_to(PROJECT_ROOT)), "source_detail": str(detail.relative_to(PROJECT_ROOT)),
              "source_marker_sha256": sha256(marker), "source_detail_sha256": sha256(detail), "replayed_units": len(rows),
              "definition": "exact V8 H2 equals V9/V10 S1/H2 on all historical cells/folds/seeds"}
    write_json(Path(run_dir) / "audit/h2_definition_replay.json", report, immutable=True); return report


def run_audit(config: Config, progress) -> dict:
    root = config.run_dir
    progress.stage("audit", "official split, overlap/exposure, model capability, and V1-V10 hash audit")
    git_snapshot(root / "audit/git_before.json")
    before = historical_hashes(); write_json(root / "audit/historical_hashes_before.json", before)
    h2_replay = h2_replay_guard(root)
    manifests = {}
    for index, cell in enumerate(config.cells, 1):
        progress.count(index, len(config.cells), item="split audit", extra={"cell": cell})
        manifests[cell] = split_manifest(cell, root, smoke=config.smoke)
    if not config.smoke and {"scier_128", "scier_384"} <= set(config.cells):
        check_scier_linkage(root)
    exposure = exposure_audit(config); write_json(root / "test_exposure/exposure.json", exposure); write_json(root / "audit/test_exposure/exposure.json", exposure)
    evidence = {cell: {"positive_labels": True, "hard_negative_labels": cell != "scirex",
        "mapping": {"positive": "SAFE_EVIDENCE", "hard_negative": "UNSAFE_EVIDENCE", "easy_negative": "EXCLUDE"},
        "note": "SciREX official TEST has exact supporting spans but no frozen HN classification" if cell == "scirex" else "prepared frozen labeling pipeline"}
        for cell in config.cells}
    write_json(root / "manifests/evidence_label_availability.json", evidence); write_json(root / "audit/evidence_label_availability.json", evidence)
    write_json(root / "manifests/evidence_label_mapping.json", {"positive": "SAFE_EVIDENCE", "hard_negative": "UNSAFE_EVIDENCE", "ambiguous": "EXCLUDE", "easy_negative": "EXCLUDE"})
    write_json(root / "audit/evidence_label_mapping.json", {"positive": "SAFE_EVIDENCE", "hard_negative": "UNSAFE_EVIDENCE", "ambiguous": "EXCLUDE", "easy_negative": "EXCLUDE"})
    provenance = model_provenance(config); write_json(root / "manifests/model_provenance.json", provenance)
    capabilities = capability_matrix(config, provenance); write_json(root / "manifests/system_capability_matrix.json", capabilities)
    write_json(root / "audit/method_capability_matrix.json", capabilities)
    write_json(root / "audit/official_split_source.json", {cell: {"mode": CELL_SPECS[cell].split_mode, "source": CELL_SPECS[cell].official_source} for cell in config.cells})
    write_json(root / "audit/split_hashes.json", {cell: manifest["sha256"] for cell, manifest in manifests.items()})
    write_json(root / "audit/evaluation_population.json", {cell: {"rule": "canonical answerable subset", "prediction_population": "all prepared TEST questions", "evaluation_filter_applied_after_blind_prediction": True} for cell in config.cells})
    write_json(root / "audit/preprocessing_hashes.json", {cell: {"chunk_size": CELL_SPECS[cell].chunk_size, "overlap": CELL_SPECS[cell].overlap, "split_manifest_sha256": manifests[cell]["sha256"]} for cell in config.cells})
    write_json(root / "audit/confidence_definition.json", {"extractive_QA": "raw start/end span joint score from frozen reader", "empty_answer_selection_confidence": -1e30, "tie_break": "rank then chunk_id", "BGE": "one-logit frozen cross-encoder score"})
    report = {"split_manifests": {k: v["sha256"] for k, v in manifests.items()}, "exposure": exposure, "historical_tree_sha256": before["tree_sha256"], "capability_blocked": len(capabilities["blocked"]), "h2_definition_replay": h2_replay}
    write_json(root / "audit/AUDIT_COMPLETE.json", report, immutable=True)
    return report


def assert_history_unchanged(run_dir: Path) -> None:
    before = read_json(Path(run_dir) / "audit/historical_hashes_before.json")
    after = historical_hashes(); write_json(Path(run_dir) / "audit/historical_hashes_after.json", after)
    if before["tree_sha256"] != after["tree_sha256"] or before["entries"] != after["entries"]:
        raise RuntimeError("RUN_INVALID_HISTORY_MODIFIED: V1-V10 artifacts changed")
