"""Generate reports/results_v3.md from artifacts. Never invent numbers."""

from __future__ import annotations

import json
from pathlib import Path

from .freeze import freeze_path
from .io_util import load_json


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:
            return "NA"
        return f"{value:.4f}"
    return str(value)


def write_results_v3(output_dir: Path, *, allow_internal_test: bool = False) -> Path:
    reports = Path(output_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = ["# QA chunk selection V3 results", ""]
    cfg_path = Path(output_dir) / "config_resolved.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        lines += ["## 1. Configuration", "", "```json", json.dumps(cfg, indent=2, ensure_ascii=False)[:5000], "```", ""]
    lines += ["## 2. Splits and INTERNAL_TEST lock", ""]
    splits = Path(output_dir) / "data" / "v3_cv_splits.json"
    if splits.exists():
        payload = load_json(splits)
        lines += [
            f"n_documents={payload.get('n_documents')} n_folds={payload.get('n_folds')} seed={payload.get('split_seed')}",
            f"document_id_hash={payload.get('document_id_hash')}",
            "",
        ]
    locked = not freeze_path(output_dir).exists() or not allow_internal_test
    lines += [f"INTERNAL_TEST lock: {'LOCKED' if locked else 'OPENED for final-eval'}", ""]
    overlay = Path(output_dir) / "annotation" / "sampling_summary.json"
    lines += ["## 3. Target-status overlay", ""]
    if overlay.exists():
        lines += ["```json", json.dumps(load_json(overlay), indent=2, ensure_ascii=False), "```", ""]
    else:
        lines += ["Overlay sampling artifacts missing.", ""]
    for section, rel in [
        ("4. OOF comparison", "analysis/oof_compare.json"),
        ("5. Selection", "analysis/selection.json"),
        ("6. Attention AT0", "analysis/attention/at0.json"),
        ("7. Attention AT1", "analysis/attention/at1.json"),
        ("8. Positive quality / D4 D5 gates", "analysis/positive_quality/summary.json"),
        ("9. Paired bootstrap (OOF, not confirmatory TEST)", "analysis/bootstrap.json"),
    ]:
        lines += [f"## {section}", ""]
        path = Path(output_dir) / rel
        if path.exists():
            lines += ["```json", json.dumps(load_json(path), indent=2, ensure_ascii=False)[:4000], "```", ""]
        else:
            lines += ["artifact missing.", ""]
    lines += [
        "## Notes",
        "",
        "- OOF bootstrap is model-selection evidence, not the confirmatory INTERNAL_TEST.",
        "- `source_label` is never rewritten; V3 target-status is an overlay.",
        "- Gold is not used at ranking/inference except diagnostic gold-span attention.",
        "",
    ]
    path = reports / "results_v3.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
