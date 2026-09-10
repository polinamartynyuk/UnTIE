"""Generate reports/results_v2.md from artifacts. Never invent numbers."""

from __future__ import annotations

import json
from pathlib import Path

from .io_util import load_json, read_jsonl
from .freeze import freeze_path


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:
            return "NA"
        return f"{value:.4f}"
    return str(value)


def write_results_v2(output_dir: Path, *, allow_internal_test: bool = False) -> Path:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = ["# QA chunk selection V2 results", ""]
    cfg_path = output_dir / "config_resolved.json"
    contract_path = output_dir / "contract.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        lines += ["## 1. Configuration and versions", "", "```json", json.dumps(cfg, indent=2, ensure_ascii=False)[:4000], "```", ""]
    if contract_path.exists():
        lines += ["Contract: `contract.json`", ""]
    splits_path = output_dir / "data" / "data_summary.json"
    lines += ["## 2. Splits and INTERNAL_TEST lock status", ""]
    if splits_path.exists():
        lines += ["```json", json.dumps(load_json(splits_path), indent=2, ensure_ascii=False)[:5000], "```", ""]
    locked = not freeze_path(output_dir).exists() or not allow_internal_test
    lines += [f"INTERNAL_TEST lock: {'LOCKED' if locked else 'OPENED for final-eval'}", ""]
    lines += ["## 3. E0 regression", ""]
    reg = output_dir / "regression" / "e0_regression.json"
    if reg.exists():
        payload = load_json(reg)
        lines += [f"R1 ok={payload.get('r1', {}).get('ok')}", f"R2 ok={payload.get('r2', {}).get('ok')}", ""]
        lines += ["```json", json.dumps(payload, indent=2, ensure_ascii=False)[:4000], "```", ""]
    else:
        lines.append("E0 regression artifact missing.")
        lines.append("")
    lines += ["## 4. Reader oracle", ""]
    oracle = output_dir / "analysis" / "reader_oracle" / "summary.json"
    if oracle.exists():
        lines += ["```json", json.dumps(load_json(oracle), indent=2, ensure_ascii=False)[:3000], "```", ""]
    lines += ["## 5. Ambiguous audit", ""]
    for tag in ("e0", "b2"):
        path = output_dir / "analysis" / "ambiguous_audit" / tag / "summary.json"
        if path.exists():
            lines += [f"### {tag}", "", "```json", json.dumps(load_json(path), indent=2, ensure_ascii=False)[:2500], "```", ""]
    lines += ["## 6. HN quality warnings", ""]
    flags = output_dir / "data" / "hn_quality_flags.jsonl"
    if flags.exists():
        rows = read_jsonl(flags)
        n_warn = sum(1 for row in rows if row.get("hn_gold_string_warning"))
        n_el = sum(1 for row in rows if row.get("training_eligible_hn"))
        lines += [f"flagged HN gold-string warnings: {n_warn}", f"training_eligible_hn: {n_el}", ""]
    lines += ["## 7. B2 variants", ""]
    sel = output_dir / "models" / "b2" / "selection.json"
    if sel.exists():
        lines += ["```json", json.dumps(load_json(sel), indent=2, ensure_ascii=False)[:3000], "```", ""]
    for section, rel in [
        ("8. Reader-aware", "analysis/reader_aware/summary.json"),
        ("9. M sweep", "analysis/c_sweep/summary.json"),
        ("10. Adaptive M", "analysis/c_adaptive/summary.json"),
        ("11. D2 ablations", "analysis/d2_ablation/summary.json"),
        ("12. D4", "analysis/d4/summary.json"),
        ("13. D5", "analysis/d5/summary.json"),
        ("17. Paired comparisons", "analysis/paired_comparisons.json"),
    ]:
        lines += [f"## {section}", ""]
        path = output_dir / rel
        if path.exists():
            lines += ["```json", json.dumps(load_json(path), indent=2, ensure_ascii=False)[:3500], "```", ""]
        else:
            lines += ["Not run or artifact missing.", ""]
    lines += ["## 23. INTERNAL_TEST results", ""]
    if locked:
        lines += ["INTERNAL_TEST NOT OPENED", ""]
    else:
        final = output_dir / "reports" / "final_internal_test.md"
        lines += [f"See `{final}` if final-eval completed.", ""]
    lines += ["## Metric scales", "", "- Token F1: 0..100", "- Char F1: 0..1", "- ROUGE-L: 0..1", "- EM: 0..1", ""]
    path = reports / "results_v2.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
