"""Autodump V3.1-Silver results. Not a human scientific report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_guard import load_json, write_json


def write_results_v31(output_dir: Path) -> Path:
    root = Path(output_dir)
    payload: dict[str, Any] = {
        "title": "V3.1-Silver autodump (not a human report)",
        "preconditions": _maybe(root / "analysis" / "preconditions.json"),
        "s0": _maybe(root / "analysis" / "s0_audit.json"),
        "b0": _maybe(root / "analysis" / "oof" / "b0" / "metrics.json"),
        "b1": _maybe(root / "analysis" / "oof" / "b1" / "metrics.json"),
        "b0_silver": _maybe(root / "analysis" / "oof" / "b0" / "metrics_silver.json"),
        "b1_silver": _maybe(root / "analysis" / "oof" / "b1" / "metrics_silver.json"),
        "g1": _maybe(root / "analysis" / "g1.json"),
        "bootstrap": _maybe(root / "analysis" / "bootstrap.json"),
        "note": (
            "Primary endpoint is original-gold Token F1. "
            "SilverTargetMismatch metrics are QA-based silver diagnostics."
        ),
    }
    json_path = root / "reports" / "results_v3_1.json"
    md_path = root / "reports" / "results_v3_1.md"
    write_json(json_path, payload)
    lines = [
        "# V3.1-Silver autodump",
        "",
        "This file is generated. It is not the human scientific report.",
        "",
        f"- G1: `{(payload.get('g1') or {}).get('decision')}`",
        f"- B0 Token F1: `{(payload.get('b0') or {}).get('token_f1')}`",
        f"- B1 Token F1: `{(payload.get('b1') or {}).get('token_f1')}`",
        f"- B0 HN@1: `{(payload.get('b0') or {}).get('hn@1')}`",
        f"- B1 HN@1: `{(payload.get('b1') or {}).get('hn@1')}`",
        f"- Silver TM@1 B0: `{(payload.get('b0_silver') or {}).get('SilverTargetMismatch@1_lower_bound')}`",
        f"- Silver TM@1 B1: `{(payload.get('b1_silver') or {}).get('SilverTargetMismatch@1_lower_bound')}`",
        "",
        payload["note"],
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _maybe(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None
