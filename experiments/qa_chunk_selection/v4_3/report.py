"""Human-readable V4.3 autodump. Not a substitute for a later scientific writeup."""

from __future__ import annotations

from typing import Any, Mapping

from .io_guard import write_json


def write_report(output_dir, payload: Mapping[str, Any]) -> None:
    lines = [
        "# V4.3 autodump",
        "",
        "This file is generated. It is not the human scientific report.",
        "",
        f"- B0-live Token F1: `{payload.get('b0_token_f1')}`",
        f"- V6-K5 Token F1: `{payload.get('v6_token_f1')}`",
        f"- SEM1 Token F1: `{payload.get('sem1_token_f1')}`",
        f"- SEM1 vs V6 ΔF1: `{payload.get('sem1_vs_v6')}`",
        f"- SEM1 gate: `{payload.get('sem1_gate')}`",
        f"- LEXCORE Token F1: `{payload.get('lexcore_token_f1')}`",
        f"- SEMCORE Token F1: `{payload.get('semcore_token_f1')}`",
        f"- SEMCORE vs LEXCORE ΔF1: `{payload.get('semcore_vs_lexcore')}`",
        f"- ADAPT1 Token F1: `{payload.get('adapt1_token_f1')}`",
        f"- SEMCTX1 Token F1: `{payload.get('semctx1_token_f1')}`",
        f"- Selected (OOF): `{payload.get('selected')}`",
        f"- INTERNAL_TEST locked: `{payload.get('internal_test_locked', True)}`",
        "",
        "V6 was not retuned. Semantic encoder is frozen all-MiniLM-L6-v2.",
        "Prior E0/V2/V3/V3.1/D45/V4/V4.1/V4.2 trees were read-only.",
        "",
    ]
    path = output_dir / "reports" / "results_v4_3.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "analysis" / "compare_summary.json", dict(payload))
