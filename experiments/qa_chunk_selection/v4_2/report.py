"""Human-readable V4.2 autodump. Not a substitute for a later scientific writeup."""

from __future__ import annotations

from typing import Any, Mapping

from .io_guard import write_json


def write_report(output_dir, payload: Mapping[str, Any]) -> None:
    lines = [
        "# V4.2 autodump",
        "",
        "This file is generated. It is not the human scientific report.",
        "",
        f"- B0-live Token F1: `{payload.get('b0_token_f1')}`",
        f"- V6-K5 Token F1: `{payload.get('v6_token_f1')}`",
        f"- CORE1 Token F1: `{payload.get('core1_token_f1')}`",
        f"- CORE2 Token F1: `{payload.get('core2_token_f1')}`",
        f"- ATT1 Token F1: `{payload.get('att1_token_f1')}`",
        f"- ATT1 vs V6 ΔF1: `{payload.get('att1_vs_v6')}`",
        f"- ATT1 gate: `{payload.get('att1_gate')}`",
        f"- Selected (OOF): `{payload.get('selected')}`",
        f"- INTERNAL_TEST locked: `{payload.get('internal_test_locked', True)}`",
        "",
        "V6 was not retuned. Attention is a shortlist-relative correction on B0 top-5.",
        "Prior E0/V2/V3/V3.1/D45/V4/V4.1 trees were read-only.",
        "",
    ]
    path = output_dir / "reports" / "results_v4_2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "analysis" / "compare_summary.json", dict(payload))
