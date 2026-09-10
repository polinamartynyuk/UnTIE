"""Human-readable V4 autodump. Not a substitute for a later scientific writeup."""

from __future__ import annotations

from typing import Any, Mapping

from .io_guard import write_json


def write_report(output_dir, payload: Mapping[str, Any]) -> None:
    lines = [
        "# V4 autodump",
        "",
        "This file is generated. It is not the human scientific report.",
        "",
        f"- B0-live Token F1: `{payload.get('b0_token_f1')}`",
        f"- Compared systems: `{', '.join(payload.get('systems') or [])}`",
        f"- INTERNAL_TEST locked: `{payload.get('internal_test_locked', True)}`",
        f"- Selected (if any): `{payload.get('selected')}`",
        "",
        "Gold was not used at runtime ranking. Prior E0/V2/V3/V3.1/D45 trees were read-only.",
        "",
    ]
    path = output_dir / "reports" / "results_v4.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "analysis" / "compare_summary.json", dict(payload))
