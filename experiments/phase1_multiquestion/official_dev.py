"""Official SciREX DEV multi-question: Q5 then Q3, select thresholds.

Does not rewrite preprocessing, does not run TEST, and does not touch the
legacy hash-split artifacts under artifacts/phase1_multiquestion/dev_q* or
frozen_thresholds.json.

Usage from the project root:

    ./untie_venv/bin/python -m experiments.phase1_multiquestion.official_dev --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .run import main as run_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "artifacts" / "phase1_multiquestion"
PREPROCESS = PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing_dev"
PHASE1_DEV = PROJECT_ROOT / "artifacts" / "phase1_scirex" / "official_dev" / "qa_chunk_dataset.parquet"
ROOT = PROJECT_ROOT / "artifacts" / "phase1_multiquestion"
CACHE = ROOT / "official_dev_qa_prediction_cache.jsonl"
THRESHOLDS = ROOT / "official_frozen_thresholds.json"
DEV_Q5 = ROOT / "official_dev_q5"
DEV_Q3 = ROOT / "official_dev_q3"

STEPS = ("official_dev_q5", "official_dev_q3")


def _ok(output_dir: Path) -> bool:
    stats = output_dir / "statistics.json"
    sanity = output_dir / "sanity_errors.json"
    if not stats.exists() or not sanity.exists():
        return False
    errors = json.loads(sanity.read_text(encoding="utf-8"))
    return errors == []


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_official_thresholds() -> dict:
    payload = {
        "Q0": {
            "chosen_threshold": 1.0,
            "source": "fixed: Q0 has only 0/1 consistency",
        }
    }
    q5 = DEV_Q5 / "threshold_selection.json"
    q3 = DEV_Q3 / "threshold_selection.json"
    if q5.exists():
        data = _load_json(q5)
        payload["Q5"] = {
            "chosen_threshold": data["chosen_threshold"],
            "criterion": data.get("criterion"),
            "candidates": data.get("candidates"),
            "source": str(q5),
        }
    if q3.exists():
        data = _load_json(q3)
        payload["Q3"] = {
            "chosen_threshold": data["chosen_threshold"],
            "criterion": data.get("criterion"),
            "candidates": data.get("candidates"),
            "source": str(q3),
        }
    ROOT.mkdir(parents=True, exist_ok=True)
    THRESHOLDS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _run(argv: list[str]) -> None:
    code = run_main(argv)
    if code:
        raise SystemExit(code)


def step_status() -> list[tuple[str, bool, str]]:
    return [
        (
            "official_dev_q5",
            _ok(DEV_Q5) and (DEV_Q5 / "threshold_selection.json").exists(),
            str(DEV_Q5),
        ),
        (
            "official_dev_q3",
            _ok(DEV_Q3) and (DEV_Q3 / "threshold_selection.json").exists(),
            str(DEV_Q3),
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official SciREX DEV multi-question Q5 then Q3 with threshold selection."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Re-run steps that already have clean outputs")
    parser.add_argument(
        "--only",
        choices=STEPS,
        nargs="*",
        default=(),
        help="Run only these steps (still in listed order). Default: remaining unfinished steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which steps would run and exit without launching models.",
    )
    return parser


def _should(name: str, args: argparse.Namespace, done: bool) -> bool:
    if args.only and name not in args.only:
        return False
    if done and not args.force:
        return False
    return True


def _assert_isolated() -> None:
    if DEV_Q5.resolve() == (LEGACY_ROOT / "dev_q5").resolve():
        raise SystemExit("refusing to write official DEV Q5 into the legacy hash-split directory")
    if DEV_Q3.resolve() == (LEGACY_ROOT / "dev_q3").resolve():
        raise SystemExit("refusing to write official DEV Q3 into the legacy hash-split directory")
    if THRESHOLDS.resolve() == (LEGACY_ROOT / "frozen_thresholds.json").resolve():
        raise SystemExit("refusing to overwrite legacy frozen_thresholds.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not PREPROCESS.exists():
        raise SystemExit(f"missing official DEV preprocessing: {PREPROCESS}")
    _assert_isolated()
    ROOT.mkdir(parents=True, exist_ok=True)
    common = [
        "--device",
        str(args.device),
        "--cache-path",
        str(CACHE),
        "--input-dir",
        str(PREPROCESS),
        "--split",
        "dev",
        "--select-threshold",
    ]
    if PHASE1_DEV.exists():
        common.extend(["--canonical-predictions", str(PHASE1_DEV)])

    print("=== official SciREX DEV multi-question ===")
    print(f"input: {PREPROCESS}")
    print(f"cache: {CACHE}")
    print(f"thresholds: {THRESHOLDS}")
    print(f"device: {args.device}")
    print("legacy hash-split DEV artifacts are not modified.")
    print("status:")
    for name, done, path in step_status():
        mark = "done" if done else "pending"
        print(f"  [{mark:7}] {name:16}  {path}")
    planned = [name for name, done, _ in step_status() if _should(name, args, done)]
    if args.dry_run:
        print("dry-run planned:", planned or "(nothing)")
        print("TEST is never run by this script.")
        return 0

    if _should("official_dev_q5", args, _ok(DEV_Q5) and (DEV_Q5 / "threshold_selection.json").exists()):
        print("\n[1/2] official DEV Q5 + threshold selection")
        _run(["--mode", "Q5", "--output-dir", str(DEV_Q5), *common])
    else:
        print("\n[1/2] skip official DEV Q5 (already complete)")

    if _should("official_dev_q3", args, _ok(DEV_Q3) and (DEV_Q3 / "threshold_selection.json").exists()):
        print("\n[2/2] official DEV Q3 + threshold selection (reuses Q5 cache)")
        _run(["--mode", "Q3", "--output-dir", str(DEV_Q3), *common])
    else:
        print("\n[2/2] skip official DEV Q3")

    thresholds = merge_official_thresholds()
    print(
        "official frozen thresholds:",
        {key: thresholds[key]["chosen_threshold"] for key in ("Q0", "Q3", "Q5") if key in thresholds},
    )
    print("legacy frozen_thresholds.json was not rewritten.")
    print("TEST was not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
