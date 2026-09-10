"""Resume the multi-question experiment from the last finished step.

Does not rewrite preprocessing, does not run TEST, does not retouch gold.

Usage from the project root:

    ./untie_venv/bin/python -m experiments.phase1_multiquestion.continue --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .run import main as run_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "artifacts" / "phase1_multiquestion"
PREPROCESS = PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing"
CANONICAL = PROJECT_ROOT / "artifacts" / "phase1_scirex" / "train" / "qa_chunk_dataset.parquet"
CACHE = ROOT / "qa_prediction_cache.jsonl"
THRESHOLDS = ROOT / "frozen_thresholds.json"

STEPS = (
    "debug_q5",
    "dev_q5",
    "dev_q3",
    "train_q0",
    "train_q5",
    "train_q3",
    "compare",
)


def _ok(output_dir: Path) -> bool:
    stats = output_dir / "statistics.json"
    sanity = output_dir / "sanity_errors.json"
    if not stats.exists() or not sanity.exists():
        return False
    errors = json.loads(sanity.read_text(encoding="utf-8"))
    return errors == []


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_thresholds() -> dict:
    payload = {
        "Q0": {
            "chosen_threshold": 1.0,
            "source": "fixed: Q0 has only 0/1 consistency",
        }
    }
    q5 = ROOT / "dev_q5" / "threshold_selection.json"
    q3 = ROOT / "dev_q3" / "threshold_selection.json"
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
    debug_dir = ROOT / "debug_q5"
    dev_q5 = ROOT / "dev_q5"
    dev_q3 = ROOT / "dev_q3"
    train_q0 = ROOT / "train_q0"
    train_q5 = ROOT / "train_q5"
    train_q3 = ROOT / "train_q3"
    comparison = ROOT / "q0_q3_q5_comparison.json"
    return [
        ("debug_q5", _ok(debug_dir), str(debug_dir)),
        (
            "dev_q5",
            _ok(dev_q5) and (dev_q5 / "threshold_selection.json").exists(),
            str(dev_q5),
        ),
        (
            "dev_q3",
            _ok(dev_q3) and (dev_q3 / "threshold_selection.json").exists(),
            str(dev_q3),
        ),
        ("train_q0", _ok(train_q0), str(train_q0)),
        ("train_q5", _ok(train_q5), str(train_q5)),
        ("train_q3", _ok(train_q3), str(train_q3)),
        ("compare", comparison.exists(), str(comparison)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue SciREX multi-question Phase 1 from the last completed step."
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
    parser.add_argument("--skip-compare", action="store_true")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not PREPROCESS.exists():
        raise SystemExit(f"missing preprocessing: {PREPROCESS}")
    if not CANONICAL.exists():
        raise SystemExit(f"missing Q0 canonical TRAIN predictions: {CANONICAL}")
    ROOT.mkdir(parents=True, exist_ok=True)
    common = [
        "--device",
        str(args.device),
        "--cache-path",
        str(CACHE),
        "--canonical-predictions",
        str(CANONICAL),
        "--input-dir",
        str(PREPROCESS),
    ]

    print("=== multi-question continue ===")
    print(f"cache: {CACHE}")
    print(f"device: {args.device}")
    print("status:")
    for name, done, path in step_status():
        mark = "done" if done else "pending"
        print(f"  [{mark:7}] {name:8}  {path}")
    if args.dry_run:
        planned = [
            name
            for name, done, _ in step_status()
            if _should(name, args, done) and not (name == "compare" and args.skip_compare)
        ]
        print("dry-run planned:", planned or "(nothing)")
        print("TEST is never run by this script.")
        return 0

    debug_dir = ROOT / "debug_q5"
    if _should("debug_q5", args, _ok(debug_dir)):
        print("\n[1/7] DEBUG Q5 on 8 TRAIN documents")
        _run(
            [
                "--mode",
                "Q5",
                "--split",
                "train",
                "--debug",
                "--debug-docs",
                "8",
                "--output-dir",
                str(debug_dir),
                *common,
            ]
        )
    else:
        print("\n[1/7] skip DEBUG Q5 (already complete)")

    dev_q5 = ROOT / "dev_q5"
    if _should("dev_q5", args, _ok(dev_q5) and (dev_q5 / "threshold_selection.json").exists()):
        print("\n[2/7] DEV Q5 + threshold selection")
        _run(
            [
                "--mode",
                "Q5",
                "--split",
                "dev",
                "--select-threshold",
                "--output-dir",
                str(dev_q5),
                *common,
            ]
        )
    else:
        print("\n[2/7] skip DEV Q5 (already complete)")

    dev_q3 = ROOT / "dev_q3"
    if _should("dev_q3", args, _ok(dev_q3) and (dev_q3 / "threshold_selection.json").exists()):
        print("\n[3/7] DEV Q3 + threshold selection (reuses Q5 DEV cache)")
        _run(
            [
                "--mode",
                "Q3",
                "--split",
                "dev",
                "--select-threshold",
                "--output-dir",
                str(dev_q3),
                *common,
            ]
        )
    else:
        print("\n[3/7] skip DEV Q3")

    thresholds = merge_thresholds()
    need_q5 = _should("train_q5", args, _ok(ROOT / "train_q5"))
    need_q3 = _should("train_q3", args, _ok(ROOT / "train_q3"))
    if need_q5 and "Q5" not in thresholds:
        raise SystemExit("Q5 DEV threshold is missing; cannot freeze TRAIN Q5")
    if need_q3 and "Q3" not in thresholds:
        raise SystemExit("Q3 DEV threshold is missing; run step dev_q3 first")
    print(
        "frozen thresholds:",
        {key: thresholds[key]["chosen_threshold"] for key in ("Q0", "Q3", "Q5") if key in thresholds},
    )

    train_q0 = ROOT / "train_q0"
    if _should("train_q0", args, _ok(train_q0)):
        print("\n[4/7] TRAIN Q0 (canonical question, reuse stored spans)")
        _run(
            [
                "--mode",
                "Q0",
                "--split",
                "train",
                "--threshold-file",
                str(THRESHOLDS),
                "--output-dir",
                str(train_q0),
                *common,
            ]
        )
    else:
        print("\n[4/7] skip TRAIN Q0")

    train_q5 = ROOT / "train_q5"
    if _should("train_q5", args, _ok(train_q5)):
        print("\n[5/7] TRAIN Q5 (longest step; q0 reused, q1–q4 from cache or frozen QA)")
        _run(
            [
                "--mode",
                "Q5",
                "--split",
                "train",
                "--threshold-file",
                str(THRESHOLDS),
                "--output-dir",
                str(train_q5),
                *common,
            ]
        )
    else:
        print("\n[5/7] skip TRAIN Q5")

    train_q3 = ROOT / "train_q3"
    if _should("train_q3", args, _ok(train_q3)):
        print("\n[6/7] TRAIN Q3 (should be cache-only after Q5)")
        _run(
            [
                "--mode",
                "Q3",
                "--split",
                "train",
                "--threshold-file",
                str(THRESHOLDS),
                "--output-dir",
                str(train_q3),
                *common,
            ]
        )
    else:
        print("\n[6/7] skip TRAIN Q3")

    comparison = ROOT / "q0_q3_q5_comparison.json"
    if args.skip_compare or (args.only and "compare" not in args.only):
        print("\n[7/7] skip compare")
    elif comparison.exists() and not args.force and _ok(train_q0) and _ok(train_q3) and _ok(train_q5):
        print("\n[7/7] skip compare (file exists; pass --force to rebuild)")
    else:
        if not (_ok(train_q0) and _ok(train_q3) and _ok(train_q5)):
            raise SystemExit("compare needs complete TRAIN Q0, Q3 and Q5")
        print("\n[7/7] compare Q0 vs Q3 vs Q5")
        _run(
            [
                "--output-dir",
                str(ROOT),
                "--compare",
                f"Q0={train_q0}",
                f"Q3={train_q3}",
                f"Q5={train_q5}",
            ]
        )

    print("\nTEST was not run.")
    print("outputs:", ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
