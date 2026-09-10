#!/usr/bin/env python3
"""Convert official SciREX train/dev/test JSONL into structured JSONL files.

Does not overwrite datasets/scirex_structured.json when that file already exists.
That file is the official SciREX train split produced earlier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from untie.scirex_release import prepare_official_structured_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "scirex_release",
    )
    parser.add_argument(
        "--overwrite-train",
        action="store_true",
        help="Also rewrite scirex_structured.json. Default is to leave it untouched.",
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    skip = () if args.overwrite_train else ("train",)
    report = prepare_official_structured_splits(
        datasets_dir=args.datasets_dir,
        release_dir=args.release_dir,
        skip_existing=skip,
        download=not args.no_download,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
