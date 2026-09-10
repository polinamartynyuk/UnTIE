#!/usr/bin/env python3
"""Mark existing SciREX aspect-QA tables as official train without dropping rows.

Keeps every parquet row. Copies the old hash-based split into legacy_internal_split
and sets split='train' on documents, aspect_instances, qa_input, and source_metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from untie.aspect_qa.config import default_output_dir
from untie.aspect_qa.relabel import relabel_preprocessing_as_official_train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir(PROJECT_ROOT),
    )
    args = parser.parse_args()
    report = relabel_preprocessing_as_official_train(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
