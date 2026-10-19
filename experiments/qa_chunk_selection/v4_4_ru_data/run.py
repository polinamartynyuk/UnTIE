"""Command line entry point for full RuSERRC preparation."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Sequence

from .config import ASPECT_SPECS, DEFAULT_OUTPUT, RuDataConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare all native RuSERRC aspects for V4.4")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-dir", type=Path,
        help="Directory with the three official aspects_data_part_*.csv files; disables download.",
    )
    source.add_argument("--structured-source", type=Path, help="Complete local ruserrc_structured.csv with inline tags.")
    source.add_argument("--download", action="store_true", help="Download the official corpus (default if no local source is given).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-max-tokens", type=int, default=128)
    parser.add_argument("--overlap-tokens", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--gap-bootstrap", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-docs", type=int, default=0, help="Local diagnostic slice; 0 means all.")
    parser.add_argument(
        "--aspects", default="all",
        help="Comma-separated native names, or all. Default: every RuSERRC aspect.",
    )
    return parser


def _parse_aspects(raw: str) -> tuple[str, ...]:
    known = tuple(spec[0] for spec in ASPECT_SPECS.values())
    if raw.strip().lower() == "all":
        return known
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RuDataConfig(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        structured_source=args.structured_source,
        download=args.source_dir is None and args.structured_source is None,
        device=args.device,
        seed=args.seed,
        chunk_max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.overlap_tokens,
        hard_negative_top_n=args.top_n,
        gap_bootstrap=args.gap_bootstrap,
        n_folds=args.n_folds,
        limit_docs=args.limit_docs,
        aspects=_parse_aspects(args.aspects),
    )
    pipeline = None
    try:
        from .pipeline import RuDataPipeline

        pipeline = RuDataPipeline(config)
        pipeline.run()
        return 0
    except Exception as error:
        if pipeline is not None:
            payload = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            (pipeline.output / "FAILED.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            pipeline.progress.log("FAILED", **payload)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
