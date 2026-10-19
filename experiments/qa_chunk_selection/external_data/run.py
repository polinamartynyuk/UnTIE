"""CLI for isolated SciMDIX and SciER preparation."""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import ExternalDataConfig


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / f"qa_chunk_selection_external_data_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SciMDIX and SciER for chunk-selection experiments")
    parser.add_argument("--output-dir", type=Path, default=None, help="Must be a new directory below artifacts/.")
    parser.add_argument("--datasets", nargs="+", choices=("scimdix", "scier"), default=("scimdix", "scier"))
    parser.add_argument("--scimdix-source", type=Path, help="Local SciMDIX snapshot or files directory.")
    parser.add_argument("--scier-source", type=Path, help="Local SciER repository or PLM directory.")
    parser.add_argument("--no-download", action="store_true", help="Use both supplied local source paths.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scimdix-chunk-max-tokens", type=int, default=128)
    parser.add_argument("--scimdix-overlap-tokens", type=int, default=24)
    parser.add_argument("--scier-chunk-max-tokens", type=int, default=384)
    parser.add_argument("--scier-overlap-tokens", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--gap-bootstrap", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit-docs", type=int, default=0, help="Diagnostic documents per official split; 0 means all.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExternalDataConfig(
        output_dir=args.output_dir or _default_output(), datasets=tuple(args.datasets),
        scimdix_source=args.scimdix_source, scier_source=args.scier_source,
        download=not args.no_download, device=args.device, seed=args.seed,
        scimdix_chunk_max_tokens=args.scimdix_chunk_max_tokens,
        scimdix_overlap_tokens=args.scimdix_overlap_tokens,
        scier_chunk_max_tokens=args.scier_chunk_max_tokens,
        scier_overlap_tokens=args.scier_overlap_tokens,
        hard_negative_top_n=args.top_n, gap_bootstrap=args.gap_bootstrap,
        n_folds=args.n_folds, limit_docs=args.limit_docs,
    )
    pipeline = None
    try:
        from .pipeline import ExternalDataPipeline
        pipeline = ExternalDataPipeline(config)
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
