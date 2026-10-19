from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.final_official_split_benchmark.io_guard import refuse_overwrite

from .pipeline import Pipeline


DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/final_official_split_benchmark/run_20260930_175541"
DEFAULT_MODEL = "smangla/ModernBERT-base-squad2"
DEFAULT_REVISION = "f80f767241e36220e9aa805886ab3ae035349bc3"


def parser():
    value = argparse.ArgumentParser(description="Post-hoc frozen ModernBERT-SQuAD2 baselines")
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--output", type=Path)
    value.add_argument("--cells", nargs="+", default=["scirex", "scier_384", "scier_128"])
    value.add_argument("--model-id", default=DEFAULT_MODEL)
    value.add_argument("--model-revision", default=DEFAULT_REVISION)
    value.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    value.add_argument("--device", default="auto")
    value.add_argument("--max-length", type=int, default=512)
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--bootstrap", type=int, default=10_000)
    value.add_argument("--allow-downloads", action="store_true")
    value.add_argument("--resume", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "artifacts/final_official_split_benchmark_posthoc_squad" / f"run_{stamp}"
    refuse_overwrite(args.output, args.resume)
    return Pipeline(args).run()


if __name__ == "__main__":
    raise SystemExit(main())

