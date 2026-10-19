from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from experiments.final_official_split_benchmark.contract import CELLS, SEEDS
from experiments.final_official_split_benchmark.io_guard import refuse_overwrite
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .pipeline import Pipeline


DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/final_official_split_benchmark/run_20260930_175541"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Post-hoc H3 zero-anchor agreement scaling ablation")
    value.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    value.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    value.add_argument("--inner-folds", type=int, choices=(3, 4), default=4)
    value.add_argument("--bootstrap", type=int, default=10_000)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.run_dir or PROJECT_ROOT / "artifacts/qa_chunk_selection_h3_zero_anchor_ablation" / f"run_{stamp}"
    refuse_overwrite(output, resume=False)
    return Pipeline(
        args.source_run,
        output,
        tuple(args.cells),
        tuple(args.seeds),
        args.inner_folds,
        args.bootstrap,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())

