"""CLI for isolated D4 oracle + D5 audit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, V2_OUTPUT
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT, V3_1_RANKING_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

from .config import D45Config, parse_stages
from .contract import V3_D45_ARTIFACT_ROOT
from .pipeline import PipelineD45


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3 D4 oracle + D5 audit. Writes only to artifacts/qa_chunk_selection_v3_d45/.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / V3_D45_ARTIFACT_ROOT)
    parser.add_argument("--v3-artifact-dir", type=Path, default=DEFAULT_V3_OUTPUT)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> D45Config:
    return D45Config(
        output_dir=Path(args.output_dir),
        v3_artifact_dir=Path(args.v3_artifact_dir),
        stages=parse_stages(args.stages),
        device=str(args.device),
        seed=int(args.seed),
        smoke=bool(args.smoke),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    out = Path(config.output_dir).resolve()
    forbidden = {
        "E0": E0_OUTPUT.resolve(),
        "V2": V2_OUTPUT.resolve(),
        "V3": DEFAULT_V3_OUTPUT.resolve(),
        "manual V3.1 prep": (PROJECT_ROOT / V3_1_PREP_ARTIFACT_ROOT).resolve(),
        "automated silver": (PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT).resolve(),
        "V3.1 ranking": (PROJECT_ROOT / V3_1_RANKING_ARTIFACT_ROOT).resolve(),
        "V4": (PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v4").resolve(),
    }
    for name, path in forbidden.items():
        if out == path or path in out.parents:
            print(f"Refusing to write D4/D5 diagnostic into {name} artifacts: {path}")
            return 2
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"v3_artifacts={config.v3_artifact_dir} (read-only)", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke} device={config.device}", flush=True)
    return PipelineD45(config).run(list(config.stages))
