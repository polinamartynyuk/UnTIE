"""CLI for isolated V3.1-Silver ranking.

Writes only to artifacts/qa_chunk_selection_v3_1/.
Never overwrites E0, V2, V3, manual prep, or automated silver trees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, V2_OUTPUT
from experiments.qa_chunk_selection.v3_1_auto_prep.contract import V3_1_AUTO_ARTIFACT_ROOT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

from .config import ExperimentConfigV31, parse_stages
from .contract import V3_1_ARTIFACT_ROOT
from .pipeline import PipelineV31


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V3.1-Silver ranking on a frozen automated overlay. "
            "Writes only to artifacts/qa_chunk_selection_v3_1/."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / V3_1_ARTIFACT_ROOT)
    parser.add_argument("--v3-artifact-dir", type=Path, default=DEFAULT_V3_OUTPUT)
    parser.add_argument("--v2-artifact-dir", type=Path, default=V2_OUTPUT)
    parser.add_argument("--silver-dir", type=Path, default=PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-r1", action="store_true")
    parser.add_argument("--allow-internal-test", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfigV31:
    if args.allow_internal_test:
        raise PermissionError("INTERNAL_TEST is locked for V3.1 until freeze-final.")
    return ExperimentConfigV31(
        output_dir=Path(args.output_dir),
        v3_artifact_dir=Path(args.v3_artifact_dir),
        v2_artifact_dir=Path(args.v2_artifact_dir),
        silver_dir=Path(args.silver_dir),
        device=str(args.device),
        seed=int(args.seed),
        stages=parse_stages(args.stages),
        skip_qa=bool(args.skip_qa),
        smoke=bool(args.smoke),
        resume=bool(args.resume),
        force_r1=bool(args.force_r1),
        allow_internal_test=False,
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
    }
    for name, path in forbidden.items():
        if out == path or path in out.parents:
            print(f"Refusing to write V3.1 ranking into {name} artifacts: {path}")
            return 2
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"v3_artifacts={config.v3_artifact_dir} (read-only)", flush=True)
    print(f"silver_dir={config.silver_dir} (read-only)", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke} device={config.device} b0_epochs={config.b0_epochs}", flush=True)
    return PipelineV31(config).run()
