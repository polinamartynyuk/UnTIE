"""CLI for isolated QA-based V3.1 silver overlay preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, V2_OUTPUT
from experiments.qa_chunk_selection.v3_1_prep.contract import V3_1_PREP_ARTIFACT_ROOT

from .config import AutoPrepConfig, parse_stages
from .contract import V3_1_AUTO_ARTIFACT_ROOT, V3_1_RANKING_ARTIFACT_ROOT, V3_ARTIFACT_ROOT
from .pipeline import PipelineAutoPrep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the QA-based V3.1 silver overlay. Writes only to artifacts/qa_chunk_selection_v3_1_auto/."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / V3_1_AUTO_ARTIFACT_ROOT)
    parser.add_argument("--v3-artifact-dir", type=Path, default=PROJECT_ROOT / V3_ARTIFACT_ROOT)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-candidates", type=int, default=0)
    parser.add_argument("--allow-partial-freeze", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--reuse-stage-a", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> AutoPrepConfig:
    reuse = Path(args.reuse_stage_a) if args.reuse_stage_a else None
    return AutoPrepConfig(
        output_dir=Path(args.output_dir),
        v3_artifact_dir=Path(args.v3_artifact_dir),
        stages=parse_stages(args.stages),
        device=str(args.device),
        seed=int(args.seed),
        limit_candidates=int(args.limit_candidates),
        allow_partial_freeze=bool(args.allow_partial_freeze),
        allow_existing_output=bool(args.allow_existing_output),
        reuse_stage_a_dir=reuse,
        resume=not bool(args.no_resume),
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
        "V3.1 ranking": (PROJECT_ROOT / V3_1_RANKING_ARTIFACT_ROOT).resolve(),
        "V4": (PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v4").resolve(),
    }
    for name, path in forbidden.items():
        if out == path or path in out.parents:
            print(f"Refusing to write QA silver prep into {name} artifacts: {path}")
            return 2
    if config.reuse_stage_a_dir is not None and Path(config.reuse_stage_a_dir).resolve() == out:
        print("Refusing reuse_stage_a_dir == output_dir. Use a new output directory.")
        return 2
    if (out / "analysis" / "readiness.json").exists() and not config.allow_existing_output:
        print(
            f"Refusing to write into completed prep {out}. "
            "Use a new OUT_DIR so the previous run is not overwritten."
        )
        return 2
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"v3_artifacts={config.v3_artifact_dir} (read-only)", flush=True)
    print(f"reuse_stage_a={config.reuse_stage_a_dir}", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke} device={config.device} annotator=frozen_extractive_qa", flush=True)
    return PipelineAutoPrep(config).run(list(config.stages))
