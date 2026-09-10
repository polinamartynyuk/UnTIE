"""CLI for isolated V3.1 target-status overlay preparation.

Writes only to artifacts/qa_chunk_selection_v3_1_prep/.
Never overwrites E0, V2, or V3 experiment artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

from .config import PrepConfig, parse_stages
from .contract import V3_1_PREP_ARTIFACT_ROOT, V3_ARTIFACT_ROOT
from .pipeline import PipelinePrep

# Avoid circular imports: resolve V3 path from contract.
from experiments.qa_chunk_selection.v3.io_util import DEFAULT_V3_OUTPUT, E0_OUTPUT, V2_OUTPUT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the V3.1 target-status overlay. "
            "Reads V3 artifacts read-only. Writes only to artifacts/qa_chunk_selection_v3_1_prep/."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / V3_1_PREP_ARTIFACT_ROOT)
    parser.add_argument("--v3-artifact-dir", type=Path, default=PROJECT_ROOT / V3_ARTIFACT_ROOT)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b0-seed", type=int, default=42)
    parser.add_argument("--annotator-id", default="annotator_1")
    parser.add_argument("--coverage-completion-n", type=int, default=0)
    parser.add_argument("--coverage-completion-k", type=int, default=10)
    parser.add_argument("--min-train-questions-with-tm", type=int, default=10)
    parser.add_argument("--min-eval-questions-with-target-and-tm", type=int, default=10)
    parser.add_argument("--annotations-glob", default="drafts/*.jsonl")
    parser.add_argument("--allow-partial-freeze", action="store_true")
    parser.add_argument("--keep-low-confidence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PrepConfig:
    stages = parse_stages(args.stages)
    if int(args.coverage_completion_n) > 0 and "expand-coverage" not in stages:
        stages = tuple(["init", "export-annotator", "expand-coverage"] + [s for s in stages if s not in {"init", "export-annotator"}])
    return PrepConfig(
        output_dir=Path(args.output_dir),
        v3_artifact_dir=Path(args.v3_artifact_dir),
        stages=stages,
        seed=int(args.seed),
        b0_seed=int(args.b0_seed),
        annotator_id=str(args.annotator_id),
        coverage_completion_n=int(args.coverage_completion_n),
        coverage_completion_k=int(args.coverage_completion_k),
        min_train_questions_with_TM=int(args.min_train_questions_with_tm),
        min_eval_questions_with_target_and_TM=int(args.min_eval_questions_with_target_and_tm),
        allow_partial_freeze=bool(args.allow_partial_freeze),
        keep_low_confidence=bool(args.keep_low_confidence),
        annotations_glob=str(args.annotations_glob),
        smoke=bool(args.smoke),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    out = Path(config.output_dir).resolve()
    if out == E0_OUTPUT.resolve() or E0_OUTPUT.resolve() in out.parents:
        print("Refusing to write V3.1 prep into E0 artifacts.")
        return 2
    if out == V2_OUTPUT.resolve() or V2_OUTPUT.resolve() in out.parents:
        print("Refusing to write V3.1 prep into V2 artifacts.")
        return 2
    if out == DEFAULT_V3_OUTPUT.resolve() or DEFAULT_V3_OUTPUT.resolve() in out.parents:
        print("Refusing to write V3.1 prep into V3 artifacts. Use artifacts/qa_chunk_selection_v3_1_prep/.")
        return 2
    v4_output = (PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v4").resolve()
    if out == v4_output or v4_output in out.parents:
        print("Refusing to write V3.1 prep into V4 artifacts.")
        return 2
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"output_dir={config.output_dir}", flush=True)
    print(f"v3_artifacts={config.v3_artifact_dir} (read-only)", flush=True)
    print(f"stages={','.join(config.stages)}", flush=True)
    print(f"smoke={config.smoke}", flush=True)
    return PipelinePrep(config).run(list(config.stages))
