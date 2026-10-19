"""CLI with no test-unlock, retuning, seed-selection, or overwrite switches."""
import argparse
from pathlib import Path
from .config import ExperimentConfigV44
from .io_guard import validate_output


def build_parser():
    parser = argparse.ArgumentParser(description="Isolated V4.4 development stability replay; INTERNAL_TEST stays locked")
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfigV44().output_dir)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Synthetic data and hash encoder; no real dataset/model access")
    parser.add_argument("--no-support", action="store_true", help="Disable conditional support diagnostic before replay")
    parser.add_argument("--soft", action="store_true", help="Enable optional ASOFT diagnostic, only if H1 passes")
    parser.add_argument("--check-output", action="store_true", help="Validate fresh output path and exit without creating it")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_output(args.output_dir)
        if args.check_output:
            return 0
        from .pipeline import PipelineV44
        config = ExperimentConfigV44(args.output_dir, args.device, args.smoke, not args.no_support, args.soft)
        return PipelineV44(config).run()
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"V4.4: {exc}\n")
