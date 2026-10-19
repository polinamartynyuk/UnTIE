"""CLI for a fresh, isolated Russian V4.4 development replay."""
import argparse
from datetime import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from .config import Config
from .contract import DEFAULT_INPUT_ARTIFACT
from .io_guard import validate_input, validate_output


def default_output():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "artifacts" / f"qa_chunk_selection_v4_4_ru_{stamp}"


def build_parser():
    parser = argparse.ArgumentParser(description="Isolated Russian V4.4 cross-fitted development replay")
    parser.add_argument("--input-artifact", type=Path, default=DEFAULT_INPUT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Fresh output directory; default has a timestamp and never overwrites a run")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-support", action="store_true", help="Skip the conditional support diagnostic")
    parser.add_argument("--smoke", action="store_true", help="Synthetic data and hash encoder; no model or real data")
    parser.add_argument("--check-only", action="store_true", help="Validate paths and inputs, then exit")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output_dir or default_output()
    try:
        if not args.smoke:
            validate_input(args.input_artifact)
        validate_output(output, args.input_artifact, smoke=args.smoke)
        if args.check_only:
            print(f"OK: input={args.input_artifact.resolve()}")
            print(f"OK: fresh output={output.resolve()}")
            return 0
        from .pipeline import Pipeline
        return Pipeline(Config(input_artifact=args.input_artifact, output_dir=output,
                               device=args.device, support=not args.no_support,
                               soft=False, smoke=args.smoke)).run()
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"V4.4-RU: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
