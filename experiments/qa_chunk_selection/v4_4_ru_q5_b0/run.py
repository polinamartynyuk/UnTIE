from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .config import Config


def build_parser():
    parser = argparse.ArgumentParser(description="Russian Q5 + answerable-only CV + three-seed B0 OOF")
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--b0-model", default="DeepPavlov/rubert-base-cased-sentence")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--smoke", action="store_true", help="Use TF-IDF B0; Q5 still uses frozen Russian models.")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = Config(
        base_artifact=args.base_artifact, output_dir=args.output_dir, device=args.device,
        seeds=tuple(int(item) for item in args.seeds.split(",") if item.strip()),
        b0_model=args.b0_model, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_length=args.max_length, smoke=args.smoke, resume=args.resume,
    )
    pipeline = None
    try:
        from .pipeline import Pipeline
        pipeline = Pipeline(config)
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

