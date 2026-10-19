import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, write_json
from .config import Config
from .contract import DEFAULT_INPUT, MODEL_SPECS
from .io_guard import claim_output, validate_input


def default_output():
    return PROJECT_ROOT / "artifacts" / f"qa_chunk_selection_v4_4_ru_ablation_{datetime.now():%Y%m%d_%H%M%S}"


def parser():
    out = argparse.ArgumentParser(description="Isolated Russian V4.4 hypothesis and cross-encoder ablations")
    out.add_argument("--input-artifact", type=Path, default=DEFAULT_INPUT)
    out.add_argument("--output-dir", type=Path)
    out.add_argument("--phase", choices=("protocol", "cross-encoders", "all"), default="all")
    out.add_argument("--device", default="auto")
    out.add_argument("--models", default=",".join(key for key, _ in MODEL_SPECS),
                     help="Comma-separated model ids: bge_m3")
    out.add_argument("--k-values", default="3,4,5")
    out.add_argument("--epochs", type=int, default=5)
    out.add_argument("--patience", type=int, default=2)
    out.add_argument("--micro-batch-size", type=int, default=2)
    out.add_argument("--gradient-accumulation", type=int, default=8)
    out.add_argument("--learning-rate", type=float, default=1e-5)
    out.add_argument("--max-length", type=int, default=512)
    out.add_argument("--resume", action="store_true")
    out.add_argument("--smoke", action="store_true")
    out.add_argument("--check-only", action="store_true")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    known = dict(MODEL_SPECS)
    ids = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = sorted(set(ids) - set(known))
    if unknown:
        raise SystemExit(f"Unknown model ids: {unknown}")
    source = validate_input(args.input_artifact)
    output = args.output_dir or default_output()
    config = Config(input_artifact=source, output_dir=output, phase=args.phase,
                    device=args.device, models=tuple((key, known[key]) for key in ids),
                    k_values=tuple(int(x) for x in args.k_values.split(",") if x.strip()),
                    epochs=args.epochs, patience=args.patience,
                    micro_batch_size=args.micro_batch_size,
                    gradient_accumulation=args.gradient_accumulation,
                    learning_rate=args.learning_rate, max_length=args.max_length,
                    resume=args.resume, smoke=args.smoke)
    config.validate()
    payload = config.to_dict()
    payload.pop("resume", None)
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if args.check_only:
        if output.exists() and not args.resume:
            raise SystemExit(f"Output already exists: {output}")
        print(f"OK input: {source}")
        print(f"OK output: {output.resolve()}")
        print(f"OK phase={config.phase}; models={ids}; K={config.k_values}")
        return 0
    claimed = claim_output(output, source, resume=args.resume, fingerprint=fingerprint)
    write_json(claimed / "config_resolved.json", {**config.to_dict(), "fingerprint": fingerprint})
    from .pipeline import Pipeline
    return Pipeline(config, claimed).run()


if __name__ == "__main__":
    raise SystemExit(main())

