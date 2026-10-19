import argparse
import hashlib
import json
import traceback
from datetime import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, write_json

from .config import Config
from .contract import (
    DATASET_SPECS, EXPERIMENT_ID, Q5_ROUTING_QUANTILE, RANKER_SPECS,
    ROUTING_QUANTILE, STRATEGIES, SYSTEMS,
)
from .io_guard import claim_output, validate_dataset


def default_output():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "artifacts" / f"qa_chunk_selection_v4_4_ru_rankers_{stamp}"


def _csv(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parser():
    out = argparse.ArgumentParser(
        description="Independent Russian V4.4 ranker x second-component factorial",
    )
    out.add_argument("--datasets", default=",".join(DATASET_SPECS))
    out.add_argument("--rankers", default=",".join(RANKER_SPECS))
    out.add_argument("--output-dir", type=Path)
    out.add_argument("--phase", choices=("rankers", "protocol", "analysis", "all"), default="all")
    out.add_argument("--device", default="auto")
    out.add_argument("--k-values", default="3,4,5")
    out.add_argument("--epochs", type=int, default=5)
    out.add_argument("--patience", type=int, default=2)
    out.add_argument("--micro-batch-size", type=int, default=2)
    out.add_argument("--prediction-batch-size", type=int, default=8)
    out.add_argument("--gradient-accumulation", type=int, default=8)
    out.add_argument("--learning-rate", type=float, default=1e-5)
    out.add_argument("--max-length", type=int, default=512)
    out.add_argument("--bootstrap", type=int, default=5000)
    out.add_argument("--local-files-only", action="store_true")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--smoke", action="store_true")
    out.add_argument("--check-only", action="store_true")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    output = args.output_dir or default_output()
    config = Config(
        output_dir=output,
        datasets=_csv(args.datasets),
        rankers=_csv(args.rankers),
        phase=args.phase,
        device=args.device,
        k_values=tuple(int(value) for value in _csv(args.k_values)),
        epochs=args.epochs,
        patience=args.patience,
        micro_batch_size=args.micro_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        n_bootstrap=args.bootstrap,
        local_files_only=args.local_files_only,
        resume=args.resume,
        smoke=args.smoke,
    )
    config.validate()
    inputs = {key: validate_dataset(DATASET_SPECS[key]) for key in config.datasets}
    fingerprint = hashlib.sha256(json.dumps(
        config.fingerprint_payload(), ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    if args.check_only:
        if output.exists() and not args.resume:
            raise SystemExit(f"Output already exists: {output}")
        print(f"OK datasets: {', '.join(config.datasets)}")
        print(f"OK rankers: {', '.join(config.rankers)}")
        print(f"OK phase={config.phase}; K={config.k_values}; seeds={config.seeds}")
        for key, paths in inputs.items():
            print(f"OK {key}: {len(paths)} required input files")
        print(f"OK output: {output.resolve()}")
        print("No model was loaded and no experiment was run.")
        return 0
    claimed = claim_output(output, resume=args.resume, fingerprint=fingerprint)
    write_json(claimed / "config_resolved.json", {
        **config.to_dict(), "fingerprint": fingerprint,
    })
    write_json(claimed / "contract.json", {
        "experiment_id": EXPERIMENT_ID,
        "design": "dataset x ranker x question_strategy x K x V4.4_system",
        "datasets": list(config.datasets),
        "rankers": {
            key: {
                "model": RANKER_SPECS[key].model_name,
                "revision": RANKER_SPECS[key].revision,
            }
            for key in config.rankers
        },
        "strategies": list(STRATEGIES),
        "k_values": list(config.k_values),
        "systems": list(SYSTEMS),
        "outer_cv": "five document-grouped folds; three seeds",
        "selection_source": "train+calibration only within each outer run",
        "routing_quantile": ROUTING_QUANTILE,
        "q5_routing_quantile": Q5_ROUTING_QUANTILE,
        "previous_results_read_only": True,
        "model_weights_retained": False,
    })
    try:
        from .pipeline import Pipeline
        return Pipeline(config, claimed).run()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

