import argparse
import hashlib
import json
import traceback
from datetime import datetime
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, write_json

from .audit import audit_dataset
from .config import Config
from .contract import DATASETS, EXPERIMENT_ID, RANKERS
from .io_guard import claim_output, validate_inputs


def default_output():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_5_ru" / f"run_{stamp}"


def _csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parser():
    out = argparse.ArgumentParser(description="V4.5-RU compact ranker plus frozen adaptive residual correction")
    out.add_argument("--datasets", default="russerrc,scimdix")
    out.add_argument("--phase", choices=("audit", "rankers", "stage2", "benchmark", "analysis", "all"), default="all")
    out.add_argument("--output-dir", type=Path)
    out.add_argument("--device", default="auto")
    out.add_argument("--epochs", type=int, default=5)
    out.add_argument("--patience", type=int, default=2)
    out.add_argument("--micro-batch-size", type=int, default=2)
    out.add_argument("--prediction-batch-size", type=int, default=8)
    out.add_argument("--gradient-accumulation", type=int, default=8)
    out.add_argument("--learning-rate", type=float, default=1e-5)
    out.add_argument("--max-length", type=int, default=512)
    out.add_argument("--bootstrap", type=int, default=5000)
    out.add_argument("--benchmark-questions", type=int, default=20)
    out.add_argument("--benchmark-warmup", type=int, default=10)
    out.add_argument("--benchmark-repetitions", type=int, default=10)
    out.add_argument("--dity-throughput-batch", type=int, default=16)
    out.add_argument("--bge-throughput-batch", type=int, default=8)
    out.add_argument("--local-files-only", action="store_true")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--smoke", action="store_true")
    out.add_argument("--check-only", action="store_true")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    datasets = _csv(args.datasets)
    if args.smoke:
        datasets = datasets[:1]
    output = args.output_dir or default_output()
    config = Config(
        output_dir=output, datasets=datasets, phase=args.phase, device=args.device,
        seeds=(42,) if args.smoke else (42, 43, 44), epochs=args.epochs,
        patience=args.patience, micro_batch_size=args.micro_batch_size,
        prediction_batch_size=args.prediction_batch_size,
        gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
        max_length=args.max_length, n_bootstrap=args.bootstrap,
        benchmark_questions=args.benchmark_questions,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_repetitions=args.benchmark_repetitions,
        dity_throughput_batch=args.dity_throughput_batch,
        bge_throughput_batch=args.bge_throughput_batch,
        local_files_only=args.local_files_only, resume=args.resume, smoke=args.smoke,
    )
    config.validate()
    for key in config.datasets:
        validate_inputs(DATASETS[key])
    if args.check_only:
        for key in config.datasets:
            result = audit_dataset(DATASETS[key])
            print(f"OK {key}: {result['manifest']['questions']} questions, {result['manifest']['documents']} documents, K={DATASETS[key].k}")
        print(f"OK DiTy: {RANKERS['dity'].model_name}@{RANKERS['dity'].revision}")
        print(f"OK BGE: {RANKERS['bge'].model_name}@{RANKERS['bge'].revision}")
        print(f"OK output would be new and isolated: {output.resolve()}")
        print("No model was loaded, no artifact was written, and no experiment was run.")
        return 0
    fingerprint = hashlib.sha256(json.dumps(config.fingerprint_payload(), ensure_ascii=False,
                                               sort_keys=True).encode("utf-8")).hexdigest()
    claimed = claim_output(output, resume=args.resume, fingerprint=fingerprint)
    write_json(claimed / "config_resolved.json", {**config.to_dict(), "fingerprint": fingerprint})
    write_json(claimed / "status.json", {"status": "running", "experiment_id": EXPERIMENT_ID,
                                          "phase": config.phase, "datasets": list(config.datasets)})
    try:
        from .pipeline import Pipeline
        code = Pipeline(config, claimed).run()
        write_json(claimed / "status.json", {"status": "complete", "exit_code": code,
                                              "phase": config.phase, "datasets": list(config.datasets)})
        return code
    except Exception as exc:
        write_json(claimed / "status.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

