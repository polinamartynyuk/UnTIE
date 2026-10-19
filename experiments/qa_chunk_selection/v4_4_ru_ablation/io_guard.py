import hashlib
import json
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, write_json
from .contract import EXPERIMENT_ID


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(root):
    root = Path(root).resolve()
    required = ["COMPLETED.json", "data/answerable_dataset.jsonl",
                "data/answerable_cv_splits.json", "data/hn_quality_flags.jsonl",
                "b0/oof_predictions_scored.jsonl", "q5/prediction_cache.jsonl",
                "q5/question_predictions.parquet", "q5/chunk_labels.parquet"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Russian Q5/B0 input: {missing}")
    return root


def claim_output(path, source, *, resume, fingerprint):
    output = Path(path).resolve()
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    source = Path(source).resolve()
    if artifacts not in output.parents or output == source or source in output.parents:
        raise ValueError("output must be a separate directory below project artifacts/")
    if any(item.is_symlink() for item in (output, *output.parents)):
        raise ValueError("output path cannot contain symlinks")
    marker = output / "config.json"
    if output.exists():
        if not resume or (output / "COMPLETED.json").exists() or not marker.is_file():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("experiment_id") != EXPERIMENT_ID or previous.get("fingerprint") != fingerprint:
            raise ValueError("resume fingerprint mismatch")
    else:
        output.mkdir(parents=True)
        write_json(marker, {"experiment_id": EXPERIMENT_ID, "fingerprint": fingerprint})
    return output

