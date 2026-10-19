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


def required_paths(spec):
    paths = {
        "prepared_marker": spec.prepared_artifact / "COMPLETED.json",
        "reference_marker": spec.reference_artifact / "COMPLETED.json",
        "records": spec.prepared_artifact / spec.records_rel,
        "splits": spec.prepared_artifact / spec.splits_rel,
        "flags": spec.prepared_artifact / spec.flags_rel,
        "qcache": spec.reference_artifact / spec.qcache_rel,
        "qpred": spec.reference_artifact / spec.qpred_rel,
        "current_b0": spec.reference_artifact / spec.current_b0_rel,
    }
    if spec.chunk_labels_rel:
        paths["chunk_labels"] = spec.reference_artifact / spec.chunk_labels_rel
    return paths


def validate_dataset(spec):
    paths = required_paths(spec)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete input for {spec.key}: {missing}")
    return {name: path.resolve() for name, path in paths.items()}


def claim_output(path, *, resume, fingerprint):
    output = Path(path).resolve()
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    if artifacts not in output.parents:
        raise ValueError("output must be a new directory below project artifacts/")
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
