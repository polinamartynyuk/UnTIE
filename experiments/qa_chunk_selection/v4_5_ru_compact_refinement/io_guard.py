import hashlib
import json
import math
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT, write_json

from .contract import EXPERIMENT_ID


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(values):
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def required_paths(spec):
    paths = {
        "prepared_marker": spec.prepared_artifact / "COMPLETED.json",
        "reference_marker": spec.reference_artifact / "COMPLETED.json",
        "records": spec.prepared_artifact / spec.records_rel,
        "splits": spec.prepared_artifact / spec.splits_rel,
        "flags": spec.prepared_artifact / spec.flags_rel,
        "qcache": spec.reference_artifact / spec.qcache_rel,
        "historical_b0": spec.reference_artifact / spec.current_b0_rel,
        "historical_dity": spec.historical_factorial_root / "cross_encoders/dity_msmarco/oof_predictions_scored.jsonl",
        "historical_factorial_marker": spec.historical_factorial_root / "COMPLETED.json",
    }
    if spec.historical_bge_oof is not None:
        paths["historical_bge"] = spec.historical_bge_oof
    return paths


def validate_inputs(spec):
    paths = required_paths(spec)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete historical input for {spec.key}: {missing}")
    if not spec.qa_model.is_dir():
        raise FileNotFoundError(f"frozen QA model missing: {spec.qa_model}")
    return {name: path.resolve() for name, path in paths.items()}


def claim_output(path, *, resume, fingerprint):
    output = Path(path).resolve()
    root = (PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_5_ru").resolve()
    if root not in output.parents:
        raise ValueError(f"output must be a new run directory below {root}")
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


def validate_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")



def null_undefined_pairwise(value):
    """Represent an inapplicable pairwise diagnostic as JSON null.

    The shared ranking metrics intentionally return NaN when an instance or an
    aggregate contains no positive/hard-negative pairs. That is missing
    diagnostic information, rather than a non-finite model score. V4.5 keeps
    strict finite checks for every other field and serializes only this known
    undefined diagnostic as null.
    """
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("pairwise_") and isinstance(item, float) and not math.isfinite(item):
                normalized[key] = None
            else:
                normalized[key] = null_undefined_pairwise(item)
        return normalized
    if isinstance(value, list):
        return [null_undefined_pairwise(item) for item in value]
    if isinstance(value, tuple):
        return tuple(null_undefined_pairwise(item) for item in value)
    return value
