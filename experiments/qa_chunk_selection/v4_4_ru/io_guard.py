"""Fresh-output guard. The Q5/B0 input and every older artifact stay read-only."""
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v4.io_guard import (
    file_sha256, load_json, read_jsonl, write_json, write_jsonl,
)


def validate_input(path: Path) -> Path:
    root = Path(path).resolve()
    required = (
        root / "COMPLETED.json",
        root / "data/answerable_dataset.jsonl",
        root / "data/answerable_cv_splits.json",
        root / "q5/prediction_cache.jsonl",
        root / "b0/oof_predictions_scored.jsonl",
    )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Russian Q5/B0 artifact is incomplete: {missing}")
    return root


def validate_output(path: Path, input_artifact: Path, *, smoke=False) -> Path:
    raw = Path(path).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("V4.4-RU output cannot use symlinks")
    output, project, source = raw.resolve(), PROJECT_ROOT.resolve(), Path(input_artifact).resolve()
    if project == output or project in output.parents:
        artifacts = project / "artifacts"
        if artifacts not in output.parents:
            raise ValueError("outputs inside the project must be below artifacts/")
    elif not smoke:
        raise ValueError("real V4.4-RU output must be inside project artifacts/")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the read-only input artifact")
    # Every existing project artifact is protected by the fresh-root rule.
    if output.exists():
        raise FileExistsError(f"V4.4-RU requires a new output directory: {output}")
    return output


def claim_output(path: Path, input_artifact: Path, *, smoke=False) -> Path:
    output = validate_output(path, input_artifact, smoke=smoke)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


__all__ = ["claim_output", "validate_input", "validate_output", "file_sha256",
           "load_json", "read_jsonl", "write_json", "write_jsonl"]
