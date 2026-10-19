"""Exclusive fresh output roots; never instantiate a writable cache on old trees."""
import json
import re
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v3.data import instance_from_dict
from experiments.qa_chunk_selection.v3.contract import UNLOCKED_V2_SPLITS
from experiments.qa_chunk_selection.v4.io_guard import file_sha256, load_json, read_jsonl, write_json, write_jsonl
from .contract import PROTECTED_WRITE_ROOTS


def validate_output(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("V4.4 output cannot use symlinks")
    resolved, root = path.resolve(), project_root.resolve()
    # Only the artifacts subtree (or external temporary folders for tests) may be used.
    if resolved == root or root in resolved.parents:
        artifacts = root / "artifacts"
        if artifacts not in resolved.parents:
            raise ValueError("V4.4 outputs inside the project must be below artifacts/")
    for relative in PROTECTED_WRITE_ROOTS:
        protected = (root / relative).resolve()
        if resolved == protected or protected in resolved.parents or resolved in protected.parents:
            raise ValueError(f"V4.4 refuses protected output: {relative}")
    if resolved.exists():
        raise FileExistsError(f"V4.4 requires a NEW output directory, even after a failed run: {resolved}")
    return resolved


def claim_output(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    resolved = validate_output(path, project_root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(exist_ok=False)  # concurrent launch: exactly one process succeeds
    return resolved


_QID = re.compile(r'"question_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def load_unlocked_snapshot(path: Path, allowed_qids: set[str]):
    """Scan shared JSONL, decoding records ONLY for the unlocked B0 whitelist.

    The historical file physically mixes splits. Skipped lines are neither
    deserialized into records nor sent to inference/metrics. No test file is opened.
    """
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = _QID.search(line)
            if not match or json.loads(match.group(1)) not in allowed_qids:
                continue
            payload = json.loads(line)
            if payload["split"] not in UNLOCKED_V2_SPLITS:
                raise ValueError("Locked/unknown split in unlocked whitelist")
            item = instance_from_dict(payload)
            if item.question_id in result:
                raise ValueError("Duplicate snapshot question")
            result[item.question_id] = item
    if set(result) != allowed_qids:
        raise ValueError("Incomplete unlocked snapshot")
    return tuple(result[qid] for qid in sorted(result))
