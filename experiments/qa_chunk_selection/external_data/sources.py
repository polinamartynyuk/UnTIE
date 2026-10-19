from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable

from .config import SCIER_REVISION, SCIMDIX_REVISION


SCIMDIX_FILES = {
    "aspects_train_ru.parquet": "aspects/train_ru-00000-of-00001.parquet",
    "aspects_test_ru.parquet": "aspects/test_ru-00000-of-00001.parquet",
    "ner_re_train_ru.parquet": "ner_re/train_ru-00000-of-00001.parquet",
    "ner_re_test_ru.parquet": "ner_re/test_ru-00000-of-00001.parquet",
    "README.md": "README.md",
}
SCIER_FILES = {
    **{f"{split}.jsonl": f"SciER/PLM/{split}.jsonl" for split in ("train", "dev", "test", "test_ood")},
    "README.md": "README.md",
    "LICENSE": "LICENSE",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "UnTIE-external-data/1.0"})
    last: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise ValueError(f"empty response from {url}")
            destination.write_bytes(payload)
            return
        except Exception as error:  # network retry is intentionally narrow and bounded
            last = error
            if attempt < 4:
                time.sleep(attempt * 2)
    raise RuntimeError(f"could not download {url}: {last}")


def _locate(source: Path, local_name: str, remote_name: str, dataset: str) -> Path:
    candidates = [source / local_name, source / remote_name, source / "files" / local_name]
    if dataset == "scier":
        candidates.extend((source / "SciER" / "PLM" / local_name, source / "PLM" / local_name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {dataset} source file {local_name} below {source}")


def materialize_source(
    dataset: str,
    destination: Path,
    *,
    local_source: Path | None,
    progress: Callable[[str], None],
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=False)
    if dataset == "scimdix":
        files, revision = SCIMDIX_FILES, SCIMDIX_REVISION
        base = f"https://huggingface.co/datasets/tvbat/SciMDIX/resolve/{revision}/"
        repository = "https://huggingface.co/datasets/tvbat/SciMDIX"
    elif dataset == "scier":
        files, revision = SCIER_FILES, SCIER_REVISION
        base = f"https://raw.githubusercontent.com/edzq/SciER/{revision}/"
        repository = "https://github.com/edzq/SciER"
    else:
        raise ValueError(dataset)
    manifest_files = []
    for index, (local_name, remote_name) in enumerate(files.items(), start=1):
        target = destination / local_name
        if local_source is None:
            url = base + remote_name
            progress(f"downloading {dataset} {local_name} ({index}/{len(files)})")
            _fetch(url, target)
            origin = url
        else:
            source = _locate(Path(local_source), local_name, remote_name, dataset)
            progress(f"copying {dataset} {local_name} ({index}/{len(files)})")
            shutil.copyfile(source, target)
            origin = str(source.resolve())
        manifest_files.append({
            "name": local_name,
            "origin": origin,
            "bytes": target.stat().st_size,
            "sha256": sha256_path(target),
        })
    return {
        "dataset": dataset,
        "repository": repository,
        "resolved_revision": revision,
        "mode": "download" if local_source is None else "local_copy",
        "files": manifest_files,
    }
