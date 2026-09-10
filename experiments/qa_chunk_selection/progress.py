"""Stdout + JSONL progress so a background run is inspectable."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


class ProgressLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._stage = "init"
        self._started = time.time()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str, message: str = "") -> None:
        self._stage = name
        suffix = f" | {message}" if message else ""
        self.log(f"STAGE {name}{suffix}")

    def log(self, message: str, **fields: Any) -> None:
        elapsed = time.time() - self._started
        line = f"[{_now()}] [{elapsed:8.1f}s] [{self._stage}] {message}"
        print(line, flush=True)
        payload: dict[str, Any] = {
            "ts": _now(),
            "elapsed_s": round(elapsed, 3),
            "stage": self._stage,
            "message": message,
        }
        payload.update(fields)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def count(
        self,
        current: int,
        total: int,
        *,
        item: str = "item",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        total = max(int(total), 1)
        pct = 100.0 * float(current) / float(total)
        every = max(1, total // 20)
        if current == 1 or current == total or current % every == 0:
            suffix = ""
            if extra:
                suffix = " | " + ", ".join(f"{key}={value}" for key, value in extra.items())
            self.log(f"{item} {current}/{total} ({pct:.1f}%){suffix}", current=current, total=total)

    def warn(self, message: str, **fields: Any) -> None:
        self.log(f"WARN {message}", **fields)
        print(f"[{_now()}] WARN {message}", file=sys.stderr, flush=True)
