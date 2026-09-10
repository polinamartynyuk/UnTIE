"""Feature cache. Keys include text and model hashes so stale IDs cannot hit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .io_util import load_json, write_json


def cache_key(**parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


class FeatureCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self.path_for(namespace, key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        payload = load_json(path)
        return payload if isinstance(payload, dict) else None

    def set(self, namespace: str, key: str, payload: Mapping[str, Any]) -> None:
        write_json(self.path_for(namespace, key), dict(payload))

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": float(self.hits / total) if total else 0.0,
        }
