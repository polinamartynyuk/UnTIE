"""Read-through cache: write only to V4.1, optionally read V4 QA cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.qa_chunk_selection.v2.cache import FeatureCache
from experiments.qa_chunk_selection.v2.io_util import load_json


class ReadThroughCache:
    def __init__(self, primary: FeatureCache, fallback: FeatureCache | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_hits = 0

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        hit = self.primary.get(namespace, key)
        if hit is not None:
            return hit
        if self.fallback is None:
            return None
        path = self.fallback.path_for(namespace, key)
        if not path.exists():
            return None
        payload = load_json(path)
        if not isinstance(payload, dict):
            return None
        self.primary.set(namespace, key, payload)
        self.fallback_hits += 1
        self.primary.hits += 1
        self.primary.misses = max(self.primary.misses - 1, 0)
        return payload

    def set(self, namespace: str, key: str, payload: Mapping[str, Any]) -> None:
        self.primary.set(namespace, key, payload)

    def stats(self) -> dict[str, float]:
        payload = self.primary.stats()
        payload["fallback_hits"] = float(self.fallback_hits)
        return payload

    def path_for(self, namespace: str, key: str) -> Path:
        return self.primary.path_for(namespace, key)
