"""Cached sentence-transformer encoder for topic representations and KeyBERT."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_TOPIC_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedSentenceEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.revision = str(
            getattr(self.model, "model_card_data", None)
            and getattr(self.model.model_card_data, "base_model_revision", "")
            or getattr(self.model, "revision", "")
            or "unknown"
        )
        probe = np.asarray(self.model.encode(["probe"], normalize_embeddings=normalize))
        self.embedding_dimension = int(probe.shape[-1])
        self.cache_dir = cache_dir
        self._memory: dict[str, np.ndarray] = {}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = cache_dir / "embedding_index.json"
            self._vectors_path = cache_dir / "embeddings.npy"
            self._load_disk()

    def spec(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalize,
            "device": self.device,
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        missing: list[tuple[int, str, str]] = []
        output = np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            key = _hash_text(text)
            cached = self._memory.get(key)
            if cached is None:
                missing.append((index, text, key))
            else:
                output[index] = cached
        if missing:
            vectors = np.asarray(
                self.model.encode(
                    [item[1] for item in missing],
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            for row, (_, _, key) in zip(vectors, missing):
                self._memory[key] = row
            for local, (index, _, _) in enumerate(missing):
                output[index] = vectors[local]
            self._flush_disk()
        return output

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _load_disk(self) -> None:
        if not self._index_path.is_file() or not self._vectors_path.is_file():
            return
        index = json.loads(self._index_path.read_text(encoding="utf-8"))
        vectors = np.load(self._vectors_path)
        for key, row in zip(index, vectors):
            self._memory[str(key)] = np.asarray(row, dtype=np.float32)

    def _flush_disk(self) -> None:
        if self.cache_dir is None:
            return
        keys = sorted(self._memory)
        matrix = np.stack([self._memory[key] for key in keys]).astype(np.float32)
        np.save(self._vectors_path, matrix)
        self._index_path.write_text(json.dumps(keys), encoding="utf-8")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms
