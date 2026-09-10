"""Frozen semantic encoder + disk cache. No SciREX fine-tuning."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device

from .contract import NORMALIZATION_VERSION, SEMANTIC_MODEL, Z_EPS
from .text import l2_normalize, semantic_normalize


def text_cache_key(text: str, *, model_hash: str, normalization_version: str) -> str:
    payload = f"{model_hash}\n{normalization_version}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HashEncoder:
    """Deterministic smoke encoder. Identical text → identical unit vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = int(dim)
        self.model_name = "hash-smoke"
        self.revision = "smoke"
        self.device = "cpu"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            out[index] = l2_normalize(rng.normal(size=self.dim), eps=Z_EPS)
        return out


class MiniLMEncoder:
    def __init__(self, model_name: str, *, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.revision = str(
            getattr(self.model, "model_card_data", None)
            and getattr(self.model.model_card_data, "base_model_revision", "")
            or getattr(self.model, "revision", "")
            or "unknown"
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            vectors = self.model.encode(
                list(texts),
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingStore:
    def __init__(self, root: Path, *, namespace: str, model_hash: str, normalization_version: str) -> None:
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.model_hash = model_hash
        self.normalization_version = normalization_version
        self.hits = 0
        self.misses = 0
        self._memory: dict[str, np.ndarray] = {}

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        key = text_cache_key(text, model_hash=self.model_hash, normalization_version=self.normalization_version)
        cached = self._memory.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        path = self._path(key)
        if path.exists():
            vector = np.load(path)
            self._memory[key] = vector
            self.hits += 1
            return vector
        self.misses += 1
        return None

    def set(self, text: str, vector: np.ndarray) -> np.ndarray:
        key = text_cache_key(text, model_hash=self.model_hash, normalization_version=self.normalization_version)
        arr = np.asarray(vector, dtype=np.float32)
        self._memory[key] = arr
        np.save(self._path(key), arr)
        return arr

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": float(self.hits / total) if total else 0.0,
            "n_memory": float(len(self._memory)),
        }


class SemanticEncoder:
    def __init__(
        self,
        *,
        cache_root: Path,
        device: str = "cpu",
        smoke: bool = False,
        model_name: str = SEMANTIC_MODEL,
        normalization_version: str = NORMALIZATION_VERSION,
        progress=None,
    ) -> None:
        self.smoke = bool(smoke)
        self.normalization_version = normalization_version
        resolved = "cpu" if smoke else resolve_encoder_device(device)
        self.device = resolved
        self.backend: HashEncoder | MiniLMEncoder = HashEncoder() if smoke else MiniLMEncoder(model_name, device=resolved)
        self.model_name = self.backend.model_name
        self.model_hash = hashlib.sha256(f"{self.backend.model_name}|{getattr(self.backend, 'revision', '')}".encode("utf-8")).hexdigest()[:16]
        self.progress = progress
        self.encode_seconds = 0.0
        self.n_encoded = 0
        self.stores = {
            "answer": EmbeddingStore(cache_root / "answer_embeddings", namespace="vec", model_hash=self.model_hash, normalization_version=normalization_version),
            "query_answer": EmbeddingStore(cache_root / "query_answer_embeddings", namespace="vec", model_hash=self.model_hash, normalization_version=normalization_version),
            "local_evidence": EmbeddingStore(cache_root / "local_evidence_embeddings", namespace="vec", model_hash=self.model_hash, normalization_version=normalization_version),
            "query": EmbeddingStore(cache_root / "query_embeddings", namespace="vec", model_hash=self.model_hash, normalization_version=normalization_version),
            "answer_local": EmbeddingStore(cache_root / "answer_local_embeddings", namespace="vec", model_hash=self.model_hash, normalization_version=normalization_version),
        }

    def spec(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "normalization_version": self.normalization_version,
            "device": self.device,
            "smoke": self.smoke,
            "frozen": True,
        }

    def embed_many(self, texts: Iterable[str], *, store: str, normalize_text: bool = True) -> dict[str, np.ndarray]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw in texts:
            text = semantic_normalize(raw) if normalize_text else str(raw)
            if text in seen:
                continue
            seen.add(text)
            unique.append(text)
        bucket = self.stores[store]
        missing: list[str] = []
        out: dict[str, np.ndarray] = {}
        for text in unique:
            hit = bucket.get(text)
            if hit is None:
                missing.append(text)
            else:
                out[text] = l2_normalize(hit, eps=Z_EPS)
        if missing:
            started = time.time()
            vectors = self.backend.encode(missing)
            self.encode_seconds += time.time() - started
            self.n_encoded += len(missing)
            for index, text in enumerate(missing):
                unit = l2_normalize(vectors[index], eps=Z_EPS)
                bucket.set(text, unit.astype(np.float32))
                out[text] = unit
            if self.progress:
                self.progress.count(len(missing), max(len(missing), 1), item=f"embed-{store}")
        return out

    def embed_one(self, text: str, *, store: str, normalize_text: bool = True) -> np.ndarray:
        mapped = self.embed_many([text], store=store, normalize_text=normalize_text)
        key = semantic_normalize(text) if normalize_text else str(text)
        return mapped[key]

    def stats(self) -> dict[str, Any]:
        return {
            "encode_seconds": self.encode_seconds,
            "n_encoded": self.n_encoded,
            "per_text_s": self.encode_seconds / max(self.n_encoded, 1),
            "device": self.device,
            "stores": {name: store.stats() for name, store in self.stores.items()},
        }

    def write_contract(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.spec(), "stats": self.stats()}, indent=2), encoding="utf-8")
