"""Read V4.3 embeddings without loading MiniLM; lazy model only for new support texts."""
import hashlib
import time
from pathlib import Path

import numpy as np
from experiments.qa_chunk_selection.v4_3.encoder import HashEncoder, MiniLMEncoder, text_cache_key
from experiments.qa_chunk_selection.v4_3.text import l2_normalize, semantic_normalize
from experiments.qa_chunk_selection.v4.keybert_features import resolve_encoder_device
from .contract import SEMANTIC_MODEL, NORMALIZATION_VERSION
from .io_guard import load_json, file_sha256


class ReplayEncoder:
    def __init__(self, previous: Path, output: Path, *, smoke=False, device="auto"):
        self.previous, self.output = previous, output
        self.smoke, self.device = smoke, device
        self.backend = HashEncoder() if smoke else None
        self.memory = {}
        self.hits = self.encoded = 0
        self.encode_seconds = self.read_seconds = 0.0
        self.source_hashes = {}
        if smoke:
            self.spec = {"model_name": "hash-smoke", "model_hash": "smoke", "smoke": True,
                         "normalization_version": NORMALIZATION_VERSION, "frozen": True}
        else:
            self.spec = load_json(previous / "encoder_spec.json")
            if (self.spec["model_name"] != SEMANTIC_MODEL or self.spec.get("smoke")
                    or self.spec["normalization_version"] != NORMALIZATION_VERSION or not self.spec["frozen"]):
                raise ValueError("V4.3 encoder contract mismatch")

    def embed_one(self, raw: str, *, store: str):
        text = semantic_normalize(raw)
        key = text_cache_key(text, model_hash=self.spec["model_hash"], normalization_version=NORMALIZATION_VERSION)
        memory_key = (store, key)
        if memory_key in self.memory:
            self.hits += 1
            return self.memory[memory_key]
        path = self.previous / f"{store}_embeddings" / "vec" / f"{key}.npy"
        started = time.perf_counter()
        if not self.smoke and path.is_file():
            vector = l2_normalize(np.load(path, allow_pickle=False))
            self.source_hashes[str(path)] = file_sha256(path)
            self.hits += 1
            self.read_seconds += time.perf_counter() - started
        else:
            if store == "answer" and not self.smoke:
                raise FileNotFoundError(f"Required V4.3 answer embedding missing: {path}; no silent encoder replacement")
            if self.backend is None:
                self.backend = MiniLMEncoder(SEMANTIC_MODEL, device=resolve_encoder_device(self.device))
                actual = hashlib.sha256(f"{self.backend.model_name}|{self.backend.revision}".encode()).hexdigest()[:16]
                if actual != self.spec["model_hash"]:
                    raise ValueError("Loaded support encoder differs from V4.3; refusing mixed embeddings")
            vector = l2_normalize(self.backend.encode([text])[0])
            self.encode_seconds += time.perf_counter() - started
            self.encoded += 1
            dest = self.output / f"{store}_embeddings" / "vec" / f"{key}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, np.asarray(vector, dtype=np.float32))
        if not np.all(np.isfinite(vector)):
            raise ValueError("Nonfinite embedding")
        self.memory[memory_key] = vector
        return vector

    def stats(self):
        return {"cache_hits": self.hits, "new_texts_encoded": self.encoded,
                "encode_seconds": self.encode_seconds, "cache_read_seconds": self.read_seconds,
                "model_loaded": self.backend is not None, "spec": self.spec}
