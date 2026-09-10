"""Per-chunk KeyBERT + LOSCO document compatibility. Backend fixed a priori."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v2.cache import FeatureCache, cache_key, text_hash

from .clusters import overlapping_ids
from .contract import KEYBERT_BACKEND


def resolve_encoder_device(requested: str) -> str:
    from untie.models import resolve_device

    return resolve_device(requested or "cpu")

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def lexical_keyphrases(text: str, *, top_n: int, ngram_high: int) -> list[dict[str, Any]]:
    tokens = [item.lower() for item in _WORD.findall(text or "")]
    counts: Counter[str] = Counter()
    for width in range(1, max(int(ngram_high), 1) + 1):
        for start in range(0, max(0, len(tokens) - width + 1)):
            phrase = " ".join(tokens[start : start + width])
            counts[phrase] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(int(top_n), 0)]
    return [{"phrase": phrase, "score": float(score)} for phrase, score in ranked]


class KeyphraseExtractor:
    def __init__(
        self,
        *,
        backend: str,
        top_n: int,
        ngram_low: int,
        ngram_high: int,
        use_mmr: bool,
        smoke: bool,
        cache: FeatureCache | None = None,
        device: str = "cpu",
        progress=None,
    ) -> None:
        self.backend = "lexical-smoke" if smoke else backend
        self.top_n = int(top_n)
        self.ngram_low = int(ngram_low)
        self.ngram_high = int(ngram_high)
        self.use_mmr = bool(use_mmr)
        self.smoke = bool(smoke)
        self.cache = cache
        self.device = resolve_encoder_device("cpu" if smoke else device)
        self.progress = progress
        self._kw = None
        self._encoder = None

    def spec(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "top_n": self.top_n,
            "ngram": [self.ngram_low, self.ngram_high],
            "use_mmr": self.use_mmr,
            "smoke": self.smoke,
            "device": self.device,
        }

    def _ensure_backend(self) -> None:
        if self.smoke or self._kw is not None:
            return
        from experiments.keybert_topic_clustering.encoder import CachedSentenceEncoder
        from keybert import KeyBERT

        if self.progress:
            self.progress.log(f"loading KeyBERT encoder on {self.device}")
        self._encoder = CachedSentenceEncoder(self.backend or KEYBERT_BACKEND, device=self.device)
        self._kw = KeyBERT(model=self._encoder.model)

    def extract(self, text: str) -> list[dict[str, Any]]:
        payload = {
            "backend": self.backend,
            "top_n": self.top_n,
            "ngram_low": self.ngram_low,
            "ngram_high": self.ngram_high,
            "use_mmr": self.use_mmr,
            "text_hash": text_hash(text),
        }
        key = cache_key(**payload)
        if self.cache is not None:
            hit = self.cache.get("keybert", key)
            if hit is not None:
                return list(hit.get("phrases") or [])
        if self.smoke:
            phrases = lexical_keyphrases(text, top_n=self.top_n, ngram_high=self.ngram_high)
        else:
            self._ensure_backend()
            raw = self._kw.extract_keywords(
                text or " ",
                keyphrase_ngram_range=(self.ngram_low, self.ngram_high),
                stop_words="english",
                top_n=self.top_n,
                use_mmr=self.use_mmr,
                diversity=0.5,
            )
            phrases = [{"phrase": str(phrase), "score": float(score)} for phrase, score in raw or []]
        if self.cache is not None:
            self.cache.set("keybert", key, {"phrases": phrases})
        return phrases

    def embed_phrases(self, phrases: Sequence[str]) -> list[list[float]]:
        if not phrases:
            return []
        if self.smoke:
            return [[float(len(item)), float(item.count(" "))] for item in phrases]
        self._ensure_backend()
        vectors = self._encoder.encode(list(phrases))
        return [row.astype(float).tolist() for row in vectors]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    na = math.sqrt(sum(float(a) * float(a) for a in left))
    nb = math.sqrt(sum(float(b) * float(b) for b in right))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(dot / (na * nb))


def document_idf(train_records: Sequence[InstanceRecord], phrases_by_chunk: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    docs = {item.document_id for item in train_records}
    qid_to_doc = {item.question_id: item.document_id for item in train_records}
    seen_doc: dict[str, set[str]] = defaultdict(set)
    for (question_id, chunk_id), phrases in phrases_by_chunk.items():
        del chunk_id
        doc_id = qid_to_doc.get(question_id) or str(question_id).split("::", 1)[0]
        if doc_id not in docs:
            continue
        for row in phrases:
            seen_doc[str(row["phrase"])].add(doc_id)
    n_docs = max(len(docs), 1)
    for phrase, holders in seen_doc.items():
        df[phrase] = len(holders)
    return {phrase: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0 for phrase, count in df.items()}


def _phrase_map(phrases: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in phrases:
        phrase = str(row.get("phrase") or "").strip()
        if phrase:
            out[phrase] = max(out.get(phrase, 0.0), float(row.get("score") or 0.0))
    return out


def compatibility_features(
    item: InstanceRecord,
    chunk_id: int,
    phrases_by_chunk: Mapping[int, Sequence[Mapping[str, Any]]],
    embeddings_by_phrase: Mapping[str, Sequence[float]],
    idf: Mapping[str, float],
    *,
    min_overlap: float,
    mode: str,
    aspect_embedding: Sequence[float] | None = None,
) -> dict[str, float]:
    candidate = _phrase_map(phrases_by_chunk.get(int(chunk_id)) or [])
    if mode == "global":
        excluded: set[int] = set()
    elif mode == "loco":
        excluded = {int(chunk_id)}
    else:
        excluded = overlapping_ids(item, int(chunk_id), min_overlap=min_overlap)
    rest: dict[str, dict[str, float]] = {}
    support_clusters: dict[str, set[int]] = defaultdict(set)
    for other_id, phrases in phrases_by_chunk.items():
        if int(other_id) in excluded:
            continue
        cluster_seed = int(other_id)
        for row in phrases:
            phrase = str(row.get("phrase") or "")
            if not phrase:
                continue
            bucket = rest.setdefault(phrase, {"score": 0.0, "max": 0.0, "n": 0.0})
            score = float(row.get("score") or 0.0)
            bucket["score"] += score
            bucket["max"] = max(bucket["max"], score)
            bucket["n"] += 1.0
            support_clusters[phrase].add(cluster_seed)
    if not candidate:
        return {
            "s_overlap": 0.0,
            "s_embedding": 0.0,
            "s_support": 0.0,
            "s_idf": 0.0,
            "s_aspect": 0.0,
            "n_excluded": float(len(excluded)),
        }
    weights = {phrase: score * float(idf.get(phrase, 1.0)) for phrase, score in candidate.items()}
    denom = sum(weights.values()) or 1.0
    overlap = sum(weights[phrase] for phrase in candidate if phrase in rest) / denom
    sims = []
    for phrase in candidate:
        left = embeddings_by_phrase.get(phrase)
        if left is None or not rest:
            continue
        best = 0.0
        for other in rest:
            right = embeddings_by_phrase.get(other)
            if right is None:
                continue
            best = max(best, cosine(left, right))
        sims.append(best)
    embed = float(sum(sims) / len(sims)) if sims else 0.0
    support = sum(weights[phrase] * math.log(1.0 + len(support_clusters.get(phrase, set()))) for phrase in candidate)
    idf_support = sum(weights[phrase] for phrase in candidate if phrase in rest)
    aspect = 0.0
    if aspect_embedding:
        aspect_sims = []
        for phrase, score in candidate.items():
            vec = embeddings_by_phrase.get(phrase)
            if vec is None:
                continue
            aspect_sims.append(score * cosine(vec, aspect_embedding))
        aspect = float(sum(aspect_sims) / len(aspect_sims)) if aspect_sims else 0.0
    return {
        "s_overlap": float(overlap),
        "s_embedding": embed,
        "s_support": float(support),
        "s_idf": float(idf_support),
        "s_aspect": float(aspect),
        "n_excluded": float(len(excluded)),
    }


def document_profile_phrases(
    phrases_by_chunk: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    top_n: int = 15,
) -> list[str]:
    scores: Counter[str] = Counter()
    for phrases in phrases_by_chunk.values():
        for row in phrases:
            scores[str(row["phrase"])] += float(row.get("score") or 0.0)
    return [phrase for phrase, _score in scores.most_common(int(top_n))]
