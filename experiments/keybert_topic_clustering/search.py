"""Staged clustering search. No full Cartesian grid. Selection uses train+dev only."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .clustering import ClusterModel, fit_clusters, stability_scores
from .documents import PaperDocument
from .encoder import CachedSentenceEncoder
from .io_util import write_csv, write_json
from .keyphrases import KeyphraseConfig, extract_document_keyphrases
from .labels import nearest_papers, representative_terms
from .phrase_analysis import cluster_keyphrases, document_phrase_overlap
from .representations import (
    RepresentationSpec,
    TfidfSvdFit,
    build_matrix,
    fit_e0,
)
from .routing import route_matrix


@dataclass
class ClusteringCandidate:
    spec: RepresentationSpec
    algorithm: str
    k: int
    seed: int
    model: ClusterModel
    mean_dev_top1: float
    train_time_s: float
    uses_keybert: bool

    @property
    def config_id(self) -> str:
        return f"{self.spec.name}|{self.algorithm}|k{self.k}|s{self.seed}"


def _rank_score(candidates: list[ClusteringCandidate]) -> list[tuple[float, ClusteringCandidate]]:
    if not candidates:
        return []
    sil = np.asarray([item.model.silhouette for item in candidates], dtype=float)
    db = np.asarray([item.model.davies_bouldin for item in candidates], dtype=float)
    ch = np.asarray([item.model.calinski_harabasz for item in candidates], dtype=float)
    dev = np.asarray([item.mean_dev_top1 for item in candidates], dtype=float)
    spread = np.asarray(
        [
            (item.model.size_max - item.model.size_min) / max(item.model.size_mean, 1.0)
            for item in candidates
        ],
        dtype=float,
    )

    def ranks(values: np.ndarray, *, higher: bool) -> np.ndarray:
        order = np.argsort(values if higher else -values)
        result = np.empty(len(values), dtype=float)
        result[order] = np.linspace(0.0, 1.0, len(values))
        return result

    score = (
        0.25 * ranks(np.nan_to_num(sil, nan=-1.0), higher=True)
        + 0.20 * ranks(np.nan_to_num(db, nan=99.0), higher=False)
        + 0.15 * ranks(np.nan_to_num(ch, nan=0.0), higher=True)
        + 0.30 * ranks(np.nan_to_num(dev, nan=0.0), higher=True)
        + 0.10 * ranks(spread, higher=False)
    )
    return sorted(zip(score.tolist(), candidates), key=lambda pair: -pair[0])


class SearchContext:
    def __init__(
        self,
        train: Sequence[PaperDocument],
        dev: Sequence[PaperDocument],
        encoder: CachedSentenceEncoder,
        *,
        keyphrase_dir,
        default_phrases: KeyphraseConfig,
    ) -> None:
        self.train = list(train)
        self.dev = list(dev)
        self.encoder = encoder
        self.keyphrase_dir = keyphrase_dir
        self.default_phrases = default_phrases
        self.e0 = fit_e0([paper.representation("R2") for paper in self.train])
        self._phrase_cache: dict[str, dict] = {}

    def phrases(self, config: KeyphraseConfig) -> dict:
        if config.key not in self._phrase_cache:
            self._phrase_cache[config.key] = extract_document_keyphrases(
                [*self.train, *self.dev],
                self.encoder,
                config,
                cache_dir=self.keyphrase_dir,
                extract_top_n=max(config.top_n, 20),
            )
        return self._phrase_cache[config.key]

    def evaluate(self, spec: RepresentationSpec, *, algorithm: str, k: int, seed: int) -> ClusteringCandidate:
        kp = None
        if spec.uses_keybert:
            ngram_low, ngram_high, top_n, mmr, diversity = _parse_kp(spec.keyphrase_config)
            kp_cfg = KeyphraseConfig(ngram_low, ngram_high, top_n, mmr, diversity)
            kp = self.phrases(kp_cfg)
        started = time.perf_counter()
        train_vectors = build_matrix(
            self.train, spec, encoder=self.encoder, e0=self.e0, phrases=kp
        )
        model = fit_clusters(train_vectors, algorithm=algorithm, k=k, seed=seed)
        elapsed = time.perf_counter() - started
        dev_vectors = build_matrix(
            self.dev, spec, encoder=self.encoder, e0=self.e0, phrases=kp
        )
        routed = route_matrix(
            dev_vectors,
            model.centroids,
            [paper.doc_id for paper in self.dev],
            [paper.split for paper in self.dev],
        )
        mean_dev = float(np.mean([item.top1_similarity for item in routed])) if routed else 0.0
        return ClusteringCandidate(
            spec=spec,
            algorithm=algorithm,
            k=k,
            seed=seed,
            model=model,
            mean_dev_top1=mean_dev,
            train_time_s=elapsed,
            uses_keybert=spec.uses_keybert,
        )


def _parse_kp(token: str) -> tuple[int, int, int, bool, float]:
    # ng1-3_top10_nommr  or ng1-3_top10_mmr0.50
    ngram_low, ngram_high = 1, 3
    top_n = 10
    use_mmr = False
    diversity = 0.5
    for part in token.split("_"):
        if part.startswith("ng"):
            low, high = part[2:].split("-")
            ngram_low, ngram_high = int(low), int(high)
        elif part.startswith("top"):
            top_n = int(part[3:])
        elif part == "nommr":
            use_mmr = False
        elif part.startswith("mmr"):
            use_mmr = True
            diversity = float(part[3:])
    return ngram_low, ngram_high, top_n, use_mmr, diversity


def kp_token(config: KeyphraseConfig) -> str:
    return config.key


def run_staged_search(
    ctx: SearchContext,
    *,
    k_grid: tuple[int, ...] = (4, 8, 16),
    stage_k: int = 8,
    seed: int = 42,
    quick: bool = False,
) -> dict[str, object]:
    default_kp = ctx.default_phrases
    default_token = kp_token(default_kp)
    k_use = k_grid if not quick else (min(k_grid),)
    algorithms_full = ("BisectingKMeans", "KMeans", "MiniBatchKMeans")
    algorithms_stage_a = ("BisectingKMeans",)

    stage_a_specs = [
        RepresentationSpec("E0", "R2"),
        RepresentationSpec("E1", "R1"),
        RepresentationSpec("E1", "R2"),
        RepresentationSpec("E1", "R3"),
        RepresentationSpec("E2", "R2", keyphrase_config=default_token),
        RepresentationSpec("E3", "R2", alpha=0.25, keyphrase_config=default_token),
        RepresentationSpec("E3", "R2", alpha=0.50, keyphrase_config=default_token),
        RepresentationSpec("E3", "R2", alpha=0.75, keyphrase_config=default_token),
    ]
    collected: list[ClusteringCandidate] = []
    for spec in stage_a_specs:
        collected.append(
            ctx.evaluate(spec, algorithm="BisectingKMeans", k=stage_k if not quick else k_use[0], seed=seed)
        )

    ranked_a = _rank_score(collected)
    best_plain_a = next(item for _, item in ranked_a if not item.uses_keybert)
    best_key_a = next(item for _, item in ranked_a if item.uses_keybert)

    if not quick:
        stage_b: list[ClusteringCandidate] = []
        for ngram in ((1, 2), (1, 3)):
            for top_n in (5, 10, 20):
                for use_mmr, diversity in ((False, 0.5), (True, 0.5)):
                    cfg = KeyphraseConfig(ngram[0], ngram[1], top_n, use_mmr, diversity)
                    for family, alpha in (("E2", None), ("E3", best_key_a.spec.alpha or 0.5)):
                        spec = RepresentationSpec(
                            family,
                            "R2",
                            alpha=alpha,
                            keyphrase_config=kp_token(cfg),
                        )
                        stage_b.append(
                            ctx.evaluate(spec, algorithm="BisectingKMeans", k=best_key_a.k, seed=seed)
                        )
        collected.extend(stage_b)
        if any(item.spec.uses_keybert and "mmr0.50" in item.spec.keyphrase_config for item in stage_b):
            winner_mmr = max(
                (item for item in stage_b if item.uses_keybert),
                key=lambda item: item.mean_dev_top1,
            )
            if winner_mmr.spec.keyphrase_config.endswith("mmr0.50"):
                for diversity in (0.3, 0.7):
                    cfg = KeyphraseConfig(1, 3, 10, True, diversity)
                    spec = RepresentationSpec(
                        winner_mmr.spec.family,
                        "R2",
                        alpha=winner_mmr.spec.alpha,
                        keyphrase_config=kp_token(cfg),
                    )
                    extra = ctx.evaluate(spec, algorithm="BisectingKMeans", k=winner_mmr.k, seed=seed)
                    collected.append(extra)

        ranked = _rank_score(collected)
        plain_spec = next(item.spec for _, item in ranked if not item.uses_keybert)
        key_spec = next(item.spec for _, item in ranked if item.uses_keybert)
        for spec in (plain_spec, key_spec):
            for k in k_use:
                if k == stage_k:
                    continue
                collected.append(
                    ctx.evaluate(spec, algorithm="BisectingKMeans", k=k, seed=seed)
                )
        ranked = _rank_score(collected)
        plain_best = next(item for _, item in ranked if not item.uses_keybert)
        key_best = next(item for _, item in ranked if item.uses_keybert)
        for spec, base in ((plain_best.spec, plain_best), (key_best.spec, key_best)):
            for algorithm in algorithms_full:
                if algorithm == base.algorithm:
                    continue
                collected.append(
                    ctx.evaluate(spec, algorithm=algorithm, k=base.k, seed=seed)
                )
    ranked_final = _rank_score(collected)
    c_plain = next(item for _, item in ranked_final if not item.uses_keybert)
    c_keybert = next(item for _, item in ranked_final if item.uses_keybert)
    return {
        "candidates": collected,
        "ranked": ranked_final,
        "c_plain": c_plain,
        "c_keybert": c_keybert,
        "e0": ctx.e0,
    }


def freeze_and_describe(
    ctx: SearchContext,
    chosen: ClusteringCandidate,
    *,
    all_papers: Sequence[PaperDocument],
    output_dir,
    stability_seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, object]:
    kp = None
    kp_cfg = ctx.default_phrases
    if chosen.spec.uses_keybert:
        parsed = _parse_kp(chosen.spec.keyphrase_config)
        kp_cfg = KeyphraseConfig(*parsed)
        kp = extract_document_keyphrases(
            list(all_papers),
            ctx.encoder,
            kp_cfg,
            cache_dir=ctx.keyphrase_dir,
            extract_top_n=max(kp_cfg.top_n, 20),
        )
    else:
        kp = ctx.phrases(ctx.default_phrases)
    train_vectors = build_matrix(
        ctx.train, chosen.spec, encoder=ctx.encoder, e0=ctx.e0, phrases=kp
    )
    holdout = [paper for paper in all_papers if paper.split != "train"]
    hold_vectors = build_matrix(
        holdout, chosen.spec, encoder=ctx.encoder, e0=ctx.e0, phrases=kp
    )
    routed = route_matrix(
        hold_vectors,
        chosen.model.centroids,
        [paper.doc_id for paper in holdout],
        [paper.split for paper in holdout],
    )
    assignments = [
        {"doc_id": paper.doc_id, "split": "train", "cluster_id": int(label)}
        for paper, label in zip(ctx.train, chosen.model.labels)
    ]
    for item in routed:
        assignments.append(
            {
                "doc_id": item.doc_id,
                "split": item.split,
                "cluster_id": item.top1,
                "similarity": item.top1_similarity,
                "top_ids": list(item.top_ids),
                "weights": list(item.weights),
            }
        )
    terms = representative_terms(
        ctx.train,
        train_vectors,
        chosen.model.centroids,
        chosen.model.labels,
        encoder=ctx.encoder,
        phrases=kp,
    )
    papers_near = nearest_papers(
        ctx.train, train_vectors, chosen.model.centroids, chosen.model.labels
    )
    stability = stability_scores(
        train_vectors,
        algorithm=chosen.algorithm,
        k=chosen.k,
        seeds=stability_seeds,
    )
    return {
        "config_id": chosen.config_id,
        "spec": {
            "family": chosen.spec.family,
            "text_field": chosen.spec.text_field,
            "alpha": chosen.spec.alpha,
            "keyphrase_config": chosen.spec.keyphrase_config,
            "name": chosen.spec.name,
        },
        "algorithm": chosen.algorithm,
        "k": chosen.k,
        "seed": chosen.seed,
        "centroids": chosen.model.centroids,
        "train_labels": {paper.doc_id: int(label) for paper, label in zip(ctx.train, chosen.model.labels)},
        "metrics": {
            "silhouette": chosen.model.silhouette,
            "davies_bouldin": chosen.model.davies_bouldin,
            "calinski_harabasz": chosen.model.calinski_harabasz,
            "cluster_size_min": chosen.model.size_min,
            "cluster_size_max": chosen.model.size_max,
            "cluster_size_mean": chosen.model.size_mean,
            "cluster_size_median": chosen.model.size_median,
            "mean_dev_top1": chosen.mean_dev_top1,
            **stability,
        },
        "assignments": assignments,
        "routing": routed,
        "representative_terms": terms,
        "nearest_papers": papers_near,
        "phrases": kp,
        "keyphrase_config": kp_cfg,
        "train_vectors": train_vectors,
        "holdout_vectors": hold_vectors,
        "holdout_papers": holdout,
    }
