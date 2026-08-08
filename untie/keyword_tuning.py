"""Deterministic, model-agnostic keyword subset tuning.

The evaluator passed to :func:`sequential_forward_floating_selection` is the
only component that knows how a keyword subset is scored.  This module deals
solely with data preparation, objective computation, caching, and search.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


_TERM_RE = re.compile(r"[^\w]+", re.UNICODE)
_DEFAULT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }
)


@dataclass(frozen=True)
class KeywordEvidence:
    term: str
    doc_id: str
    attention: float = 0.0
    score_diff: float = 0.0
    document_support: int = 1
    supporting_docs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentEvaluation:
    doc_id: str
    quality: float
    baseline_quality: float = 0.0
    fallback: bool = False

    @property
    def delta(self) -> float:
        return self.quality - self.baseline_quality


@dataclass(frozen=True)
class ObjectiveConfig:
    downside_penalty: float = 0.5
    harm_penalty: float = 1.0
    fallback_penalty: float = 0.25
    size_penalty: float = 0.01
    harm_threshold: float = 0.0
    confidence_level: float = 0.95
    bootstrap_samples: int = 500
    bootstrap_seed: int = 0
    confidence_weight: float = 0.0


@dataclass(frozen=True)
class EvaluationResult:
    subset: tuple[str, ...]
    objective: float
    mean_gain: float
    confidence_lower_bound: float
    downside: float
    harm_rate: float
    fallback_rate: float
    size_penalty: float
    document_count: int
    deltas: tuple[float, ...] = ()


@dataclass(frozen=True)
class SearchConfig:
    max_keywords: int = 10
    evaluation_budget: int = 200
    patience: int = 4
    min_improvement: float = 1e-9
    beam_width: int = 1
    panel_sizes: tuple[int, ...] = ()
    seed: int = 0


@dataclass(frozen=True)
class SearchStep:
    iteration: int
    action: str
    keyword: str | None
    subset: tuple[str, ...]
    objective: float
    panel_size: int
    evaluations_used: int


@dataclass(frozen=True)
class SearchResult:
    keywords: tuple[str, ...]
    evaluation: EvaluationResult
    trace: tuple[SearchStep, ...]
    stop_reason: str
    evaluations_used: int
    cache_hits: int = 0


def normalize_term(term: str) -> str:
    """Normalize whitespace, punctuation, and case without language tooling."""
    return " ".join(part for part in _TERM_RE.sub(" ", term.casefold()).split() if part)


def deterministic_document_split(
    doc_ids: Iterable[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> dict[str, tuple[str, ...]]:
    """Split unique document IDs reproducibly, independently of input order."""
    if train_fraction < 0 or dev_fraction < 0 or train_fraction + dev_fraction > 1:
        raise ValueError("split fractions must be non-negative and sum to at most one")
    ids = sorted({str(doc_id) for doc_id in doc_ids})
    ranked = sorted(
        ids,
        key=lambda doc_id: (
            hashlib.sha256(f"{seed}\0{doc_id}".encode("utf-8")).digest(),
            doc_id,
        ),
    )
    train_end = int(len(ranked) * train_fraction)
    dev_end = train_end + int(len(ranked) * dev_fraction)
    return {
        "train": tuple(ranked[:train_end]),
        "dev": tuple(ranked[train_end:dev_end]),
        "test": tuple(ranked[dev_end:]),
    }


split_documents = deterministic_document_split


def aggregate_candidate_pool(
    evidence: Iterable[KeywordEvidence],
    train_doc_ids: Iterable[str],
    *,
    min_document_support: int = 2,
    stopwords: Iterable[str] = (),
) -> tuple[KeywordEvidence, ...]:
    """Aggregate train-only evidence using medians and document support."""
    train = {str(doc_id) for doc_id in train_doc_ids}
    stops = _DEFAULT_STOPWORDS | {normalize_term(word) for word in stopwords}
    grouped: dict[str, dict[str, list[float] | set[str]]] = {}
    for item in evidence:
        if item.doc_id not in train:
            continue
        term = normalize_term(item.term)
        if not term or term in stops or all(part in stops for part in term.split()):
            continue
        bucket = grouped.setdefault(
            term, {"attention": [], "score_diff": [], "docs": set()}
        )
        bucket["attention"].append(float(item.attention))  # type: ignore[union-attr]
        bucket["score_diff"].append(float(item.score_diff))  # type: ignore[union-attr]
        bucket["docs"].add(item.doc_id)  # type: ignore[union-attr]
    result = []
    for term, bucket in grouped.items():
        docs = tuple(sorted(bucket["docs"]))  # type: ignore[arg-type]
        if len(docs) < min_document_support:
            continue
        result.append(
            KeywordEvidence(
                term=term,
                doc_id="",
                attention=float(statistics.median(bucket["attention"])),  # type: ignore[arg-type]
                score_diff=float(statistics.median(bucket["score_diff"])),  # type: ignore[arg-type]
                document_support=len(docs),
                supporting_docs=docs,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                -item.document_support,
                -item.score_diff,
                -item.attention,
                item.term,
            ),
        )
    )


build_candidate_pool = aggregate_candidate_pool


def stable_subset_hash(subset: Iterable[str]) -> str:
    normalized = sorted({normalize_term(term) for term in subset if normalize_term(term)})
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


subset_hash = stable_subset_hash


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class JSONCheckpointCache:
    """Small fingerprinted JSON store with crash-safe atomic replacement."""

    def __init__(self, path: str | Path, fingerprint: str):
        self.path = Path(path)
        self.fingerprint = str(fingerprint)
        self._data: dict[str, Any] = {"fingerprint": self.fingerprint, "entries": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                loaded = None
            if isinstance(loaded, dict) and loaded.get("fingerprint") == self.fingerprint:
                self._data = loaded
                self._data.setdefault("entries", {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data["entries"].get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data["entries"][key] = _jsonable(value)
        self.flush()

    def delete(self, key: str) -> None:
        self._data["entries"].pop(key, None)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self._data,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @property
    def entries(self) -> Mapping[str, Any]:
        return dict(self._data["entries"])


CheckpointCache = JSONCheckpointCache


def build_inverted_index(
    evidence: Iterable[KeywordEvidence] | Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Build a normalized keyword-to-document index."""
    index: dict[str, set[str]] = {}
    if isinstance(evidence, Mapping):
        for doc_id, terms in evidence.items():
            for raw_term in terms:
                term = normalize_term(raw_term)
                if term:
                    index.setdefault(term, set()).add(str(doc_id))
    else:
        for item in evidence:
            term = normalize_term(item.term)
            if term:
                docs = item.supporting_docs or ((item.doc_id,) if item.doc_id else ())
                index.setdefault(term, set()).update(docs)
    return {term: tuple(sorted(docs)) for term, docs in sorted(index.items())}


inverted_keyword_index = build_inverted_index


def _bootstrap_lower_bound(
    values: Sequence[float], samples: int, confidence: float, seed: int
) -> float:
    if not values:
        return 0.0
    if len(values) == 1 or samples <= 0:
        return float(statistics.mean(values))
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    index = min(len(means) - 1, max(0, math.floor(alpha * len(means))))
    return float(means[index])


def evaluate_objective(
    evaluations: Mapping[str, DocumentEvaluation] | Iterable[DocumentEvaluation],
    subset: Iterable[str] = (),
    *,
    config: ObjectiveConfig = ObjectiveConfig(),
    baseline: Mapping[str, float | DocumentEvaluation] | None = None,
) -> EvaluationResult:
    """Compute a robust per-document delta objective."""
    items = (
        list(evaluations.values())
        if isinstance(evaluations, Mapping)
        else list(evaluations)
    )
    items.sort(key=lambda item: item.doc_id)
    deltas: list[float] = []
    for item in items:
        if baseline is None:
            base = item.baseline_quality
        else:
            raw_base = baseline.get(item.doc_id, item.baseline_quality)
            base = raw_base.quality if isinstance(raw_base, DocumentEvaluation) else float(raw_base)
        deltas.append(float(item.quality - base))
    mean_gain = float(statistics.mean(deltas)) if deltas else 0.0
    downside = float(statistics.mean(max(0.0, -delta) for delta in deltas)) if deltas else 0.0
    harm_rate = (
        sum(delta < -config.harm_threshold for delta in deltas) / len(deltas)
        if deltas
        else 0.0
    )
    fallback_rate = (
        sum(bool(item.fallback) for item in items) / len(items) if items else 0.0
    )
    keywords = tuple(sorted({normalize_term(term) for term in subset if normalize_term(term)}))
    size_cost = config.size_penalty * len(keywords)
    lower = _bootstrap_lower_bound(
        deltas,
        config.bootstrap_samples,
        config.confidence_level,
        config.bootstrap_seed,
    )
    objective = (
        mean_gain
        + config.confidence_weight * lower
        - config.downside_penalty * downside
        - config.harm_penalty * harm_rate
        - config.fallback_penalty * fallback_rate
        - size_cost
    )
    return EvaluationResult(
        subset=keywords,
        objective=float(objective),
        mean_gain=mean_gain,
        confidence_lower_bound=lower,
        downside=downside,
        harm_rate=float(harm_rate),
        fallback_rate=float(fallback_rate),
        size_penalty=float(size_cost),
        document_count=len(items),
        deltas=tuple(deltas),
    )


compute_objective = evaluate_objective


def multi_fidelity_panel_schedule(
    doc_ids: Iterable[str],
    *,
    panel_sizes: Sequence[int] = (),
    critical_doc_ids: Iterable[str] = (),
    guard_doc_ids: Iterable[str] = (),
    seed: int = 0,
) -> tuple[tuple[str, ...], ...]:
    """Create nested deterministic panels containing every critical/guard doc."""
    universe = sorted({str(doc_id) for doc_id in doc_ids})
    universe_set = set(universe)
    required = {
        str(doc_id)
        for doc_id in (*tuple(critical_doc_ids), *tuple(guard_doc_ids))
        if str(doc_id) in universe_set
    }
    remaining = sorted(
        universe_set - required,
        key=lambda doc_id: (
            hashlib.sha256(f"{seed}\0panel\0{doc_id}".encode()).digest(),
            doc_id,
        ),
    )
    sizes = [int(size) for size in panel_sizes if int(size) > 0]
    if not sizes or (universe and max(sizes) < len(universe)):
        sizes.append(len(universe))
    sizes = sorted(set(min(len(universe), size) for size in sizes))
    panels = []
    for size in sizes:
        target = max(size, len(required))
        panels.append(tuple(sorted(required | set(remaining[: max(0, target - len(required))]))))
    return tuple(dict.fromkeys(panels))


make_panel_schedule = multi_fidelity_panel_schedule


def _coerce_evaluations(
    values: Mapping[str, DocumentEvaluation | float],
) -> dict[str, DocumentEvaluation]:
    result = {}
    for doc_id, value in values.items():
        result[str(doc_id)] = (
            value
            if isinstance(value, DocumentEvaluation)
            else DocumentEvaluation(str(doc_id), float(value))
        )
    return result


def _result_from_json(data: Mapping[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        subset=tuple(data["subset"]),
        objective=float(data["objective"]),
        mean_gain=float(data["mean_gain"]),
        confidence_lower_bound=float(data["confidence_lower_bound"]),
        downside=float(data["downside"]),
        harm_rate=float(data["harm_rate"]),
        fallback_rate=float(data["fallback_rate"]),
        size_penalty=float(data["size_penalty"]),
        document_count=int(data["document_count"]),
        deltas=tuple(float(value) for value in data.get("deltas", ())),
    )


def sequential_forward_floating_selection(
    candidates: Iterable[str],
    doc_ids: Iterable[str],
    evaluator: Callable[
        [tuple[str, ...], tuple[str, ...]],
        Mapping[str, DocumentEvaluation | float],
    ],
    *,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    search_config: SearchConfig = SearchConfig(),
    critical_doc_ids: Iterable[str] = (),
    guard_doc_ids: Iterable[str] = (),
    checkpoint: JSONCheckpointCache | str | Path | None = None,
    fingerprint: str | None = None,
) -> SearchResult:
    """Budgeted deterministic SFFS with memoization and resumable state."""
    terms = tuple(sorted({normalize_term(term) for term in candidates if normalize_term(term)}))
    docs = tuple(sorted({str(doc_id) for doc_id in doc_ids}))
    critical_docs = tuple(sorted({str(doc_id) for doc_id in critical_doc_ids}))
    guard_docs = tuple(sorted({str(doc_id) for doc_id in guard_doc_ids}))
    panels = multi_fidelity_panel_schedule(
        docs,
        panel_sizes=search_config.panel_sizes,
        critical_doc_ids=critical_docs,
        guard_doc_ids=guard_docs,
        seed=search_config.seed,
    )
    if not panels:
        panels = ((),)
    config_signature = {
        "terms": terms,
        "docs": docs,
        "objective": asdict(objective_config),
        "search": {
            key: value
            for key, value in asdict(search_config).items()
            if key != "evaluation_budget"
        },
        "critical": critical_docs,
        "guard": guard_docs,
    }
    derived_fingerprint = hashlib.sha256(
        json.dumps(config_signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    store: JSONCheckpointCache | None
    if isinstance(checkpoint, JSONCheckpointCache):
        store = checkpoint
        if fingerprint is not None and store.fingerprint != fingerprint:
            raise ValueError("checkpoint fingerprint does not match search")
    elif checkpoint is not None:
        store = JSONCheckpointCache(checkpoint, fingerprint or derived_fingerprint)
    else:
        store = None

    saved = store.get("search_state") if store else None
    if saved:
        current = tuple(saved["current"])
        iteration = int(saved["iteration"])
        no_improvement = int(saved["no_improvement"])
        trace = [
            SearchStep(
                iteration=int(step["iteration"]),
                action=str(step["action"]),
                keyword=step.get("keyword"),
                subset=tuple(step["subset"]),
                objective=float(step["objective"]),
                panel_size=int(step["panel_size"]),
                evaluations_used=int(step["evaluations_used"]),
            )
            for step in saved.get("trace", ())
        ]
        memo = {
            key: _result_from_json(value) for key, value in saved.get("memo", {}).items()
        }
        baseline_cache = {
            key: {doc_id: float(score) for doc_id, score in value.items()}
            for key, value in saved.get("baselines", {}).items()
        }
        evaluations_used = int(saved.get("evaluations_used", len(memo)))
        cache_hits = int(saved.get("cache_hits", 0))
    else:
        current = ()
        iteration = 0
        no_improvement = 0
        trace: list[SearchStep] = []
        memo: dict[str, EvaluationResult] = {}
        baseline_cache: dict[str, dict[str, float]] = {}
        evaluations_used = 0
        cache_hits = 0

    def persist() -> None:
        if store:
            store.set(
                "search_state",
                {
                    "current": current,
                    "iteration": iteration,
                    "no_improvement": no_improvement,
                    "trace": trace,
                    "memo": memo,
                    "baselines": baseline_cache,
                    "evaluations_used": evaluations_used,
                    "cache_hits": cache_hits,
                },
            )

    def score(subset: tuple[str, ...], panel: tuple[str, ...]) -> EvaluationResult | None:
        nonlocal evaluations_used, cache_hits
        canonical = tuple(sorted(subset))
        panel_key = hashlib.sha256(
            json.dumps(panel, separators=(",", ":")).encode()
        ).hexdigest()
        key = f"{stable_subset_hash(canonical)}:{panel_key}"
        if key in memo:
            cache_hits += 1
            return memo[key]
        if evaluations_used >= search_config.evaluation_budget:
            return None
        base = baseline_cache.get(panel_key)
        if base is None:
            if evaluations_used >= search_config.evaluation_budget:
                return None
            raw_baseline = _coerce_evaluations(evaluator((), panel))
            evaluations_used += 1
            base = {doc_id: item.quality for doc_id, item in raw_baseline.items()}
            baseline_cache[panel_key] = base
            empty_result = evaluate_objective(
                raw_baseline, (), config=objective_config, baseline=base
            )
            memo[f"{stable_subset_hash(())}:{panel_key}"] = empty_result
            if not canonical:
                persist()
                return empty_result
        if evaluations_used >= search_config.evaluation_budget:
            return None
        raw = _coerce_evaluations(evaluator(canonical, panel))
        evaluations_used += 1
        result = evaluate_objective(
            raw, canonical, config=objective_config, baseline=base
        )
        memo[key] = result
        persist()
        return result

    stop_reason = "converged"
    if not docs:
        empty = evaluate_objective({}, (), config=objective_config)
        return SearchResult((), empty, tuple(trace), "no_documents", evaluations_used, cache_hits)
    if not terms:
        base = score((), panels[-1]) or evaluate_objective({}, ())
        return SearchResult((), base, tuple(trace), "no_candidates", evaluations_used, cache_hits)

    while True:
        if len(current) >= search_config.max_keywords:
            stop_reason = "max_keywords"
            break
        if evaluations_used >= search_config.evaluation_budget:
            stop_reason = "evaluation_budget"
            break
        if no_improvement >= search_config.patience:
            stop_reason = "patience"
            break

        panel = panels[min(iteration, len(panels) - 1)]
        incumbent = score(current, panel)
        if incumbent is None:
            stop_reason = "evaluation_budget"
            break
        additions: list[tuple[float, str, EvaluationResult]] = []
        for term in terms:
            if term in current:
                continue
            proposal = tuple(sorted((*current, term)))
            result = score(proposal, panel)
            if result is None:
                stop_reason = "evaluation_budget"
                break
            additions.append((result.objective, term, result))
        if stop_reason == "evaluation_budget":
            break
        additions.sort(key=lambda item: (-item[0], item[1]))
        additions = additions[: max(1, search_config.beam_width)]

        # Successive-halving promotion.  A cheap panel may only screen
        # proposals; acceptance is based on the highest fidelity reachable
        # within the remaining budget.  Keep one third (at least one) after
        # every promotion round.
        accepted_panel = panel
        for promoted_panel in panels[min(iteration, len(panels) - 1) + 1 :]:
            promoted: list[tuple[float, str, EvaluationResult]] = []
            for _, term, _ in additions:
                proposal = tuple(sorted((*current, term)))
                result = score(proposal, promoted_panel)
                if result is None:
                    break
                promoted.append((result.objective, term, result))
            if not promoted:
                break
            promoted.sort(key=lambda item: (-item[0], item[1]))
            keep = max(1, math.ceil(len(promoted) / 3))
            additions = promoted[:keep]
            accepted_panel = promoted_panel

        best_add = additions[0] if additions else None
        promoted_incumbent = score(current, accepted_panel)
        if promoted_incumbent is None:
            stop_reason = "evaluation_budget"
            break
        if best_add is None or best_add[0] <= promoted_incumbent.objective + search_config.min_improvement:
            no_improvement += 1
            iteration += 1
            trace.append(
                SearchStep(
                    iteration, "no_improvement", None, current,
                    promoted_incumbent.objective, len(accepted_panel), evaluations_used
                )
            )
            persist()
            continue

        _, added_term, accepted = best_add
        current = accepted.subset
        iteration += 1
        no_improvement = 0
        trace.append(
            SearchStep(
                iteration, "add", added_term, current,
                accepted.objective, len(accepted_panel), evaluations_used
            )
        )
        persist()

        # Floating backward phase: repeatedly remove any term that improves
        # the just-accepted subset on the same (promoted) panel.
        panel = accepted_panel
        while len(current) > 1:
            incumbent = score(current, panel)
            if incumbent is None:
                stop_reason = "evaluation_budget"
                break
            removals: list[tuple[float, str, EvaluationResult]] = []
            for term in current:
                proposal = tuple(item for item in current if item != term)
                result = score(proposal, panel)
                if result is None:
                    stop_reason = "evaluation_budget"
                    break
                removals.append((result.objective, term, result))
            if stop_reason == "evaluation_budget":
                break
            removals.sort(key=lambda item: (-item[0], item[1]))
            best_remove = removals[0]
            if best_remove[0] <= incumbent.objective + search_config.min_improvement:
                break
            _, removed_term, accepted = best_remove
            current = accepted.subset
            iteration += 1
            trace.append(
                SearchStep(
                    iteration, "remove", removed_term, current,
                    accepted.objective, len(panel), evaluations_used
                )
            )
            persist()
        if stop_reason == "evaluation_budget":
            break

    final = score(current, panels[-1])
    if final is None:
        # Budget exhaustion can prevent a full-panel score; use the highest
        # fidelity cached result for this exact subset.
        matches = [
            result for result in memo.values() if result.subset == tuple(sorted(current))
        ]
        final = max(matches, key=lambda result: result.document_count) if matches else evaluate_objective({}, current)
    persist()
    return SearchResult(
        keywords=tuple(sorted(current)),
        evaluation=final,
        trace=tuple(trace),
        stop_reason=stop_reason,
        evaluations_used=evaluations_used,
        cache_hits=cache_hits,
    )


sffs_search = sequential_forward_floating_selection
search_keywords = sequential_forward_floating_selection


def stability_selection(
    selections: Iterable[Iterable[str]],
    *,
    threshold: float = 0.6,
) -> tuple[tuple[str, ...], float]:
    """Return frequently selected terms and mean pairwise Jaccard stability."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    runs = [
        {normalize_term(term) for term in selection if normalize_term(term)}
        for selection in selections
    ]
    if not runs:
        return (), 1.0
    universe = sorted(set().union(*runs))
    selected = tuple(
        term
        for term in universe
        if sum(term in run for run in runs) / len(runs) >= threshold
    )
    similarities = []
    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    stability = float(statistics.mean(similarities)) if similarities else 1.0
    return selected, stability


select_stable_keywords = stability_selection
