"""Оркестрация настройки статического словаря поверх evidence-кэша."""

from __future__ import annotations

import json
import math
import re
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .keyword_evidence import (
    CachedDocumentEvidence,
    ExtractionMetricCache,
    rerank_cached_document,
    stable_fingerprint,
)
from .keyword_tuning import (
    DocumentEvaluation,
    KeywordEvidence,
    ObjectiveConfig,
    SearchConfig,
    SearchResult,
    aggregate_candidate_pool,
    deterministic_document_split,
    evaluate_objective,
    sequential_forward_floating_selection,
    stability_selection,
)
from .model_params import KeywordMetadata
from .ranking import WeightedKeyword


SCORING_STRATEGIES: dict[str, float] = {
    "only_score_diff": 0.0,
    "only_weight": 1.0,
    "equal_weight_score_diff": 0.5,
}
CLUSTER_STRATEGIES = (
    "highest_avg_score",
    "weighted_score",
    "highest_cohesion",
)
ANSWER_STRATEGIES = (
    "highest_chunk_score",
    "highest_similarity",
    "combined_score",
)

ProgressCallback = Callable[[str, int, int, Mapping[str, Any]], None]


@dataclass(frozen=True)
class StrategyConfig:
    score_chunk_strategy: str = "equal_weight_score_diff"
    choose_cluster_strategy: str = "weighted_score"
    choose_answer_strategy: str = "combined_score"

    @property
    def weight_ratio(self) -> float:
        try:
            return SCORING_STRATEGIES[self.score_chunk_strategy]
        except KeyError as error:
            raise ValueError(
                f"Unknown score strategy: {self.score_chunk_strategy}"
            ) from error

    @property
    def name(self) -> str:
        return (
            f"{self.score_chunk_strategy}+{self.choose_cluster_strategy}"
            f"+{self.choose_answer_strategy}"
        )


@dataclass(frozen=True)
class MetricWeights:
    char_f1: float = 0.25
    token_f1: float = 0.25
    rouge_l_f1: float = 0.25
    bertscore_f1: float = 0.25

    def normalized(self, *, include_bertscore: bool) -> dict[str, float]:
        values = {
            "char_f1": self.char_f1,
            "token_f1": self.token_f1,
            "rouge_l_f1": self.rouge_l_f1,
        }
        if include_bertscore:
            values["bertscore_f1"] = self.bertscore_f1
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            raise ValueError("At least one metric weight must be positive")
        return {
            name: max(0.0, value) / total for name, value in values.items()
        }


@dataclass(frozen=True)
class TrainingConfig:
    language: str
    min_document_support: int = 2
    max_candidates: int = 150
    max_keywords: int = 20
    evaluation_budget: int = 250
    patience: int = 2
    beam_width: int = 5
    stability_runs: int = 5
    stability_threshold: float = 0.7
    seed: int = 42
    include_bertscore: bool = False
    critical_quality_threshold: float = 0.65
    guard_fraction: float = 0.1
    strategies: tuple[StrategyConfig, ...] = ()

    def strategy_grid(self) -> tuple[StrategyConfig, ...]:
        if self.strategies:
            return self.strategies
        return tuple(
            StrategyConfig(score, cluster, answer)
            for score in SCORING_STRATEGIES
            for cluster in CLUSTER_STRATEGIES
            for answer in ANSWER_STRATEGIES
        )


@dataclass(frozen=True)
class StrategyOutcome:
    strategy: StrategyConfig
    keywords: tuple[str, ...]
    objective: float
    mean_gain: float
    harm_rate: float
    fallback_rate: float
    confidence_lower_bound: float
    stability: float
    selections: tuple[tuple[str, ...], ...]
    stop_reasons: tuple[str, ...]
    evaluations_used: int
    traces: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class AblationResult:
    name: str
    objective: float
    mean_gain: float
    harm_rate: float
    fallback_rate: float
    keyword_count: int


@dataclass(frozen=True)
class TuningOutcome:
    strategy: StrategyConfig
    keywords: tuple[str, ...]
    keyword_metadata: tuple[KeywordMetadata, ...]
    objective: float
    mean_gain: float
    harm_rate: float
    fallback_rate: float
    confidence_lower_bound: float
    stability: float
    test_objective: float
    test_mean_gain: float
    test_harm_rate: float
    test_fallback_rate: float
    test_confidence_lower_bound: float
    seed: int
    train_doc_ids: tuple[str, ...]
    dev_doc_ids: tuple[str, ...]
    test_doc_ids: tuple[str, ...]
    strategy_outcomes: tuple[StrategyOutcome, ...]
    ablations: tuple[AblationResult, ...]
    fingerprint: str

    def tuning_metadata(self) -> dict[str, Any]:
        return {
            "method": "multi-fidelity-sffs-stability",
            "score": self.objective,
            "document_count": len(self.train_doc_ids) + len(self.dev_doc_ids),
            "random_seed": self.seed,
            "split_hash": stable_fingerprint(
                {
                    "train": self.train_doc_ids,
                    "dev": self.dev_doc_ids,
                    "test": self.test_doc_ids,
                }
            ),
            "strategy": asdict(self.strategy),
            "mean_gain": self.mean_gain,
            "harm_rate": self.harm_rate,
            "fallback_rate": self.fallback_rate,
            "confidence_lower_bound": self.confidence_lower_bound,
            "stability": self.stability,
            "test": {
                "objective": self.test_objective,
                "mean_gain": self.test_mean_gain,
                "harm_rate": self.test_harm_rate,
                "fallback_rate": self.test_fallback_rate,
                "confidence_lower_bound": self.test_confidence_lower_bound,
            },
            "release_recommended": (
                self.test_mean_gain > 0
                and self.test_confidence_lower_bound >= 0
                and self.test_harm_rate <= 0.1
            ),
            "fingerprint": self.fingerprint,
            "ablations": [asdict(item) for item in self.ablations],
        }


def candidate_evidence_from_documents(
    documents: Iterable[CachedDocumentEvidence],
) -> tuple[KeywordEvidence, ...]:
    result: list[KeywordEvidence] = []
    for document in documents:
        for candidate in document.candidates:
            result.append(
                KeywordEvidence(
                    term=candidate.word,
                    doc_id=document.doc_id,
                    attention=candidate.attention_weight,
                    score_diff=candidate.score_difference,
                )
            )
    return tuple(result)


def _keyword_map(
    pool: Sequence[KeywordEvidence],
    metadata_by_word: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, WeightedKeyword]:
    forms = metadata_by_word or {}
    result: dict[str, WeightedKeyword] = {}
    for item in pool:
        lemma, stem = forms.get(item.term, (item.term, item.term))
        result[item.term] = WeightedKeyword(
            word=item.term,
            lemma=lemma,
            stem=stem,
            attention_weight=max(0.0, item.attention),
            score_difference=max(0.0, item.score_diff),
        )
    return result


def metric_quality(
    metrics: Mapping[str, float],
    *,
    weights: MetricWeights,
    include_bertscore: bool,
) -> float:
    normalized_weights = weights.normalized(
        include_bertscore=include_bertscore
    )
    values = {
        "char_f1": float(metrics.get("char_f1", 0.0)),
        "token_f1": float(metrics.get("token_f1", 0.0)) / 100.0,
        "rouge_l_f1": float(metrics.get("rouge_l_f1", 0.0)),
        "bertscore_f1": float(metrics.get("bertscore_f1", 0.0)),
    }
    return sum(values[name] * weight for name, weight in normalized_weights.items())


class CachedKeywordSubsetEvaluator:
    """Adapter from cached documents to the model-agnostic SFFS callback."""

    def __init__(
        self,
        documents: Mapping[str, CachedDocumentEvidence],
        keyword_map: Mapping[str, WeightedKeyword],
        metric_cache: ExtractionMetricCache,
        *,
        language: str,
        strategy: StrategyConfig,
        include_bertscore: bool = False,
        metric_weights: MetricWeights = MetricWeights(),
    ) -> None:
        self.documents = dict(documents)
        self.keyword_map = dict(keyword_map)
        self.metric_cache = metric_cache
        self.language = language
        self.strategy = strategy
        self.include_bertscore = include_bertscore
        self.metric_weights = metric_weights
        self._baseline_quality: dict[str, float] = {}

    def baseline_quality(self, document: CachedDocumentEvidence) -> float:
        if document.doc_id not in self._baseline_quality:
            metrics = self.metric_cache.score(
                document.baseline_answer,
                document.references,
                language=self.language,
                include_bertscore=self.include_bertscore,
            )
            self._baseline_quality[document.doc_id] = metric_quality(
                metrics,
                weights=self.metric_weights,
                include_bertscore=self.include_bertscore,
            )
        return self._baseline_quality[document.doc_id]

    def __call__(
        self, subset: tuple[str, ...], doc_ids: tuple[str, ...]
    ) -> dict[str, DocumentEvaluation]:
        keywords = [
            self.keyword_map[term]
            for term in subset
            if term in self.keyword_map
        ]
        evaluations: dict[str, DocumentEvaluation] = {}
        for doc_id in doc_ids:
            document = self.documents[doc_id]
            if keywords:
                prediction, diagnostics = rerank_cached_document(
                    document,
                    keywords,
                    weight_ratio=self.strategy.weight_ratio,
                    cluster_strategy=self.strategy.choose_cluster_strategy,
                    answer_strategy=self.strategy.choose_answer_strategy,
                )
            else:
                prediction = document.baseline_answer
                diagnostics = {"fallback": True}
            metrics = self.metric_cache.score(
                prediction,
                document.references,
                language=self.language,
                include_bertscore=self.include_bertscore,
            )
            quality = metric_quality(
                metrics,
                weights=self.metric_weights,
                include_bertscore=self.include_bertscore,
            )
            evaluations[doc_id] = DocumentEvaluation(
                doc_id=doc_id,
                quality=quality,
                baseline_quality=self.baseline_quality(document),
                fallback=bool(diagnostics.get("fallback", False)),
            )
        return evaluations


def critical_and_guard_documents(
    documents: Mapping[str, CachedDocumentEvidence],
    evaluator: CachedKeywordSubsetEvaluator,
    doc_ids: Sequence[str],
    candidate_terms: Sequence[str],
    *,
    quality_threshold: float,
    guard_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    term_patterns = [
        re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        for term in candidate_terms
    ]
    critical = []
    good = []
    for doc_id in sorted(doc_ids):
        document = documents[doc_id]
        baseline = evaluator.baseline_quality(document)
        has_candidate = any(
            pattern.search(chunk.text)
            for pattern in term_patterns
            for chunk in document.chunks
        )
        if baseline < quality_threshold and len(document.chunks) > 1 and has_candidate:
            critical.append(doc_id)
        elif baseline >= quality_threshold:
            good.append(doc_id)
    guard_count = min(
        len(good),
        max(1, math.ceil(len(doc_ids) * guard_fraction)) if good else 0,
    )
    guards = sorted(
        good,
        key=lambda doc_id: stable_fingerprint({"seed": seed, "doc_id": doc_id}),
    )[:guard_count]
    return tuple(critical), tuple(sorted(guards))


def _stability_panels(
    doc_ids: Sequence[str], runs: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    docs = tuple(sorted(set(doc_ids)))
    if not docs:
        return ((),)
    panels = []
    size = max(1, math.ceil(len(docs) * 0.8))
    for run in range(max(1, runs)):
        ranked = sorted(
            docs,
            key=lambda doc_id: stable_fingerprint(
                {"seed": seed + run, "stability": doc_id}
            ),
        )
        panels.append(tuple(sorted(ranked[:size])))
    return tuple(panels)


def _prune_stable_subset(
    subset: tuple[str, ...],
    evaluator: CachedKeywordSubsetEvaluator,
    doc_ids: tuple[str, ...],
    objective_config: ObjectiveConfig,
) -> tuple[tuple[str, ...], Any]:
    current = tuple(sorted(subset))
    current_eval = evaluate_objective(
        evaluator(current, doc_ids), current, config=objective_config
    )
    changed = True
    while changed and len(current) > 1:
        changed = False
        proposals = []
        for term in current:
            proposal = tuple(item for item in current if item != term)
            result = evaluate_objective(
                evaluator(proposal, doc_ids),
                proposal,
                config=objective_config,
            )
            proposals.append((result.objective, term, proposal, result))
        best = max(proposals, key=lambda item: (item[0], item[1]))
        if best[0] > current_eval.objective:
            _, _, current, current_eval = best
            changed = True
    return current, current_eval


def tune_global_keywords(
    documents: Sequence[CachedDocumentEvidence],
    *,
    config: TrainingConfig,
    metric_cache: ExtractionMetricCache,
    objective_config: ObjectiveConfig = ObjectiveConfig(),
    metric_weights: MetricWeights = MetricWeights(),
    split: Mapping[str, Sequence[str]] | None = None,
    checkpoint_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TuningOutcome:
    """Настраивает словарь и стратегию, не используя test при выборе."""
    by_id = {document.doc_id: document for document in documents}
    split_map = (
        {name: tuple(values) for name, values in split.items()}
        if split is not None
        else deterministic_document_split(by_id, seed=config.seed)
    )
    train_ids = tuple(split_map["train"])
    dev_ids = tuple(split_map["dev"])
    test_ids = tuple(split_map["test"])
    if not train_ids or not dev_ids:
        raise ValueError("Keyword tuning requires non-empty train and dev splits")

    evidence = candidate_evidence_from_documents(
        by_id[doc_id] for doc_id in train_ids
    )
    pool = aggregate_candidate_pool(
        evidence,
        train_ids,
        min_document_support=config.min_document_support,
    )[: config.max_candidates]
    if not pool:
        raise ValueError("Candidate pool is empty after train-only filtering")
    if progress_callback is not None:
        progress_callback(
            "candidate_pool",
            len(pool),
            config.max_candidates,
            {"candidate_count": len(pool)},
        )

    form_votes: dict[str, list[tuple[str, str]]] = {}
    for doc_id in train_ids:
        for candidate in by_id[doc_id].candidates:
            form_votes.setdefault(candidate.word, []).append(
                (candidate.lemma, candidate.stem)
            )
    forms = {
        word: max(set(values), key=values.count)
        for word, values in form_votes.items()
    }
    keywords = _keyword_map(pool, forms)
    candidate_terms = tuple(item.term for item in pool)
    outcomes: list[StrategyOutcome] = []
    run_details: dict[str, list[SearchResult]] = {}
    strategies = config.strategy_grid()

    for strategy_index, strategy in enumerate(strategies, start=1):
        if progress_callback is not None:
            progress_callback(
                "strategy_start",
                strategy_index,
                len(strategies),
                {"strategy": strategy.name},
            )
        evaluator = CachedKeywordSubsetEvaluator(
            by_id,
            keywords,
            metric_cache,
            language=config.language,
            strategy=strategy,
            include_bertscore=config.include_bertscore,
            metric_weights=metric_weights,
        )
        critical, guards = critical_and_guard_documents(
            by_id,
            evaluator,
            dev_ids,
            candidate_terms,
            quality_threshold=config.critical_quality_threshold,
            guard_fraction=config.guard_fraction,
            seed=config.seed,
        )
        selections = []
        search_results = []
        for run, panel in enumerate(
            _stability_panels(dev_ids, config.stability_runs, config.seed)
        ):
            panel_sizes = tuple(
                sorted(
                    {
                        max(1, math.ceil(len(panel) * fraction))
                        for fraction in (0.25, 0.5, 1.0)
                    }
                )
            )
            checkpoint = (
                Path(checkpoint_dir) / f"{strategy.name}-run-{run}.json"
                if checkpoint_dir is not None
                else None
            )
            result = sequential_forward_floating_selection(
                candidate_terms,
                panel,
                evaluator,
                objective_config=objective_config,
                search_config=SearchConfig(
                    max_keywords=config.max_keywords,
                    evaluation_budget=config.evaluation_budget,
                    patience=config.patience,
                    beam_width=config.beam_width,
                    panel_sizes=panel_sizes,
                    seed=config.seed + run,
                ),
                critical_doc_ids=set(critical) & set(panel),
                guard_doc_ids=set(guards) & set(panel),
                checkpoint=checkpoint,
            )
            selections.append(result.keywords)
            search_results.append(result)
            if progress_callback is not None:
                progress_callback(
                    "stability_run",
                    run + 1,
                    max(1, config.stability_runs),
                    {
                        "strategy": strategy.name,
                        "keywords": result.keywords,
                        "objective": result.evaluation.objective,
                        "stop_reason": result.stop_reason,
                        "evaluations_used": result.evaluations_used,
                    },
                )
        stable, stability = stability_selection(
            selections, threshold=config.stability_threshold
        )
        final_keywords, final_evaluation = _prune_stable_subset(
            stable, evaluator, dev_ids, objective_config
        )
        outcomes.append(
            StrategyOutcome(
                strategy=strategy,
                keywords=final_keywords,
                objective=final_evaluation.objective,
                mean_gain=final_evaluation.mean_gain,
                harm_rate=final_evaluation.harm_rate,
                fallback_rate=final_evaluation.fallback_rate,
                confidence_lower_bound=final_evaluation.confidence_lower_bound,
                stability=stability,
                selections=tuple(selections),
                stop_reasons=tuple(item.stop_reason for item in search_results),
                evaluations_used=sum(
                    item.evaluations_used for item in search_results
                ),
                traces=tuple(item.trace for item in search_results),
            )
        )
        run_details[strategy.name] = search_results
        if progress_callback is not None:
            progress_callback(
                "strategy_complete",
                strategy_index,
                len(strategies),
                {
                    "strategy": strategy.name,
                    "keywords": final_keywords,
                    "objective": final_evaluation.objective,
                    "stability": stability,
                },
            )

    best = max(
        outcomes,
        key=lambda item: (
            item.objective,
            item.mean_gain,
            -item.harm_rate,
            item.strategy.name,
        ),
    )
    best_pool = {item.term: item for item in pool}
    frequencies = {
        term: sum(term in selection for selection in best.selections)
        / len(best.selections)
        for term in best.keywords
    }
    best_evaluator = CachedKeywordSubsetEvaluator(
        by_id,
        keywords,
        metric_cache,
        language=config.language,
        strategy=best.strategy,
        include_bertscore=config.include_bertscore,
        metric_weights=metric_weights,
    )
    full_result = evaluate_objective(
        best_evaluator(best.keywords, dev_ids),
        best.keywords,
        config=objective_config,
    )
    test_result = evaluate_objective(
        best_evaluator(best.keywords, test_ids),
        best.keywords,
        config=objective_config,
    )
    baseline_test = evaluate_objective(
        best_evaluator((), test_ids),
        (),
        config=objective_config,
    )
    frequency_keywords = tuple(
        item.term for item in pool[: max(1, len(best.keywords))]
    )
    frequency_test = evaluate_objective(
        best_evaluator(frequency_keywords, test_ids),
        frequency_keywords,
        config=objective_config,
    )
    ablations = (
        AblationResult(
            "empty_baseline",
            baseline_test.objective,
            baseline_test.mean_gain,
            baseline_test.harm_rate,
            baseline_test.fallback_rate,
            0,
        ),
        AblationResult(
            "frequency_only",
            frequency_test.objective,
            frequency_test.mean_gain,
            frequency_test.harm_rate,
            frequency_test.fallback_rate,
            len(frequency_keywords),
        ),
        AblationResult(
            "floating_tuned",
            test_result.objective,
            test_result.mean_gain,
            test_result.harm_rate,
            test_result.fallback_rate,
            len(best.keywords),
        ),
    )
    metadata = []
    for term in best.keywords:
        without = tuple(item for item in best.keywords if item != term)
        without_result = evaluate_objective(
            best_evaluator(without, dev_ids),
            without,
            config=objective_config,
        )
        source = best_pool[term]
        weighted = keywords[term]
        metadata.append(
            KeywordMetadata(
                word=term,
                lemma=weighted.lemma,
                stem=weighted.stem,
                attention_weight=weighted.attention_weight,
                score_difference=weighted.score_difference,
                document_support=source.document_support,
                selection_frequency=frequencies[term],
                marginal_gain=full_result.objective - without_result.objective,
            )
        )

    fingerprint = stable_fingerprint(
        {
            "config": asdict(config),
            "objective": asdict(objective_config),
            "metrics": asdict(metric_weights),
            "documents": {
                doc_id: by_id[doc_id].fingerprint for doc_id in sorted(by_id)
            },
            "selection": best.keywords,
            "strategy": asdict(best.strategy),
        }
    )
    metric_cache.save()
    outcome = TuningOutcome(
        strategy=best.strategy,
        keywords=best.keywords,
        keyword_metadata=tuple(metadata),
        objective=full_result.objective,
        mean_gain=full_result.mean_gain,
        harm_rate=full_result.harm_rate,
        fallback_rate=full_result.fallback_rate,
        confidence_lower_bound=full_result.confidence_lower_bound,
        stability=best.stability,
        test_objective=test_result.objective,
        test_mean_gain=test_result.mean_gain,
        test_harm_rate=test_result.harm_rate,
        test_fallback_rate=test_result.fallback_rate,
        test_confidence_lower_bound=test_result.confidence_lower_bound,
        seed=config.seed,
        train_doc_ids=train_ids,
        dev_doc_ids=dev_ids,
        test_doc_ids=test_ids,
        strategy_outcomes=tuple(
            sorted(outcomes, key=lambda item: item.objective, reverse=True)
        ),
        ablations=ablations,
        fingerprint=fingerprint,
    )
    if progress_callback is not None:
        progress_callback(
            "complete",
            1,
            1,
            {
                "strategy": outcome.strategy.name,
                "keywords": outcome.keywords,
                "objective": outcome.objective,
            },
        )
    return outcome


def save_tuning_trace(outcome: TuningOutcome, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_strategy": asdict(outcome.strategy),
        "keywords": list(outcome.keywords),
        "keyword_metadata": [
            item.to_dict() for item in outcome.keyword_metadata
        ],
        "objective": outcome.objective,
        "mean_gain": outcome.mean_gain,
        "harm_rate": outcome.harm_rate,
        "fallback_rate": outcome.fallback_rate,
        "confidence_lower_bound": outcome.confidence_lower_bound,
        "stability": outcome.stability,
        "test": {
            "objective": outcome.test_objective,
            "mean_gain": outcome.test_mean_gain,
            "harm_rate": outcome.test_harm_rate,
            "fallback_rate": outcome.test_fallback_rate,
            "confidence_lower_bound": outcome.test_confidence_lower_bound,
        },
        "release_recommended": (
            outcome.test_mean_gain > 0
            and outcome.test_confidence_lower_bound >= 0
            and outcome.test_harm_rate <= 0.1
        ),
        "split": {
            "train": outcome.train_doc_ids,
            "dev": outcome.dev_doc_ids,
            "test": outcome.test_doc_ids,
        },
        "strategy_outcomes": [
            {
                **asdict(item),
                "strategy": asdict(item.strategy),
            }
            for item in outcome.strategy_outcomes
        ],
        "ablations": [asdict(item) for item in outcome.ablations],
        "fingerprint": outcome.fingerprint,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)


def save_strategy_summary_csv(
    outcome: TuningOutcome, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "strategy",
        "objective",
        "mean_gain",
        "harm_rate",
        "fallback_rate",
        "confidence_lower_bound",
        "stability",
        "keyword_count",
        "evaluations_used",
        "stop_reasons",
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in outcome.strategy_outcomes:
            writer.writerow(
                {
                    "strategy": item.strategy.name,
                    "objective": item.objective,
                    "mean_gain": item.mean_gain,
                    "harm_rate": item.harm_rate,
                    "fallback_rate": item.fallback_rate,
                    "confidence_lower_bound": item.confidence_lower_bound,
                    "stability": item.stability,
                    "keyword_count": len(item.keywords),
                    "evaluations_used": item.evaluations_used,
                    "stop_reasons": "|".join(item.stop_reasons),
                }
            )
    temporary.replace(destination)
