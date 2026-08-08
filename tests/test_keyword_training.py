from __future__ import annotations

import json

from untie.keyword_evidence import (
    CachedChunkAnswer,
    CachedDocumentEvidence,
    CandidateObservation,
)
from untie.keyword_training import (
    MetricWeights,
    StrategyConfig,
    TrainingConfig,
    metric_quality,
    save_strategy_summary_csv,
    save_tuning_trace,
    tune_global_keywords,
)
from untie.keyword_tuning import ObjectiveConfig


class FakeMetricCache:
    def __init__(self) -> None:
        self.references_seen: list[tuple[str, ...]] = []
        self.saved = False

    def score(
        self,
        prediction,
        references,
        *,
        language,
        include_bertscore=False,
    ):
        refs = tuple(references)
        self.references_seen.append(refs)
        exact = 1.0 if prediction in refs else 0.0
        result = {
            "char_f1": exact,
            "token_f1": exact * 100,
            "rouge_l_f1": exact,
        }
        if include_bertscore:
            result["bertscore_f1"] = exact
        return result

    def save(self) -> None:
        self.saved = True


def _document(
    doc_id: str,
    *,
    reference: str,
    with_candidate: bool,
) -> CachedDocumentEvidence:
    return CachedDocumentEvidence(
        doc_id=doc_id,
        aspect_id="task",
        question="Which task?",
        references=(reference,),
        chunks=(
            CachedChunkAnswer(0, "generic introduction", 2, "wrong", 0.8),
            CachedChunkAnswer(
                1,
                "useful keyword describes the task",
                5,
                reference,
                0.9,
            ),
        ),
        answer_similarity=((1.0, 0.0), (0.0, 1.0)),
        baseline_answer="wrong",
        baseline_confidence=0.8,
        candidates=(
            (
                CandidateObservation(
                    "useful",
                    "useful",
                    "useful",
                    0.8,
                    0.5,
                    (1,),
                ),
            )
            if with_candidate
            else ()
        ),
        fingerprint=f"fp-{doc_id}",
    )


def test_metric_quality_normalizes_token_f1_and_optional_bert() -> None:
    metrics = {
        "char_f1": 1.0,
        "token_f1": 50.0,
        "rouge_l_f1": 0.5,
        "bertscore_f1": 1.0,
    }
    assert metric_quality(
        metrics,
        weights=MetricWeights(),
        include_bertscore=False,
    ) == 2 / 3
    assert metric_quality(
        metrics,
        weights=MetricWeights(),
        include_bertscore=True,
    ) == 0.75


def test_tuning_selects_train_candidate_and_evaluates_test_only_at_end(tmp_path) -> None:
    documents = [
        _document("train-1", reference="gold train one", with_candidate=True),
        _document("train-2", reference="gold train two", with_candidate=True),
        _document("dev-1", reference="gold dev", with_candidate=False),
        _document("test-1", reference="gold test", with_candidate=False),
    ]
    cache = FakeMetricCache()
    progress_events = []
    outcome = tune_global_keywords(
        documents,
        config=TrainingConfig(
            language="en",
            min_document_support=2,
            max_keywords=2,
            max_candidates=5,
            evaluation_budget=20,
            patience=1,
            beam_width=2,
            stability_runs=2,
            stability_threshold=0.5,
            strategies=(StrategyConfig(),),
        ),
        metric_cache=cache,  # type: ignore[arg-type]
        objective_config=ObjectiveConfig(
            downside_penalty=0,
            harm_penalty=0,
            fallback_penalty=0,
            size_penalty=0,
            bootstrap_samples=10,
        ),
        split={
            "train": ("train-1", "train-2"),
            "dev": ("dev-1",),
            "test": ("test-1",),
        },
        checkpoint_dir=tmp_path / "checkpoints",
        progress_callback=lambda event, current, total, details: progress_events.append(
            (event, current, total, dict(details))
        ),
    )

    assert outcome.keywords == ("useful",)
    assert outcome.keyword_metadata[0].selection_frequency == 1.0
    assert outcome.mean_gain == 1.0
    assert cache.saved is True
    assert cache.references_seen.count(("gold test",)) == 4
    assert outcome.test_mean_gain == 1.0
    assert outcome.tuning_metadata()["release_recommended"] is True
    assert progress_events[0][0] == "candidate_pool"
    assert any(event[0] == "strategy_start" for event in progress_events)
    assert sum(event[0] == "stability_run" for event in progress_events) == 2
    assert progress_events[-1][0] == "complete"
    assert [item.name for item in outcome.ablations] == [
        "empty_baseline",
        "frequency_only",
        "floating_tuned",
    ]

    trace = tmp_path / "trace.json"
    save_tuning_trace(outcome, trace)
    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["keywords"] == ["useful"]
    assert payload["selected_strategy"]["score_chunk_strategy"]
    assert payload["ablations"][0]["name"] == "empty_baseline"

    summary = tmp_path / "strategies.csv"
    save_strategy_summary_csv(outcome, summary)
    assert "equal_weight_score_diff" in summary.read_text(encoding="utf-8")
