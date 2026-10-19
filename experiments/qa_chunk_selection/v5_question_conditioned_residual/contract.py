"""Frozen, code-level V5-A contract recovered from historical artifacts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

SEEDS = (42, 43, 44)
OUTER_FOLDS = tuple(range(5))
INNER_FOLDS = 4
BUDGETS = (1.25, 1.50, 2.00)
PRIMARY_BUDGET = 2.00
SWITCH_THRESHOLD = 0.5
EPSILON = 1e-8
BOOTSTRAP_SEED = 42

MINILM_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MINILM_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MINILM_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots" / MINILM_REVISION
DITY_ID = "DiTy/cross-encoder-russian-msmarco"
DITY_REVISION = "9029bab08103ad171724b510d312befa5b476293"

EN_QA = PROJECT_ROOT / "scripts/models_processing/models/bert_eng_qa_baseroberta_model"
RU_QA = PROJECT_ROOT / "scripts/models_processing/models/rubert_ru_qa_model"


@dataclass(frozen=True)
class CellSpec:
    key: str
    corpus: str
    language: str
    records: Path
    splits: Path
    flags: Path | None
    qa_cache: Path
    historical_b0: Path
    historical_reference: Path
    question_policy: str
    questions_q5: Path | None
    question_selection: Path | None
    k: int
    ranker_id: str
    ranker_revision: str
    ranker_recipe: str
    qa_model: Path
    historical_system: str
    bge_reference: Path | None = None


V3 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v3"
V43 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_3"
V44 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4"
EXT_DATA = PROJECT_ROOT / "artifacts/qa_chunk_selection_external_data_20260913"
SCIER384_DATA = PROJECT_ROOT / "artifacts/qa_chunk_selection_scier_384_50_20260913/scier"
EXT_V44 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_external_20260913_213335"
RU_DATA = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"
V45 = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_5_ru/run_20260915_155234"


def _external(key: str, data: Path) -> CellSpec:
    ref = EXT_V44 / key
    return CellSpec(
        key=key, corpus="SciER" if key.startswith("scier") else "SciMDIX",
        language="en" if key.startswith("scier") else "ru",
        records=data / "data/dataset.jsonl", splits=data / "data/v3_cv_splits.json",
        flags=data / "data/hn_quality_flags.jsonl", qa_cache=ref / "q5/prediction_cache.jsonl",
        historical_b0=ref / "b0/oof_predictions_scored.jsonl",
        historical_reference=ref / "results/asem/predictions_scored.jsonl",
        question_policy="nested_q5" if key.startswith("scier") else "fixed_q0",
        questions_q5=data / "questions_q5.json", question_selection=ref / "tuning/selected_by_run.json",
        k=5 if key.startswith("scier") else 4,
        ranker_id=MINILM_ID if key.startswith("scier") else DITY_ID,
        ranker_revision=MINILM_REVISION if key.startswith("scier") else DITY_REVISION,
        ranker_recipe="legacy_b2a" if key.startswith("scier") else "v45_dity",
        qa_model=EN_QA if key.startswith("scier") else RU_QA,
        historical_system="ASEM" if key.startswith("scier") else "V4.5-S3",
        bge_reference=None,
    )


CELLS = {
    "scirex": CellSpec(
        "scirex", "SciREX", "en", V3 / "data/dataset.jsonl", V3 / "data/v3_cv_splits.json",
        None, V43 / "cache/qa_topk.jsonl", V3 / "analysis/oof/b0/predictions_scored.jsonl",
        V44 / "asem/predictions_scored.jsonl", "fixed_q0", None, None, 5,
        MINILM_ID, MINILM_REVISION, "legacy_b2a", EN_QA, "V4.4-ASEM",
    ),
    "scier_384": _external("scier_384", SCIER384_DATA),
    "scier_128": _external("scier_128", EXT_DATA / "scier"),
    "russerrc": CellSpec(
        "russerrc", "RuSERRC", "ru", RU_DATA / "data/answerable_dataset.jsonl",
        RU_DATA / "data/answerable_cv_splits.json", RU_DATA / "data/hn_quality_flags.jsonl",
        RU_DATA / "q5/prediction_cache.jsonl",
        V45 / "russerrc/rankers/dity/oof_predictions_scored.jsonl",
        V45 / "russerrc/systems/S3_adaptive/predictions_scored.jsonl",
        "fixed_q0", None, None, 5, DITY_ID, DITY_REVISION, "v45_dity", RU_QA,
        "V4.5-S3", V45 / "russerrc/rankers/bge/oof_predictions_scored.jsonl",
    ),
    "scimdix": _external("scimdix", EXT_DATA / "scimdix"),
}

CELLS["russerrc"] = replace(
    CELLS["russerrc"],
    historical_reference=V45 / "russerrc/oof/S3_adaptive/predictions_scored.jsonl",
)
CELLS["scimdix"] = replace(
    CELLS["scimdix"],
    historical_b0=V45 / "scimdix/rankers/dity/oof_predictions_scored.jsonl",
    historical_reference=V45 / "scimdix/oof/S3_adaptive/predictions_scored.jsonl",
    bge_reference=V45 / "scimdix/rankers/bge/oof_predictions_scored.jsonl",
    historical_system="V4.5-S3",
)

GATE_CONTINUOUS = (
    "effective_k", "candidate_count", "normalized_gap_12", "normalized_gap_13",
    "normalized_gap_1k", "topk_score_std", "topk_score_iqr",
    "normalized_score_entropy", "top1_zscore",
)
CONTEXT = (
    "effective_k", "candidate_count", "normalized_gap_12", "normalized_gap_13",
    "normalized_score_entropy", "top1_zscore",
)
FEATURE_SETS = {
    "A0": ("delta_b0",) + CONTEXT,
    "A1": ("delta_b0", "delta_q_answer_ranker") + CONTEXT,
    "A2": ("delta_b0", "delta_q_answer_ranker", "delta_source_support", "delta_answer_group") + CONTEXT,
}

# Subtypes explicitly representing inadequate evidence can supervise the negative tier.
# Potential false negatives/evidence-positive-reader-failures remain unresolved.
SUFFICIENT = {"positive", "robust_positive", "qa_positive"}
INSUFFICIENT = {
    "hard_negative", "easy_negative", "gold_supporting_robust_negative",
    "question_sensitive",
}
UNRESOLVED = {"ambiguous", "false_negative_candidate", "evidence_positive_qa_failed"}

LEGACY_RECIPE = dict(epochs=5, patience=2, batch_size=16, learning_rate=2e-5,
                     max_length=512, ratio=(1, 3, 1))
DITY_RECIPE = dict(epochs=5, patience=2, micro_batch_size=2, prediction_batch_size=8,
                   gradient_accumulation=8, learning_rate=1e-5, max_length=512)

