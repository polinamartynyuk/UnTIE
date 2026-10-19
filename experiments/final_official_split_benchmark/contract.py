"""Predeclared scientific contract for the final benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import (
    CELLS as HISTORICAL_CELLS,
    DITY_REVISION,
    MINILM_REVISION,
)
from experiments.qa_chunk_selection_v8.contract import H1_COLUMNS, H2_COLUMNS, H3_COLUMNS

CELLS = ("scirex", "scier_384", "scier_128", "russerrc", "scimdix")
SEEDS = (42, 43, 44)
INNER_FOLDS = 4
SPLIT_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 10_000
NI_MARGIN = -1.0

SPLIT_OFFICIAL_TVT = "OFFICIAL_TRAIN_VAL_TEST"
SPLIT_OFFICIAL_TT = "OFFICIAL_TRAIN_TEST_INTERNAL_VAL"
SPLIT_NO_TEST = "NO_OFFICIAL_TEST"

REAL_SYSTEMS = ("B1", "B2", "B3", "B4a", "B4b", "B5-EN", "G1", "P1", "P2", "P3", "P4")
DIAGNOSTIC_SYSTEMS = ("D1", "D2")
ABLATIONS = ("P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK")
PROPOSED = ("P1", "P2", "P3", "P4")


@dataclass(frozen=True)
class Cell:
    key: str
    family: str
    language: str
    split_mode: str
    train: Path
    validation: Path | None
    test_runtime: Path | None
    test_gold: Path | None
    k: int
    chunk_size: int
    overlap: int
    official_source: str


ROOT = PROJECT_ROOT
EXT = ROOT / "artifacts/qa_chunk_selection_external_data_20260913"
SCIER384 = ROOT / "artifacts/qa_chunk_selection_scier_384_50_20260913/scier"
RUDATA = ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"

CELL_SPECS = {
    "scirex": Cell(
        "scirex", "SciREX", "en", SPLIT_OFFICIAL_TVT,
        ROOT / "artifacts/qa_chunk_selection_v3/data/dataset.jsonl", None,
        ROOT / "datasets/scirex_aspect_qa_preprocessing_test", ROOT / "datasets/scirex_aspect_qa_preprocessing_test",
        5, 384, 50, "datasets/scirex_release/release_data/{train,dev,test}.jsonl",
    ),
    "scier_384": Cell(
        "scier_384", "SciER", "en", SPLIT_OFFICIAL_TVT,
        SCIER384 / "data/dataset.jsonl", SCIER384 / "heldout/official_dev.jsonl",
        SCIER384 / "locked/official_test_all.jsonl", SCIER384 / "locked/official_test.jsonl",
        5, 384, 50, "official SciER train/dev/test mapped in source_manifest.json",
    ),
    "scier_128": Cell(
        "scier_128", "SciER", "en", SPLIT_OFFICIAL_TVT,
        EXT / "scier/data/dataset.jsonl", EXT / "scier/heldout/official_dev.jsonl",
        EXT / "scier/locked/official_test_all.jsonl", EXT / "scier/locked/official_test.jsonl",
        5, 128, 24, "official SciER train/dev/test mapped in source_manifest.json",
    ),
    "russerrc": Cell(
        "russerrc", "RuSERRC", "ru", SPLIT_NO_TEST,
        RUDATA / "data/answerable_dataset.jsonl", None, None, None,
        5, 128, 24, "RusErrC has no official hidden test; deterministic document holdout",
    ),
    "scimdix": Cell(
        "scimdix", "SciMDIX", "ru", SPLIT_OFFICIAL_TT,
        EXT / "scimdix/data/dataset.jsonl", None,
        EXT / "scimdix/locked/official_test_all.jsonl", EXT / "scimdix/locked/official_test.jsonl",
        4, 128, 24, "official SciMDIX train/test mapped in source_manifest.json",
    ),
}

MODEL_REGISTRY = {
    "B0_EN": {"model_id": "cross-encoder/ms-marco-MiniLM-L6-v2", "revision": MINILM_REVISION, "trainable": True},
    "B0_RU": {"model_id": "DiTy/cross-encoder-russian-msmarco", "revision": DITY_REVISION, "trainable": True},
    "BGE": {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "frozen-at-audit", "trainable": False},
    "QA_EN": {"path": str(HISTORICAL_CELLS["scirex"].qa_model.resolve()), "trainable": False},
    "QA_RU": {"path": str(HISTORICAL_CELLS["russerrc"].qa_model.resolve()), "trainable": False},
    "MODERNBERT_EN": {"model_id": "answerdotai/ModernBERT-base", "revision": "frozen-at-audit", "trainable": True},
    "B5_RU": {"status": "NOT_RUN", "reason": "no RU/multilingual extractive reader was predeclared"},
    "B5_EN_LONG": {"status": "NOT_RUN", "reason": "optional system omitted before TEST"},
    "QWEN": {"model_id": "Qwen/Qwen2.5-3B-Instruct", "revision": "frozen-at-audit", "trainable": False},
}

SYSTEM_REGISTRY = {
    "B1": "frozen language reader on every canonical chunk; maximum QA confidence",
    "B2": "dataset-trained canonical B0 top-1 followed by frozen language reader",
    "B3": "canonical B0 top-K followed by frozen reader; maximum QA confidence",
    "B4a": "frozen BGE reranker top-1 followed by frozen language reader",
    "B4b": "frozen BGE reranker top-K followed by frozen reader; maximum QA confidence",
    "B5-EN": "fine-tuned answerdotai/ModernBERT-base on the same B0 top-K; EN only",
    "B5-RU": "NOT_RUN: no valid modern RU/multilingual extractive reader predeclared before TEST",
    "B5-EN-LONG": "NOT_RUN: optional long-context comparator omitted before TEST",
    "G1": "frozen Qwen2.5-3B-Instruct, deterministic full-document prompt",
    "P1": "exact V7-S2 feature schema and policy",
    "P2": "exact V9 H2 policy",
    "P3": "exact V9 H2-ACS selection, local per cell",
    "P4": "exact V8 H3 fixed feature schema and policy",
    "D1": "QA oracle inside canonical B0 top-K; diagnostic only",
    "D2": "positive-evidence oracle inside canonical B0 top-K; diagnostic only",
    "P2-NO-FULLLEX": "H2 without fulllex fields, mandatory ablation",
    "P2-NO-PAIRLEX": "H2 without PairLEX fields, mandatory ablation",
    "P2-NO-RISK": "H2 utility model with harm gate disabled, exact ablation",
}

QWEN_PROMPTS = {
    "en": "Answer the question using only the document. Return only the shortest answer span. If absent, return NO_ANSWER.\nQuestion: {question}\nDocument:\n{document}\nAnswer:",
    "ru": "Ответь на вопрос, используя только документ. Верни только самый короткий фрагмент ответа. Если ответа нет, верни НЕТ_ОТВЕТА.\nВопрос: {question}\nДокумент:\n{document}\nОтвет:",
}



def system_registry() -> dict:
    result = {}
    for system, definition in SYSTEM_REGISTRY.items():
        applicable = [cell for cell in CELLS if not (system in {"B5-EN", "B5-EN-LONG"} and CELL_SPECS[cell].language != "en") and not (system == "B5-RU" and CELL_SPECS[cell].language != "ru")]
        result[system] = {
            "definition": definition,
            "applicable_cells": applicable,
            "implementation": "experiments.final_official_split_benchmark.systems" if system.startswith(("B", "G")) else "experiments.final_official_split_benchmark.proposed",
            "model_seeds": list(SEEDS) if system in {"B5-EN", "P1", "P2", "P3", "P4", "P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK"} else [0],
            "test_tuning": False,
            "diagnostic_only": system in DIAGNOSTIC_SYSTEMS,
        }
    return result


def fairness_matrix() -> list[dict]:
    rows = []
    for cell in CELLS:
        for system in REAL_SYSTEMS + ABLATIONS:
            if system == "B5-EN" and CELL_SPECS[cell].language != "en": continue
            candidate = "all canonical chunks" if system == "B1" else ("frozen BGE ranking" if system in {"B4a", "B4b"} else ("full canonical document" if system == "G1" else "canonical B0 ranking"))
            count = "all" if system in {"B1", "G1"} else (1 if system in {"B2", "B4a"} else CELL_SPECS[cell].k)
            reader = "Qwen/Qwen2.5-3B-Instruct" if system == "G1" else ("answerdotai/ModernBERT-base" if system == "B5-EN" else f"frozen canonical {CELL_SPECS[cell].language} extractive reader")
            rows.append({"cell": cell, "system": system, "language": CELL_SPECS[cell].language,
                "training_data": "TRAIN then frozen refit TRAIN+VAL" if system in {"B2", "B3", "B5-EN", "P1", "P2", "P3", "P4", *ABLATIONS} else "none/frozen pretrained",
                "training_labels": "evidence plus answer Token F1" if system in {"P1", "P2", "P3", "P4", *ABLATIONS} else ("extractive spans" if system == "B5-EN" else "ranker evidence labels" if system in {"B2", "B3"} else "none"),
                "chunking": f"{CELL_SPECS[cell].chunk_size}/{CELL_SPECS[cell].overlap}", "candidate_source": candidate, "candidate_count": count,
                "reader": reader, "reader_trainable": system == "B5-EN", "reranker": "BAAI/bge-reranker-v2-m3" if system in {"B4a", "B4b"} else ("canonical B0" if candidate == "canonical B0 ranking" else None),
                "post_processing": "max QA confidence" if system in {"B1", "B3", "B4b", "B5-EN"} else ("H2-family policy" if system.startswith("P") else "none"),
                "whole_document_context": system == "G1", "neural_forwards": "measured in fresh-process resources"})
    return rows

def frozen_contract() -> dict:
    return {
        "experiment": "Final Official-Split Benchmark / H2 Family Confirmation",
        "version": "1.0.0",
        "cells": {key: {**asdict(value), **{"train": str(value.train), "validation": str(value.validation) if value.validation else None, "test_runtime": str(value.test_runtime) if value.test_runtime else None, "test_gold": str(value.test_gold) if value.test_gold else None}} for key, value in CELL_SPECS.items()},
        "systems": system_registry(), "fairness_matrix": fairness_matrix(),
        "models": MODEL_REGISTRY,
        "seeds": list(SEEDS), "inner_folds": INNER_FOLDS, "split_seed": SPLIT_SEED,
        "local_only": True,
        "primary_population": "canonical answerable instances; same question IDs for every applicable real system",
        "primary_metric": "project canonical Token F1 (0..100)",
        "secondary_metrics": ["character F1", "Unicode ROUGE-L", "exact match", "RU lemma F1 where available"],
        "empty_answer_policy": "questions always remain in denominator; empty answer gets zero quality and confidence -1e30 for max-confidence selection; if all are empty deterministic rank/chunk tie-break applies",
        "fairness": {"same_chunking_within_cell": True, "same_question_wording": True, "same_answer_normalization": True,
                     "B2_B3_P_family_candidate_source": "canonical B0", "B4_candidate_source": "frozen external BGE",
                     "B5_context": "same canonical B0 top-K", "G1_context": "different full-document context regime"},
        "h2_columns": list(H2_COLUMNS), "v7_s2_columns": list(H1_COLUMNS), "h3_columns": list(H3_COLUMNS),
        "b0_k": {key: value.k for key, value in CELL_SPECS.items()},
        "modernbert": {"checkpoint": "answerdotai/ModernBERT-base", "learning_rates": [2e-5, 3e-5], "epochs": [2, 3], "max_length": 512, "stride": 128, "selection": "validation Token F1; lower compute tie-break"},
        "qwen": {"model": "Qwen/Qwen2.5-3B-Instruct", "prompts": QWEN_PROMPTS, "do_sample": False, "max_new_tokens": 64, "max_input_tokens": 8192, "truncation_side": "right", "prompt_candidates": 1},
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "cluster": "canonical source document", "scier_coupled": True},
        "noninferiority_margin_token_f1": NI_MARGIN,
        "test_firewall": {"predictions_before_gold": True, "all_real_systems_complete_before_evaluation": True, "immutable_predictions": True, "smoke_test_forbidden": True},
        "optional_systems": {"B5-RU": "NOT_RUN", "B5-EN-LONG": "NOT_RUN"},
    }
