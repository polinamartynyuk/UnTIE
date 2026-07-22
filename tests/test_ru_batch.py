from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "03_Keywords_with_attention_refactored_ru.py"


def load_ru_batch_module():
    spec = importlib.util.spec_from_file_location("untie_ru_batch", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ru_parser_uses_isolated_chunk_defaults() -> None:
    module = load_ru_batch_module()
    args = module.build_parser().parse_args([])
    assert args.chunk_max_tokens == 128
    assert args.overlap_tokens == 24


def test_ru_strategy_diagnostics_detect_real_choice() -> None:
    module = load_ru_batch_module()
    diagnostics = module.strategy_diagnostics(
        total_chunk_count=3,
        scored_chunk_count=2,
        answers=[
            SimpleNamespace(text="анализ текстов"),
            SimpleNamespace(text="классификация текстов"),
        ],
        valid_answer_count=2,
        keyword_count=5,
        chunk_max_tokens=128,
    )
    assert diagnostics["strategy_applicable"] is True
    assert diagnostics["distinct_answer_count"] == 2
    assert diagnostics["answer_candidate_count"] == 2


def test_ru_failure_row_has_stable_diagnostic_schema() -> None:
    module = load_ru_batch_module()
    record = pd.Series(
        {
            "doc_id": "ru-1",
            "original_text": "Текст.",
            "tasks_cleaned": ["задача"],
        }
    )
    row = module.no_valid_answer_row(
        record,
        None,
        None,
        total_chunk_count=2,
        chunk_max_tokens=128,
    )
    assert row["total_chunk_count"] == 2
    assert row["scored_chunk_count"] == 0
    assert row["answer_candidate_count"] == 0
    assert row["distinct_answer_count"] == 0
    assert row["strategy_applicable"] is False
    assert row["chunk_max_tokens"] == 128
