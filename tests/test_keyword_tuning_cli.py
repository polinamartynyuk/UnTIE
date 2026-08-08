from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "05_Tune_model_keywords.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("keyword_tuning_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_resume_budget_and_strategy_controls() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.max_keywords == 20
    assert args.evaluation_budget == 250
    assert args.stability_threshold == 0.4
    assert args.cache_dir.name == "keyword_tuning_cache"


def test_parser_exposes_improved_selection_controls() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    assert args.selection_policy == "relaxed"
    assert args.require_non_empty is True
    assert args.harm_cap == 0.12
    assert args.min_activation_rate == 0.20
    assert args.use_conditional_gain is True


def test_parser_accepts_multiple_explicit_strategies() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--strategy",
            "only_score_diff",
            "weighted_score",
            "combined_score",
            "--strategy",
            "only_weight",
            "highest_cohesion",
            "highest_similarity",
        ]
    )
    assert args.strategy == [
        ["only_score_diff", "weighted_score", "combined_score"],
        ["only_weight", "highest_cohesion", "highest_similarity"],
    ]


def test_load_english_training_dataset_cleans_references(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "en-1",
                    "original_text": "Document",
                    "tasks": ["Semantic_Segmentation"],
                }
            ]
        ),
        encoding="utf-8",
    )
    frame = module.load_training_dataset(path, "en")
    assert frame.loc[0, "tasks_cleaned"] == ["Semantic Segmentation"]


def test_load_russian_training_dataset_parses_list_column(tmp_path) -> None:
    module = _load_module()
    path = tmp_path / "dataset.csv"
    path.write_text(
        'id;text_clean;Task_aspects\nru-1;"Текст";"[\'анализ текстов\']"\n',
        encoding="utf-8",
    )
    frame = module.load_training_dataset(path, "ru")
    assert frame.loc[0, "doc_id"] == "ru-1"
    assert frame.loc[0, "tasks_cleaned"] == ["анализ текстов"]
