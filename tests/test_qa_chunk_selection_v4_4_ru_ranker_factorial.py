from pathlib import Path

from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.config import Config
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.contract import (
    DATASET_SPECS,
    RANKER_SPECS,
    SYSTEMS,
)
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.io_guard import (
    validate_dataset,
)


def test_factorial_contract_contains_both_datasets_and_rankers():
    assert set(DATASET_SPECS) == {"russerrc", "scimdix"}
    assert {spec.model_name for spec in RANKER_SPECS.values()} == {
        "ARGA100/ru-reranker-modernbert-small",
        "DiTy/cross-encoder-russian-msmarco",
    }
    assert all(len(spec.revision) == 40 for spec in RANKER_SPECS.values())


def test_protocol_has_symmetric_semantic_and_lexical_ablations():
    assert SYSTEMS == (
        "B0", "V6", "SEM", "LEX", "LEX-LEM",
        "ASEM", "ALEX", "ASEM-Q5", "ALEX-Q5",
    )


def test_phase_does_not_change_scientific_resume_fingerprint(tmp_path):
    left = Config(output_dir=tmp_path / "x", phase="rankers")
    right = Config(output_dir=tmp_path / "y", phase="protocol", resume=True)
    assert left.fingerprint_payload() == right.fingerprint_payload()


def test_completed_inputs_are_read_only_and_present():
    for spec in DATASET_SPECS.values():
        paths = validate_dataset(spec)
        assert all(Path(path).is_file() for path in paths.values())


def test_default_plan_is_full_factorial(tmp_path):
    config = Config(output_dir=tmp_path / "out")
    config.validate()
    assert config.datasets == ("russerrc", "scimdix")
    assert config.rankers == ("modernbert_small", "dity_msmarco")
    assert config.k_values == (3, 4, 5)
    assert config.seeds == (42, 43, 44)
