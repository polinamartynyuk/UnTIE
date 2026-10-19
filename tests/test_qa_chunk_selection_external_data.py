from pathlib import Path

import pytest

from experiments.qa_chunk_selection.external_data.common import (
    claim_output, detokenize_with_offsets, mentions_from_token_spans, parse_inline_annotations,
)
from experiments.qa_chunk_selection.external_data.config import (
    SCIER_NATIVE_MAP, SCIMDIX_NATIVE_MAP, ExternalDataConfig,
)
from experiments.qa_chunk_selection.external_data.pipeline import PRIMARY_TO_EXPERIMENT


def test_scimdix_inline_annotations_keep_exact_offsets():
    annotated = "Цель — [проверить метод|T1|AIM]. Применён [BERT|T2|METHOD]."
    text, mentions = parse_inline_annotations(annotated, SCIMDIX_NATIVE_MAP)
    assert text == "Цель — проверить метод. Применён BERT."
    assert [(m.aspect, text[m.char_start:m.char_end]) for m in mentions] == [
        ("goal", "проверить метод"), ("method", "BERT"),
    ]


def test_scier_document_offsets_and_entity_spans_are_exact():
    tokens = ["We", "use", "BERT", ".", "Results", "improve", "."]
    text, offsets = detokenize_with_offsets(tokens)
    mentions = mentions_from_token_spans([[[2, 2, "Method"]], []], tokens, offsets, SCIER_NATIVE_MAP)
    assert text == "We use BERT. Results improve."
    assert text[mentions[0].char_start:mentions[0].char_end] == "BERT"
    assert (mentions[0].token_start, mentions[0].token_end) == (2, 3)


def test_phase1_mapping_does_not_treat_other_annotations_as_hard_negatives():
    assert PRIMARY_TO_EXPERIMENT["hard_negative"] == "hard_negative"
    assert set(PRIMARY_TO_EXPERIMENT) == {
        "qa_positive", "evidence_positive_qa_failed", "confirmed_false_negative",
        "false_negative_candidate", "hard_negative", "easy_negative",
    }


def test_config_requires_local_sources_when_download_is_disabled(tmp_path: Path):
    with pytest.raises(ValueError):
        ExternalDataConfig(output_dir=tmp_path / "out", download=False).validate()


def test_output_guard_refuses_existing_and_project_code_paths(tmp_path: Path):
    output = claim_output(tmp_path / "fresh")
    assert output.is_dir()
    with pytest.raises(FileExistsError):
        claim_output(output)
    with pytest.raises(ValueError):
        claim_output(Path("experiments/qa_chunk_selection/external_data/new-output"))


def test_language_specific_chunking_defaults(tmp_path: Path):
    config = ExternalDataConfig(output_dir=tmp_path / "out")
    assert config.chunking("scimdix") == (128, 24)
    assert config.chunking("scier") == (384, 50)
    with pytest.raises(ValueError):
        config.chunking("unknown")
