from __future__ import annotations

import inspect

from untie.models import ExtractiveQuestionAnswerer, ModelFactory


def test_supported_model_factory_uses_extractive_qa_only() -> None:
    source = inspect.getsource(ModelFactory) + inspect.getsource(
        ExtractiveQuestionAnswerer
    )
    assert "AutoModelForQuestionAnswering" in source
    assert "AutoModelForCausalLM" not in source
    assert ".generate(" not in source
