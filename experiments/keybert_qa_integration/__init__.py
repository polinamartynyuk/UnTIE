"""Isolated KeyBERT + extractive QA integration experiment (hyp3).

Does not change production UnTIE baseline / attention / static-keyword behaviour.
KeyBERT is only a thematic prior; attention keywords stay the task-specific signal.
"""

CE_NAME = "ExtractiveQuestionAnswerer"
CE_MODULE = "untie.models.ExtractiveQuestionAnswerer"
