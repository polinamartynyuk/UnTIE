"""Frozen canonical / paraphrase / ownership question sets. Do not tune after EVAL."""

from __future__ import annotations

from .contract import QUESTION_SET_VERSION

CANONICAL_FALLBACK = {
    "method": "What method or model is used or proposed in the paper?",
    "task": "What task is addressed in the paper?",
    "material": "What dataset or material is used in the paper?",
    "metric": "What evaluation metric is used in the paper?",
}

PARAPHRASES = {
    "method": (
        "What method do the authors propose or use?",
        "What model is used in this work?",
        "What approach is introduced in the paper?",
    ),
    "task": (
        "What problem do the authors solve?",
        "What task is studied in this work?",
        "What task is addressed in this paper?",
    ),
    "material": (
        "What dataset is used in this work?",
        "What material or corpus do the authors evaluate on?",
        "What data is used in the paper?",
    ),
    "metric": (
        "What metric do the authors report?",
        "What evaluation measure is used in this work?",
        "How is performance measured in the paper?",
    ),
}

OWNERSHIP = {
    "method": {
        "target": "What method do the authors of this paper use or propose?",
        "external": "What method from previous, cited, baseline, or compared work is mentioned?",
    },
    "task": {
        "target": "What task do the authors of this paper address?",
        "external": "What task from previous, cited, baseline, or compared work is mentioned?",
    },
    "material": {
        "target": "What dataset or material is used in this work?",
        "external": "What dataset or material is mentioned only for another work, baseline, or comparison?",
    },
    "metric": {
        "target": "What evaluation metric is used in this work?",
        "external": "What metric from previous, cited, baseline, or compared work is mentioned?",
    },
}


def aspect_key(aspect: str) -> str:
    text = str(aspect or "").strip().casefold()
    if text in CANONICAL_FALLBACK:
        return text
    for key in CANONICAL_FALLBACK:
        if key in text:
            return key
    return "method"


def canonical_question(item_question: str, aspect: str) -> str:
    text = str(item_question or "").strip()
    return text or CANONICAL_FALLBACK[aspect_key(aspect)]


def paraphrases_for(aspect: str) -> tuple[str, ...]:
    return PARAPHRASES[aspect_key(aspect)]


def ownership_for(aspect: str) -> dict[str, str]:
    return dict(OWNERSHIP[aspect_key(aspect)])


def question_payload() -> dict[str, object]:
    return {
        "question_set_version": QUESTION_SET_VERSION,
        "canonical_fallback": dict(CANONICAL_FALLBACK),
        "paraphrases": {key: list(value) for key, value in PARAPHRASES.items()},
        "ownership": OWNERSHIP,
        "note": "dataset instance.question is Q0 when present; paraphrases/ownership are frozen extras",
    }
