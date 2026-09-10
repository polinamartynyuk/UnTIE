"""Fixed paraphrase sets. Same questions for every document; no document facts."""

from __future__ import annotations

from untie.aspect_qa.config import ASPECT_QUESTIONS

# Index 0 is the preprocessing canonical question (Q0 baseline).
# Q3 uses [:3], Q5 uses [:5]. All ask the same aspect; none mention a paper-specific entity.
ASPECT_QUESTION_VARIANTS: dict[str, tuple[str, ...]] = {
    "method": (
        ASPECT_QUESTIONS["method"],
        "Which method or model does the paper use or propose?",
        "What method or model is introduced or employed in the paper?",
        "Which modeling approach is used or proposed in the paper?",
        "What method or model is presented in the paper?",
    ),
    "task": (
        ASPECT_QUESTIONS["task"],
        "Which task does the paper address?",
        "What is the main task studied in the paper?",
        "Which task is the focus of the paper?",
        "What task is the paper concerned with?",
    ),
    "metric": (
        ASPECT_QUESTIONS["metric"],
        "Which evaluation metric does the paper use?",
        "What metric is used to evaluate the results in the paper?",
        "Which performance metric is reported in the paper?",
        "What evaluation measure is used in the paper?",
    ),
    "material": (
        ASPECT_QUESTIONS["material"],
        "Which dataset or material does the paper use?",
        "What data or dataset is used in the paper?",
        "Which dataset is used in the paper?",
        "What material or dataset is employed in the paper?",
    ),
}


def questions_for(aspect: str, n_questions: int) -> tuple[str, ...]:
    variants = ASPECT_QUESTION_VARIANTS[aspect]
    if n_questions < 1 or n_questions > len(variants):
        raise ValueError(f"n_questions must be 1..{len(variants)} for {aspect}")
    selected = variants[: int(n_questions)]
    if selected[0] != ASPECT_QUESTIONS[aspect]:
        raise ValueError(f"canonical question drifted for aspect {aspect}")
    return selected


def assert_canonical_baseline() -> None:
    for aspect, canonical in ASPECT_QUESTIONS.items():
        if ASPECT_QUESTION_VARIANTS[aspect][0] != canonical:
            raise ValueError(f"Q0 baseline mismatch for {aspect}")
