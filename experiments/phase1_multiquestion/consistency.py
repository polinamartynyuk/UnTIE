"""QAConsistency and exclusive chunk labels. Any-correct is diagnostic only."""

from __future__ import annotations


def qa_consistency(n_correct: int, n_questions: int) -> float:
    if n_questions <= 0:
        raise ValueError("n_questions must be positive")
    return float(n_correct) / float(n_questions)


def consistency_class(consistency: float, *, threshold: float) -> str:
    """Partition: robust_negative / question_sensitive / robust_positive.

    question_sensitive is (0, threshold), not (0, 1), so the three labels
    stay mutually exclusive when the DEV threshold is below 1.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError("robust_positive threshold must be in (0, 1]")
    if consistency <= 0.0:
        return "robust_negative"
    if consistency + 1e-12 >= threshold:
        return "robust_positive"
    return "question_sensitive"
