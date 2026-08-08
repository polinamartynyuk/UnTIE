"""Метрики качества извлечения ответов для оценки экспериментов.

Логика перенесена из ``scripts/04_Learning_text.ipynb`` (ячейки с ``char_f1``,
``token_f1``, ``rouge_l_f1``, ``bertscore_metrics`` и ``compute_metrics_rowwise``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

GoldType = str | list[str] | None

METRIC_COLUMNS = (
    "char_f1",
    "token_f1",
    "rouge_l_f1",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_gold(gold: GoldType) -> list[str]:
    """Приводит эталон к списку непустых строк."""
    if isinstance(gold, str):
        candidates: Iterable[Any] = [gold]
    elif gold is None:
        return []
    else:
        candidates = gold

    normalized: list[str] = []
    for item in candidates:
        if _is_missing(item):
            continue
        normalized.append(str(item).strip())
    return normalized


def normalize_pred(pred: Any) -> str:
    """Приводит предсказание к строке; пустые значения -> ``\"\"``."""
    if _is_missing(pred):
        return ""
    return str(pred).strip()


def char_f1(pred: Any, gold: GoldType) -> float:
    """Character-level F1; для нескольких эталонов берётся максимум."""
    if _is_missing(pred):
        return 0.0

    if isinstance(gold, str):
        references: Iterable[Any] = [gold]
    elif gold is None:
        return 0.0
    else:
        references = gold

    reference_items = [item for item in references if not _is_missing(item)]
    if not reference_items:
        return 0.0

    # Как в ноутбуке: пустоту проверяем через strip(), но символы считаем без strip().
    pred_lower = str(pred).lower()
    pred_chars = Counter(pred_lower)

    scores: list[float] = []
    for reference in reference_items:
        ref_lower = str(reference).lower()
        gold_chars = Counter(ref_lower)
        common = sum((pred_chars & gold_chars).values())
        if common == 0:
            continue
        precision = common / len(pred_lower)
        recall = common / len(ref_lower)
        scores.append(2 * precision * recall / (precision + recall))

    return max(scores) if scores else 0.0


_squad_metric: Any | None = None
_rouge_scorer: Any | None = None


def _get_squad_metric() -> Any:
    global _squad_metric
    if _squad_metric is None:
        import evaluate

        _squad_metric = evaluate.load("squad")
    return _squad_metric


def _get_rouge_scorer() -> Any:
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return _rouge_scorer


def token_f1(pred: Any, gold: GoldType) -> float:
    """Token-level F1 через метрику SQuAD (``evaluate.load('squad')``)."""
    pred_text = normalize_pred(pred)
    references = normalize_gold(gold)
    if not pred_text or not references:
        return 0.0

    squad_metric = _get_squad_metric()
    result = squad_metric.compute(
        predictions=[{"id": "0", "prediction_text": pred_text}],
        references=[
            {
                "id": "0",
                "answers": {
                    "text": references,
                    "answer_start": [0] * len(references),
                },
            }
        ],
    )
    return float(result["f1"])


def rouge_l_f1(pred: Any, gold: GoldType) -> float:
    """ROUGE-L F1; для нескольких эталонов берётся максимум."""
    pred_text = normalize_pred(pred)
    references = normalize_gold(gold)
    if not pred_text or not references:
        return 0.0

    scorer = _get_rouge_scorer()
    best_f1 = 0.0
    for reference in references:
        try:
            scores = scorer.score(reference, pred_text)
            best_f1 = max(best_f1, float(scores["rougeL"].fmeasure))
        except Exception:
            continue
    return best_f1


def bertscore_metrics(
    pred: Any,
    gold: GoldType,
    *,
    lang: str = "en",
) -> dict[str, float]:
    """BERTScore P/R/F1; для нескольких эталонов берётся максимум по каждой оси."""
    pred_text = normalize_pred(pred)
    references = normalize_gold(gold)
    if not pred_text or not references:
        return {"p": 0.0, "r": 0.0, "f1": 0.0}

    from bert_score import score

    best_p = best_r = best_f1 = 0.0
    for reference in references:
        precision, recall, f1 = score(
            cands=[pred_text],
            refs=[reference],
            lang=lang,
            rescale_with_baseline=False,
            verbose=False,
        )
        best_p = max(best_p, float(precision.item()))
        best_r = max(best_r, float(recall.item()))
        best_f1 = max(best_f1, float(f1.item()))

    return {"p": best_p, "r": best_r, "f1": best_f1}


def compute_row_metrics(
    pred: Any,
    gold: GoldType,
    *,
    lang: str = "en",
    include_bertscore: bool = True,
) -> dict[str, float]:
    """Считает все метрики для одной пары pred/gold."""
    metrics: dict[str, float] = {
        "char_f1": char_f1(pred, gold),
        "token_f1": token_f1(pred, gold),
        "rouge_l_f1": rouge_l_f1(pred, gold),
    }
    if include_bertscore:
        bert = bertscore_metrics(pred, gold, lang=lang)
        metrics["bertscore_p"] = bert["p"]
        metrics["bertscore_r"] = bert["r"]
        metrics["bertscore_f1"] = bert["f1"]
    return metrics


def _row_apply(
    frame: pd.DataFrame,
    func: Callable[[pd.Series], Any],
    *,
    show_progress: bool,
) -> pd.Series:
    if show_progress:
        from tqdm import tqdm

        tqdm.pandas()
        return frame.progress_apply(func, axis=1)
    return frame.apply(func, axis=1)


def compute_metrics_rowwise(
    df: pd.DataFrame,
    pred_col: str = "corrected_answer",
    gold_col: str = "tasks_cleaned",
    *,
    lang: str = "en",
    include_bertscore: bool = True,
    show_progress: bool = False,
    round_digits: int = 4,
) -> pd.DataFrame:
    """Добавляет колонки метрик к каждой строке DataFrame."""
    result = df.copy()

    result["char_f1"] = _row_apply(
        result,
        lambda row: char_f1(row[pred_col], row[gold_col]),
        show_progress=show_progress,
    )
    result["token_f1"] = _row_apply(
        result,
        lambda row: token_f1(row[pred_col], row[gold_col]),
        show_progress=show_progress,
    )
    result["rouge_l_f1"] = _row_apply(
        result,
        lambda row: rouge_l_f1(row[pred_col], row[gold_col]),
        show_progress=show_progress,
    )

    if include_bertscore:
        bert_results = _row_apply(
            result,
            lambda row: bertscore_metrics(row[pred_col], row[gold_col], lang=lang),
            show_progress=show_progress,
        )
        result["bertscore_p"] = bert_results.apply(lambda item: item["p"])
        result["bertscore_r"] = bert_results.apply(lambda item: item["r"])
        result["bertscore_f1"] = bert_results.apply(lambda item: item["f1"])

    round_columns = ["char_f1", "token_f1", "rouge_l_f1"]
    if include_bertscore:
        round_columns.extend(["bertscore_p", "bertscore_r", "bertscore_f1"])
    for column in round_columns:
        result[column] = result[column].round(round_digits)

    return result
