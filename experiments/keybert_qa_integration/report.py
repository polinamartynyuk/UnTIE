"""Markdown summary and metric aggregation for hyp3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.keybert_topic_clustering.downstream import bootstrap_mean_ci


CORE_ORDER = (
    "qa_baseline",
    "qa_attention",
    "qa_attention_diff",
    "qa_keybert",
    "qa_attention_keybert",
    "qa_attention_diff_keybert",
)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_mode(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    if not rows:
        return {}
    char_values = [float(row.get("char_f1", 0.0)) for row in rows]
    mean, lo, hi = bootstrap_mean_ci(char_values, seed=seed)
    # bootstrap_mean_ci always uses 1000; recompute if a smaller smoke budget is requested.
    if bootstrap_samples != 1000:
        rng = np.random.default_rng(seed)
        array = np.asarray(char_values, dtype=float)
        means = [
            float(rng.choice(array, size=len(array), replace=True).mean())
            for _ in range(max(1, bootstrap_samples))
        ]
        lo, hi = (float(x) for x in np.quantile(means, [0.025, 0.975]))
        mean = float(array.mean())
    coverage = _mean(rows, "coverage")
    if coverage == 0.0 and rows:
        coverage = sum(1.0 for row in rows if str(row.get("prediction", "")).strip()) / len(rows)
    return {
        "n": len(rows),
        "char_f1": _mean(rows, "char_f1"),
        "token_f1": _mean(rows, "token_f1"),
        "rouge_l_f1": _mean(rows, "rouge_l_f1"),
        "bertscore_f1": _mean(rows, "bertscore_f1"),
        "bertscore_p": _mean(rows, "bertscore_p"),
        "bertscore_r": _mean(rows, "bertscore_r"),
        "exact_match": _mean(rows, "exact_match"),
        "coverage": coverage,
        "false_positive_rate": _mean(rows, "false_positive"),
        "fallback_rate": _mean(rows, "fallback"),
        "mean_latency_s": _mean(rows, "latency_s"),
        "char_f1_ci95_low": lo,
        "char_f1_ci95_high": hi,
        "char_f1_bootstrap_mean": mean,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def delta_pct(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return 100.0 * (value - baseline) / baseline


def bootstrap_paired_delta(
    baseline: Sequence[float],
    other: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 1000,
) -> dict[str, float]:
    left = np.asarray(list(baseline), dtype=float)
    right = np.asarray(list(other), dtype=float)
    if len(left) != len(right) or len(left) == 0:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(max(1, samples)):
        index = rng.integers(0, len(left), size=len(left))
        deltas.append(float(right[index].mean() - left[index].mean()))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean_delta": float(right.mean() - left.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": float(len(left)),
    }


def write_summary(path, payload: Mapping[str, Any]) -> None:
    chosen = payload.get("chosen") or {}
    cost = payload.get("cost") or {}
    paired = payload.get("paired") or {}
    questions = payload.get("conclusions") or {}
    lines = [
        "# KeyBERT + extractive QA integration (hyp3)",
        "",
        "## Dataset and isolation",
        "",
        f"- documents: {payload.get('n_documents')}",
        f"- splits: train {payload.get('n_train')} / dev {payload.get('n_dev')} / test {payload.get('n_test')}",
        f"- question: {payload.get('question')}",
        f"- output: `{payload.get('output_dir')}`",
        f"- evidence reuse: `{payload.get('evidence_dir')}`",
        "- production UnTIE behaviour is unchanged; this package only reads cached QA spans",
        "- test gold is never used for KeyBERT, attention mining, routing, or `diff`",
        f"- `diff` at test comes from the frozen global static dictionary: `{payload.get('static_model')}`",
        f"- train-only global keyword frequency documents: {payload.get('n_train')}",
        "",
        "## Environment",
        "",
        f"- KeyBERT: {payload.get('keybert_version')}",
        f"- sentence-transformers: {payload.get('st_version')}",
        f"- sklearn: {payload.get('sklearn_version')}",
        f"- KeyBERT encoder: {payload.get('topic_encoder')}",
        f"- QA: {payload.get('qa')}",
        f"- hardware: {payload.get('hardware')}",
        "",
        "## Fitted on dev, frozen for test",
        "",
        f"- KeyBERT topic variant: `{chosen.get('keybert_topic_variant')}`",
        f"- KEYBERT_TOP_K: {chosen.get('keybert_top_k')}",
        f"- KEYBERT_WEIGHT (attention+KeyBERT): {chosen.get('keybert_weight_att')}",
        f"- KEYBERT_WEIGHT (attention+diff+KeyBERT): {chosen.get('keybert_weight_full')}",
        f"- filter mode: `{chosen.get('filter_mode')}`",
        f"- cluster_k: {chosen.get('cluster_k')}",
        f"- training experiments 6–7: {payload.get('training_status')}",
        "",
        "## Core test table",
        "",
        payload.get("core_table_md", ""),
        "",
        "## Extra variants (priority 1–2)",
        "",
        payload.get("extra_table_md", ""),
        "",
        "## Ablation (independent channels)",
        "",
        payload.get("ablation_table_md", ""),
        "",
        "## Top-K KeyBERT on the best mix",
        "",
        payload.get("topk_table_md", ""),
        "",
        "## Document-level vs chunk-level KeyBERT",
        "",
        payload.get("scope_table_md", "Chunk-level KeyBERT was not run (default)."),
        "",
        "## Computational cost vs QA baseline",
        "",
        payload.get("cost_table_md", ""),
        "",
        f"- peak RAM (MB): {cost.get('peak_rss_mb')}",
        f"- peak GPU (MB): {cost.get('peak_gpu_mb')}",
        "",
        "## Bootstrap: attention+diff vs attention+diff+KeyBERT",
        "",
        f"- mean Δ char-F1: {paired.get('mean_delta')}",
        f"- 95% CI: [{paired.get('ci95_low')}, {paired.get('ci95_high')}]",
        "",
        "## Final questions",
        "",
        f"1. Does KeyBERT help on top of attention-keywords? {questions.get('q1')}",
        f"2. Best combination: {questions.get('q2')}",
        f"3. Redundant components: {questions.get('q3')}",
        f"4. Best quality/cost compromise: {questions.get('q4')}",
        f"5. Where to use KeyBERT: {questions.get('q5')}",
        "",
        "## Notes",
        "",
        "- KeyBERT-only is a control, not a target architecture.",
        "- Attention keywords remain the central task-specific signal.",
        "- Experiments 6–7 (hard negatives / auxiliary alignment) do not retrain RoBERTa unless `--run-training` is set; the default run only writes training data artifacts.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
