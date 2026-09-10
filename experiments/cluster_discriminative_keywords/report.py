"""Markdown / JSON summaries for the hyp4 experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(
    path,
    *,
    question: str,
    clustering: Mapping[str, object],
    ranking_rows: Sequence[Mapping[str, Any]],
    qa_rows: Sequence[Mapping[str, Any]],
    chosen: Mapping[str, Any],
    hypotheses: Mapping[str, str],
) -> None:
    lines = [
        "# Cluster-specific discriminative QA keywords",
        "",
        f"- question: {question}",
        f"- frozen clustering: {clustering.get('config_id')}",
        "- KeyBERT is a candidate source only; it does not replace attention",
        "- `diff` keeps cosine(aspect)−cosine(gold); negative diff is not a topic keyword",
        f"- ranking gate winner: {chosen.get('ranking_method', '')}",
        f"- QA recipe: {chosen.get('qa_recipe', '')}",
        "",
        "## Chunk ranking (dev unless noted)",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking_rows:
        lines.append(
            "| {method} | {r1} | {r3} | {r5} | {r10} | {mrr} | {med} |".format(
                method=row.get("method", ""),
                r1=_fmt(row.get("evidence_recall@1")),
                r3=_fmt(row.get("evidence_recall@3")),
                r5=_fmt(row.get("evidence_recall@5")),
                r10=_fmt(row.get("evidence_recall@10")),
                mrr=_fmt(row.get("mrr")),
                med=_fmt(row.get("median_rank"), 2),
            )
        )
    lines.extend(
        [
            "",
            "## QA quality (test)",
            "",
            "| Method | Recall@5 | Char F1 | Token F1 | ROUGE-L | BERTScore | QA chunks ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in qa_rows:
        lines.append(
            "| {method} | {r5} | {cf} | {tf} | {rl} | {bs} | {ratio} |".format(
                method=row.get("method", ""),
                r5=_fmt(row.get("evidence_recall@5")),
                cf=_fmt(row.get("char_f1")),
                tf=_fmt(row.get("token_f1"), 2),
                rl=_fmt(row.get("rouge_l_f1")),
                bs=_fmt(row.get("bertscore_f1")),
                ratio=_fmt(row.get("qa_chunk_ratio")),
            )
        )
    lines.extend(["", "## Hypotheses", ""])
    for name, verdict in hypotheses.items():
        lines.append(f"- **{name}**: {verdict}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
