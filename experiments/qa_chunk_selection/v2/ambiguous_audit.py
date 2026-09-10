"""Ambiguous subtype audit. Does not relabel."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.sanity import gold_string_in_text
from experiments.qa_chunk_selection.schema import InstanceRecord

from .io_util import write_json, write_jsonl
from .qa import answer_chunk
from .schemas import chunk_to_runtime, to_runtime
from .signals import dense_scores, relevance_scores


def run_ambiguous_audit(
    records: Sequence[InstanceRecord],
    *,
    encode,
    relevance,
    answerability,
    answer_fn,
    cache,
    model_id: str,
    inference_hash: str,
    output_dir,
    tag: str,
    progress=None,
    skip_bertscore: bool = True,
    top_n: int = 40,
) -> dict[str, Any]:
    rows = []
    subtype_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(records, start=1):
        runtime = to_runtime(item)
        dense = dense_scores(runtime, encode) if encode is not None else [0.0] * len(item.chunks)
        rel = relevance_scores(runtime, relevance) if relevance is not None else [0.0] * len(item.chunks)
        ans = answerability.score_instance(runtime) if answerability is not None else [0.0] * len(item.chunks)
        ranked_ans = sorted(range(len(item.chunks)), key=lambda i: -float(ans[i]))
        top1 = item.chunks[ranked_ans[0]].chunk_id if ranked_ans else None
        top5 = {item.chunks[i].chunk_id for i in ranked_ans[:5]}
        for chunk, d_s, a_s, b_s in zip(item.chunks, dense, rel, ans):
            if chunk.label != "ambiguous":
                continue
            pred = answer_chunk(
                answer_fn,
                item.question,
                chunk_to_runtime(chunk),
                cache=cache,
                model_id=model_id,
                inference_hash=inference_hash,
            )
            metrics = qa_row(pred.get("qa_answer"), item.gold_answers, skip_bertscore=skip_bertscore)
            row = {
                "document_id": item.document_id,
                "question_id": item.question_id,
                "aspect": item.aspect,
                "split": item.split,
                "chunk_id": chunk.chunk_id,
                "subtype": chunk.primary_class,
                "chunk_text": chunk.text,
                "gold_answers": list(item.gold_answers),
                "dense_score": d_s,
                "a_relevance_score": a_s,
                "b_score": b_s,
                "qa_answer": pred.get("qa_answer"),
                "qa_confidence_proxy": pred.get("qa_confidence_proxy"),
                "gold_string_present": gold_string_in_text(item.gold_answers, chunk.text),
                "is_answerability_top1": chunk.chunk_id == top1,
                "in_answerability_top5": chunk.chunk_id in top5,
                **metrics,
            }
            rows.append(row)
            subtype_stats[chunk.primary_class].append(row)
        if progress:
            progress.count(index, len(records), item=f"ambiguous-audit-{tag}/{item.split}")
    summary = {
        "tag": tag,
        "n_ambiguous_rows": len(rows),
        "by_subtype": {
            subtype: {
                "count": len(items),
                "mean_a": _mean([row["a_relevance_score"] for row in items]),
                "mean_b": _mean([row["b_score"] for row in items]),
                "mean_qa_proxy": _mean([row.get("qa_confidence_proxy") or 0.0 for row in items]),
                "top1_rate": _mean([float(row["is_answerability_top1"]) for row in items]),
                "top5_rate": _mean([float(row["in_answerability_top5"]) for row in items]),
                "em": _mean([row.get("exact_match") or 0.0 for row in items]),
                "token_f1": _mean([row.get("token_f1") or 0.0 for row in items]),
            }
            for subtype, items in subtype_stats.items()
        },
    }
    write_jsonl(output_dir / "rows.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    export_top_errors(rows, output_dir, top_n=top_n)
    return summary


def export_top_errors(rows: Sequence[dict[str, Any]], output_dir, *, top_n: int) -> None:
    sorts = {
        "highest_answerability": sorted(rows, key=lambda row: -float(row.get("b_score") or 0))[:top_n],
        "top1_ambiguous": [row for row in rows if row.get("is_answerability_top1")][:top_n],
        "nonzero_f1": [row for row in rows if float(row.get("token_f1") or 0) > 0][:top_n],
        "gold_string_present": [row for row in rows if row.get("gold_string_present")][:top_n],
    }
    write_jsonl(output_dir / "top_errors.jsonl", [item for group in sorts.values() for item in group])
    lines = ["# Ambiguous top errors", "", "Labels are frozen. This export is not a relabeling.", ""]
    for name, group in sorts.items():
        lines.append(f"## {name}")
        for row in group[:20]:
            lines.append(
                f"- `{row['document_id']}` `{row['aspect']}` subtype={row['subtype']} "
                f"A={row['a_relevance_score']:.3f} B={row['b_score']:.3f} "
                f"QA=`{row.get('qa_answer')}` gold=`{row.get('gold_answers')}` "
                f"EM={row.get('exact_match')} F1={row.get('token_f1')}"
            )
            lines.append("")
            text = str(row.get("chunk_text") or "").replace("\n", " ")
            lines.append(f"  {text[:400]}")
            lines.append("")
    (output_dir / "top_errors.md").write_text("\n".join(lines), encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")
