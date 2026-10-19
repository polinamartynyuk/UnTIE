#!/usr/bin/env python3
"""Recompute training-data composition and internal consistency checks.

The script reads the exact files consumed by the final benchmark. It does not
modify dataset artifacts and does not run model inference.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

SOURCES = {
    "SciREX": [
        (ROOT / "artifacts/qa_chunk_selection_v3/data/dataset.jsonl", {"source_split": {"train", "dev"}}),
    ],
    "SciER-384": [
        (ROOT / "artifacts/qa_chunk_selection_scier_384_50_20260913/scier/data/all_phase1_instances.jsonl", None),
        (ROOT / "artifacts/qa_chunk_selection_scier_384_50_20260913/scier/heldout/official_dev_all.jsonl", None),
    ],
    "SciER-128": [
        (ROOT / "artifacts/qa_chunk_selection_external_data_20260913/scier/data/all_phase1_instances.jsonl", None),
        (ROOT / "artifacts/qa_chunk_selection_external_data_20260913/scier/heldout/official_dev_all.jsonl", None),
    ],
    "RuSERRC": [
        (ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431/data/dataset_q5_all_unlocked.jsonl", {"split": {"train", "development"}}),
    ],
    "SciMDIX": [
        (ROOT / "artifacts/qa_chunk_selection_external_data_20260913/scimdix/data/all_phase1_instances.jsonl", None),
    ],
}

POSITIVE = {"qa_positive", "robust_positive"}
EVIDENCE_FAILED = {"evidence_positive_qa_failed", "gold_supporting_robust_negative"}
FALSE_NEGATIVE = {"false_negative_candidate"}
QUESTION_SENSITIVE = {"question_sensitive"}
HARD_NEGATIVE = {"hard_negative"}
EASY_NEGATIVE = {"easy_negative"}


def read_rows(path: Path, selector: dict | None):
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if selector and any(row.get(key) not in allowed for key, allowed in selector.items()):
            continue
        yield row


def category(primary: str) -> str:
    if primary in POSITIVE: return "positive"
    if primary in EVIDENCE_FAILED: return "evidence_failed"
    if primary in FALSE_NEGATIVE: return "false_negative_candidate"
    if primary in QUESTION_SENSITIVE: return "question_sensitive"
    if primary in HARD_NEGATIVE: return "hard_negative"
    if primary in EASY_NEGATIVE: return "easy_negative"
    return "other"


def audit_dataset(name: str, specs):
    rows = [row for path, selector in specs for row in read_rows(path, selector)]
    qids = [str(row["question_id"]) for row in rows]
    if len(qids) != len(set(qids)):
        duplicates = len(qids) - len(set(qids))
        raise RuntimeError(f"{name}: duplicate question ids: {duplicates}")
    all_counts = Counter()
    training_counts = Counter()
    raw_counts = Counter()
    support_checks = Counter()
    span_total = span_exact = span_missing_chunk = 0
    answerable = 0
    answerable_docs = set()
    all_docs = {str(row["document_id"]) for row in rows}
    for row in rows:
        chunk_by_id = {int(c["chunk_id"]): c for c in row["chunks"]}
        categorized = []
        has_positive = False
        for chunk in row["chunks"]:
            primary = str(chunk.get("primary_class") or "")
            cat = category(primary)
            categorized.append((cat, chunk))
            raw_counts[primary] += 1
            all_counts[cat] += 1
            support = bool(chunk.get("gold_supporting", False))
            if cat == "positive":
                has_positive = True
                if not support: support_checks["qa_positive_without_official_span"] += 1
            if cat == "evidence_failed" and not support:
                support_checks["evidence_failed_without_official_span"] += 1
            if cat in {"hard_negative", "easy_negative"} and support:
                support_checks["trainable_negative_with_official_span"] += 1
        if has_positive:
            answerable += 1
            answerable_docs.add(str(row["document_id"]))
            for cat, _ in categorized: training_counts[cat] += 1
        for span in row.get("gold_spans", []):
            span_total += 1
            chunk = chunk_by_id.get(int(span["chunk_id"]))
            if chunk is None:
                span_missing_chunk += 1
                continue
            start, end = int(span["start_char"]), int(span["end_char"])
            if chunk["text"][start:end] == str(span["text"]):
                span_exact += 1
    total_pairs = sum(training_counts.values())
    strict_neg = training_counts["hard_negative"] + training_counts["easy_negative"]
    quarantine = training_counts["evidence_failed"] + training_counts["false_negative_candidate"] + training_counts["question_sensitive"] + training_counts["other"]
    all_gold_candidates = all_counts["positive"] + all_counts["evidence_failed"]
    all_near_pool = all_counts["hard_negative"] + all_counts["false_negative_candidate"]
    return {
        "dataset": name,
        "candidate_questions": len(rows),
        "answerable_questions": answerable,
        "answerable_rate_pct": 100.0 * answerable / len(rows) if rows else 0.0,
        "source_documents": len(all_docs),
        "answerable_documents": len(answerable_docs),
        "training_chunk_question_pairs": total_pairs,
        "training_positive": training_counts["positive"],
        "training_hard_negative": training_counts["hard_negative"],
        "training_easy_negative": training_counts["easy_negative"],
        "training_evidence_failed": training_counts["evidence_failed"],
        "training_false_negative_candidate": training_counts["false_negative_candidate"],
        "training_question_sensitive": training_counts["question_sensitive"],
        "training_other": training_counts["other"],
        "strict_supervised_pairs": training_counts["positive"] + strict_neg,
        "strict_negative": strict_neg,
        "training_quarantined_pairs": quarantine,
        "training_quarantine_rate_pct": 100.0 * quarantine / total_pairs if total_pairs else 0.0,
        "all_positive": all_counts["positive"],
        "all_evidence_failed": all_counts["evidence_failed"],
        "all_false_negative_candidate": all_counts["false_negative_candidate"],
        "all_hard_negative": all_counts["hard_negative"],
        "positive_confirmation_pct_all": 100.0 * all_counts["positive"] / all_gold_candidates if all_gold_candidates else 0.0,
        "near_negative_quarantine_pct_all": 100.0 * all_counts["false_negative_candidate"] / all_near_pool if all_near_pool else 0.0,
        "qa_positive_without_official_span": support_checks["qa_positive_without_official_span"],
        "evidence_failed_without_official_span": support_checks["evidence_failed_without_official_span"],
        "trainable_negative_with_official_span": support_checks["trainable_negative_with_official_span"],
        "gold_spans_total": span_total,
        "gold_spans_exact": span_exact,
        "gold_span_exact_pct": 100.0 * span_exact / span_total if span_total else 0.0,
        "gold_span_missing_chunk": span_missing_chunk,
        "raw_primary_classes": dict(sorted(raw_counts.items())),
    }


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})


def make_plot(rows):
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r["dataset"] for r in rows]
    y = np.arange(len(rows))
    answer = [r["answerable_rate_pct"] for r in rows]
    pos = [100*r["training_positive"]/r["training_chunk_question_pairs"] for r in rows]
    neg = [100*r["strict_negative"]/r["training_chunk_question_pairs"] for r in rows]
    quar = [100*r["training_quarantined_pairs"]/r["training_chunk_question_pairs"] for r in rows]
    fig, axes = plt.subplots(1,2,figsize=(11.8,4.2),gridspec_kw={"width_ratios":[1,1.45]})
    ax=axes[0]
    bars=ax.barh(y,answer,color="#4C78A8",height=.64)
    ax.set_xlabel("Доля вопросов, %")
    ax.set_title("а) Допуск вопросов в обучение")
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,112); ax.grid(axis="x",alpha=.25)
    for bar,row in zip(bars,rows):
        label=f'{bar.get_width():.1f}% (n={row["answerable_questions"]})'
        if bar.get_width() >= 75:
            ax.text(bar.get_width()-1.8,bar.get_y()+bar.get_height()/2,label,
                    ha='right',va='center',fontsize=8,color='white',fontweight='bold')
        else:
            ax.text(bar.get_width()+1.5,bar.get_y()+bar.get_height()/2,label,
                    ha='left',va='center',fontsize=8)
    ax=axes[1]
    ax.barh(y,pos,label="положительные",color="#2A9D8F",height=.64)
    ax.barh(y,neg,left=pos,label="отрицательные",color="#457B9D",height=.64)
    bottom=np.array(pos)+np.array(neg)
    ax.barh(y,quar,left=bottom,label="неоднозначные, исключённые из обучения",color="#E9C46A",height=.64)
    ax.set_xlabel("Доля пар «вопрос - фрагмент», %")
    ax.set_title("б) Состав разметки кандидатов")
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,100); ax.grid(axis="x",alpha=.25)
    for row_index, segments in enumerate(zip(pos,neg,quar)):
        left=0.0
        for segment_index,value in enumerate(segments):
            if value >= 8:
                color = 'white' if segment_index < 2 else '#3A3326'
                ax.text(left+value/2,row_index,f'{value:.1f}%',ha='center',va='center',
                        fontsize=8,color=color,fontweight='bold')
            left += value
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",bbox_to_anchor=(.5,.015),
               ncol=3,frameon=False,fontsize=9)
    fig.suptitle("Результаты программной подготовки обучающих данных",fontsize=12,y=.98)
    fig.tight_layout(rect=(0,.10,1,.92),w_pad=2.2)
    fig.savefig(OUT/"data_preparation_statistics.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT/"data_preparation_statistics.svg",bbox_inches="tight")
    plt.close(fig)


def main():
    rows=[audit_dataset(name,specs) for name,specs in SOURCES.items()]
    (OUT/"audit_summary.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_csv(OUT/"training_examples_summary.csv",rows,[
        "dataset","candidate_questions","answerable_questions","answerable_rate_pct","source_documents","answerable_documents",
        "training_chunk_question_pairs","strict_supervised_pairs","training_quarantined_pairs","training_quarantine_rate_pct"])
    write_csv(OUT/"class_counts.csv",rows,[
        "dataset","training_positive","training_hard_negative","training_easy_negative","training_evidence_failed","training_false_negative_candidate","training_question_sensitive","training_other"])
    write_csv(OUT/"quality_metrics.csv",rows,[
        "dataset","positive_confirmation_pct_all","near_negative_quarantine_pct_all","qa_positive_without_official_span",
        "evidence_failed_without_official_span","trainable_negative_with_official_span","gold_spans_total","gold_spans_exact","gold_span_exact_pct","gold_span_missing_chunk"])
    make_plot(rows)
    for row in rows:
        print(row["dataset"],json.dumps({k:row[k] for k in row if k not in {"dataset","raw_primary_classes"}},ensure_ascii=False))

if __name__=="__main__": main()
