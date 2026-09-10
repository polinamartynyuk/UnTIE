"""One-off reconstruction of the 7 topic-shortlist harms. Does not write eval artifacts."""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from untie.ce_topic_shortlist import (
    collect_topic_ce_panel,
    ranked_topic_terms,
    shortlist_answers,
)
from untie.config import PipelineConfig
from untie.cross_encoder_ranking import SentenceTransformerCrossEncoder
from untie.extraction_metrics import char_f1
from untie.models import ModelFactory, profile_for_language, resolve_device
from untie.pipelines import DocumentProcessor
from untie.topics import (
    TopicDocument,
    load_topic_artifact,
    route_document_to_nearest_leaf,
)

HARM_IDS = [
    "007ab5528b3bd310a80d553cccad4b78dc496b02",
    "14ad9d060c1e8f0449e697ee189ac346353fbfbc",
    "1a6b67622d04df8e245575bf8fb2066fb6729720",
    "207e0ac5301a3c79af862951b70632ed650f74f7",
    "223319a93dcf3912bbc1e5f949e5ab4d53906e62",
    "372bc106c61e7eb004835e85bbfee997409f176a",
    "3e7f54801c886ea2061650fd24fc481e39be152f",
]
K = 16


def load_payload(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("documents")
    return payload


def references(raw: dict) -> list[str]:
    value = raw.get("tasks_cleaned", raw.get("tasks", ()))
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = (value,)
    values = value if isinstance(value, (list, tuple)) else (value,)
    return [
        str(item).replace("_", " ").strip()
        for item in values
        if item is not None and str(item).strip()
    ]


def to_document(raw: dict) -> TopicDocument:
    headings = raw.get("headings", ())
    if isinstance(headings, str):
        headings = (headings,)
    return TopicDocument(
        doc_id=str(raw.get("doc_id", raw.get("id"))),
        text=str(raw.get("original_text", raw.get("text", "")) or ""),
        title=str(raw.get("title", "") or ""),
        abstract=str(raw.get("abstract", "") or ""),
        headings=tuple(str(item) for item in headings),
        split="test",
    )


def aggregator_means(encoder, texts: list[str]) -> list[float]:
    if not texts:
        return []
    embeddings = np.asarray(encoder.encode(texts))
    similarities = cosine_similarity(embeddings)
    return [float(row.mean()) for row in similarities]


def main() -> None:
    pred_path = (
        ROOT
        / "artifacts/hierarchical_topic_optimized/en/ce_topic_query/evaluation"
        / "test_predictions_all_metrics.csv"
    )
    by_mode: dict[str, dict[str, dict]] = {}
    with pred_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_mode.setdefault(row["mode"], {})[row["doc_id"]] = row

    payload = load_payload(ROOT / "datasets/scirex_structured.json")
    raw_by_id = {str(item.get("doc_id", item.get("id"))): item for item in payload}

    config = PipelineConfig(profile=profile_for_language("en"), device="auto")
    models = ModelFactory(config)
    processor = DocumentProcessor(
        models.tokenizer, config, sentence_encoder=models.sentence_encoder
    )
    artifact = load_topic_artifact(
        ROOT / "artifacts/hierarchical_topic_optimized/en/topic_model.tuned.json"
    )
    reranker = SentenceTransformerCrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2", resolve_device("auto")
    )

    reports = []
    for doc_id in HARM_IDS:
        raw = raw_by_id[doc_id]
        document = to_document(raw)
        gold = references(raw)
        route = route_document_to_nearest_leaf(
            document,
            models.sentence_encoder,
            artifact,
            expected_encoder_revision=artifact.encoder_revision or None,
        )
        leaf_id = route.topics[0].node_id if route.topics else ""
        terms = (
            ranked_topic_terms(
                artifact.node_map[leaf_id].representative_terms, max_terms=8
            )
            if leaf_id
            else ()
        )
        panel = collect_topic_ce_panel(
            document,
            "Which task was solved?",
            processor=processor,
            answerer=models.answerer,
            reranker=reranker,
            leaf_terms=terms,
            leaf_id=leaf_id,
            references=gold,
        )
        baseline_pred = by_mode["baseline"][doc_id]["prediction"]
        topic_pred = by_mode["ce-topic-query-shortlist"][doc_id]["prediction"]
        title_pred = by_mode["ce-title-shortlist"][doc_id]["prediction"]

        topic_order = sorted(
            range(len(panel.chunks)),
            key=lambda i: (-panel.topic_ce_scores[i], i),
        )
        topic_rank = {index: rank + 1 for rank, index in enumerate(topic_order)}
        title_order = sorted(
            range(len(panel.chunks)),
            key=lambda i: (-panel.title_ce_scores[i], i),
        )
        title_rank = {index: rank + 1 for rank, index in enumerate(title_order)}

        all_means = aggregator_means(
            models.sentence_encoder, [answer.text for answer in panel.answers]
        )
        kept = shortlist_answers(panel.answers, panel.topic_ce_scores, K)
        kept_ids = {id(answer) for answer in kept}
        kept_means_by_id = {}
        kept_means = aggregator_means(
            models.sentence_encoder, [answer.text for answer in kept]
        )
        for answer, mean in zip(kept, kept_means):
            kept_means_by_id[id(answer)] = mean

        chunks = []
        best_gold_index = max(
            range(len(panel.answers)),
            key=lambda i: char_f1(panel.answers[i].text, gold),
        )
        baseline_indices = [
            i
            for i, answer in enumerate(panel.answers)
            if answer.text.strip() == baseline_pred.strip()
        ]
        topic_pred_indices = [
            i
            for i, answer in enumerate(panel.answers)
            if answer.text.strip() == topic_pred.strip()
        ]

        for index, (chunk, answer) in enumerate(zip(panel.chunks, panel.answers)):
            in_shortlist = topic_rank[index] <= K
            chunks.append(
                {
                    "index": index,
                    "topic_rank": topic_rank[index],
                    "title_rank": title_rank[index],
                    "topic_ce": panel.topic_ce_scores[index],
                    "title_ce": panel.title_ce_scores[index],
                    "qa_answer": answer.text,
                    "qa_confidence": answer.confidence,
                    "char_f1_gold": char_f1(answer.text, gold),
                    "is_baseline_span": answer.text.strip() == baseline_pred.strip(),
                    "is_topic_pred_span": answer.text.strip() == topic_pred.strip(),
                    "in_topic_top16": in_shortlist,
                    "agg_mean_all": all_means[index],
                    "agg_mean_shortlist": kept_means_by_id.get(id(answer)),
                    "start_sentence": chunk.start_sentence,
                    "snippet": chunk.text[:280].replace("\n", " "),
                }
            )

        mechanism = "unknown"
        baseline_in_shortlist = any(
            topic_rank[i] <= K for i in baseline_indices
        ) if baseline_indices else False
        gold_in_shortlist = topic_rank[best_gold_index] <= K
        if not baseline_indices:
            mechanism = "baseline_span_not_reproduced"
        elif not baseline_in_shortlist:
            mechanism = "correct_chunk_dropped_by_ce"
        elif gold_in_shortlist and topic_pred.strip() != baseline_pred.strip():
            mechanism = "aggregator_flipped_inside_shortlist"
        reports.append(
            {
                "doc_id": doc_id,
                "gold": gold,
                "leaf_id": leaf_id,
                "topic_terms": list(terms),
                "title_query": panel.title_query,
                "topic_query": panel.topic_query,
                "n_chunks": len(panel.chunks),
                "baseline_pred": baseline_pred,
                "title_pred": title_pred,
                "topic_pred": topic_pred,
                "baseline_char_f1": float(by_mode["baseline"][doc_id]["char_f1"]),
                "topic_char_f1": float(
                    by_mode["ce-topic-query-shortlist"][doc_id]["char_f1"]
                ),
                "title_char_f1": float(by_mode["ce-title-shortlist"][doc_id]["char_f1"]),
                "mechanism": mechanism,
                "baseline_chunk_indices": baseline_indices,
                "baseline_topic_ranks": [topic_rank[i] for i in baseline_indices],
                "topic_pred_chunk_indices": topic_pred_indices,
                "topic_pred_topic_ranks": [topic_rank[i] for i in topic_pred_indices],
                "best_gold_chunk_index": best_gold_index,
                "best_gold_topic_rank": topic_rank[best_gold_index],
                "best_gold_answer": panel.answers[best_gold_index].text,
                "best_gold_char_f1": char_f1(
                    panel.answers[best_gold_index].text, gold
                ),
                "chunks": chunks,
            }
        )
        print(f"done {doc_id[:12]} {mechanism}", flush=True)

    out = ROOT / "logs" / "topic_harm_chunk_trace.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
