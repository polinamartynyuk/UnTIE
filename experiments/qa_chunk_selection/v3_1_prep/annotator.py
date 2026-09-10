"""Annotator pack export and optional coverage-completion sampling.

Reads the frozen V3 queue / B0 OOF. Never writes back into V3 artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection.schema import InstanceRecord
from experiments.qa_chunk_selection.v3.overlay import empty_overlay_row

from .context import local_context
from .contract import ANNOTATION_VERSION, FORBIDDEN_ANNOTATOR_FIELDS, GUIDELINE_VERSION, TAXONOMY_VERSION
from .taxonomy import assert_no_model_scores, empty_fill_fields, overlay_key, strip_model_scores, taxonomy_payload

GUIDELINE_STEPS = [
    "1. Identify the aspect (task / method / material / metric).",
    "2. Find the aspect value in the chunk (if none: often partial_support / unresolved).",
    "3. Does the value belong to this paper or to another object? Other object → target_mismatch.",
    "4. If it belongs to this paper: sufficient → target_evidence; insufficient → partial_support.",
    "5. Fill v3_gold_relation against the listed gold_answers.",
    "6. If mismatch, fill v3_mismatch_type.",
    "7. Write a short checkable v3_annotation_reason.",
    "8. If substantial uncertainty remains → unresolved. Prefer unresolved over a wrong label.",
]


def _instance_index(records: Sequence[InstanceRecord]) -> dict[str, InstanceRecord]:
    return {item.question_id: item for item in records}


def metadata_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {overlay_key(row): dict(row) for row in rows}


def build_annotator_item(
    item: InstanceRecord,
    chunk_id: int,
    *,
    stratum: str | None,
    sampling_probability: float | None,
    annotator_id: str,
) -> dict[str, Any]:
    chunk = item.chunk_by_id(int(chunk_id))
    row = empty_overlay_row(item, chunk)
    row.update(empty_fill_fields())
    row.update(local_context(item, chunk))
    row.update(
        {
            "question": item.question,
            "chunk_text": chunk.text,
            "gold_answers": list(item.gold_answers),
            "source_sentence": chunk.text,
            "annotation_sampling_stratum": stratum,
            "annotation_sampling_frame": "v3_unlocked_answerable_ambiguous",
            "sampling_probability": sampling_probability,
            "annotator_id": annotator_id,
            "annotation_guideline_version": GUIDELINE_VERSION,
            "annotation_version": ANNOTATION_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "context_used": "chunk+section+neighbor_sentences",
        }
    )
    clean = strip_model_scores(row)
    assert_no_model_scores(clean)
    return clean


def export_annotator_queue(
    *,
    records: Sequence[InstanceRecord],
    queue: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    annotator_id: str,
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    by_q = _instance_index(records)
    meta = metadata_index(metadata)
    items = []
    log = progress or ProgressLogger()
    total = len(queue)
    for index, raw in enumerate(queue, start=1):
        log.count(index, total, item="annotator_export")
        question_id = str(raw["question_id"])
        chunk_id = int(raw["chunk_id"])
        item = by_q.get(question_id)
        if item is None:
            raise KeyError(f"queue question_id not in unlocked V3 universe: {question_id}")
        meta_row = meta.get((question_id, chunk_id), {})
        packed = build_annotator_item(
            item,
            chunk_id,
            stratum=str(raw.get("annotation_sampling_stratum") or meta_row.get("annotation_sampling_stratum") or "hard_case"),
            sampling_probability=meta_row.get("sampling_probability"),
            annotator_id=annotator_id,
        )
        packed["annotation_sampling_stratum"] = packed.get("annotation_sampling_stratum") or meta_row.get(
            "annotation_sampling_stratum"
        )
        items.append(packed)
    hidden = [field for field in FORBIDDEN_ANNOTATOR_FIELDS if any(field in row for row in items)]
    if hidden:
        raise ValueError(f"export leaked model fields into annotator pack: {hidden}")
    return {
        "n": len(items),
        "taxonomy": taxonomy_payload(),
        "guideline": GUIDELINE_STEPS,
        "queue": items,
    }


def coverage_completion_candidates(
    *,
    records: Sequence[InstanceRecord],
    b0_rows: Sequence[Mapping[str, Any]],
    already: set[tuple[str, int]],
    k: int,
    limit: int,
    annotator_id: str,
    progress: ProgressLogger | None = None,
) -> list[dict[str, Any]]:
    """Extra ambiguous chunks from frozen B0 top-K. Tagged coverage_completion."""
    by_q = _instance_index(records)
    ranked: list[tuple[int, str, int, dict[str, Any]]] = []
    log = progress or ProgressLogger()
    total = len(b0_rows)
    for index, row in enumerate(b0_rows, start=1):
        log.count(index, total, item="coverage_scan")
        item = by_q.get(str(row["question_id"]))
        if item is None:
            continue
        ids = [int(cid) for cid in (row.get("ranked_ids") or [])]
        for rank, chunk_id in enumerate(ids[: max(int(k), 1)], start=1):
            key = (item.question_id, int(chunk_id))
            if key in already:
                continue
            chunk = item.chunk_by_id(int(chunk_id))
            if chunk.label != "ambiguous":
                continue
            priority = 0 if rank == 1 else (1 if rank <= 5 else 2)
            packed = build_annotator_item(
                item,
                chunk.chunk_id,
                stratum="coverage_completion",
                sampling_probability=None,
                annotator_id=annotator_id,
            )
            packed["coverage_rank"] = rank
            ranked.append((priority, item.question_id, int(chunk_id), packed))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    selected = []
    seen: set[tuple[str, int]] = set()
    for _priority, question_id, chunk_id, packed in ranked:
        key = (question_id, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        selected.append(packed)
        if len(selected) >= int(limit):
            break
    return selected


def annotator_html(queue: Sequence[Mapping[str, Any]]) -> str:
    """Self-contained helper page. No model scores. Download filled JSONL locally."""
    import json

    payload = json.dumps(list(queue), ensure_ascii=False)
    steps = "\n".join(f"<li>{step}</li>" for step in GUIDELINE_STEPS)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>V3.1 target-status annotation</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: 24px auto; line-height: 1.4; }}
pre, textarea {{ width: 100%; white-space: pre-wrap; }}
.chunk {{ background: #f6f6f6; padding: 12px; }}
label {{ display: block; margin-top: 8px; }}
</style></head><body>
<h1>V3.1 target-status overlay</h1>
<p>Fill <code>v3_target_status</code> and related fields. Model scores are not shown. Do not rewrite <code>source_label</code>.</p>
<ol>{steps}</ol>
<p>Item <span id="idx">1</span> / <span id="n"></span></p>
<div id="card"></div>
<p>
<button type="button" id="prev">Previous</button>
<button type="button" id="next">Next</button>
<button type="button" id="save">Download filled JSONL</button>
</p>
<script>
const ITEMS = {payload};
let i = 0;
const FIELDS = ["v3_target_status","v3_target_value","v3_gold_relation","v3_mismatch_type","v3_owner_text","v3_support_scope","v3_annotation_confidence","v3_annotation_reason","target_entity_or_value","mismatch_entity_or_value"];
function render() {{
  const row = ITEMS[i];
  document.getElementById("idx").textContent = String(i+1);
  document.getElementById("n").textContent = String(ITEMS.length);
  const inputs = FIELDS.map(name => {{
    const val = row[name] == null ? "" : String(row[name]);
    return `<label>${{name}}<br><input data-field="${{name}}" value="${{val.replaceAll('"','&quot;')}}" style="width:100%"></label>`;
  }}).join("");
  document.getElementById("card").innerHTML = `
    <p><b>${{row.aspect}}</b> · ${{row.question_id}} · chunk ${{row.chunk_id}} · stratum ${{row.annotation_sampling_stratum}} · subtype ${{row.source_ambiguous_subtype}}</p>
    <p><b>Question</b><br>${{row.question}}</p>
    <p><b>Gold answers (for gold_relation only)</b><br>${{(row.gold_answers||[]).join(" | ")}}</p>
    <p><b>Section</b> ${{row.section_title || "(none)"}}</p>
    <p><b>Previous sentence</b><br>${{row.previous_sentence || ""}}</p>
    <pre class="chunk">${{row.chunk_text}}</pre>
    <p><b>Next sentence</b><br>${{row.next_sentence || ""}}</p>
    ${{inputs}}
  `;
  document.querySelectorAll("[data-field]").forEach(el => {{
    el.addEventListener("input", () => {{ ITEMS[i][el.dataset.field] = el.value || null; }});
  }});
}}
document.getElementById("prev").onclick = () => {{ i = (i + ITEMS.length - 1) % ITEMS.length; render(); }};
document.getElementById("next").onclick = () => {{ i = (i + 1) % ITEMS.length; render(); }};
document.getElementById("save").onclick = () => {{
  const blob = new Blob(ITEMS.map(row => JSON.stringify(row) + "\\n"), {{type: "application/jsonl"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations_filled.jsonl";
  a.click();
}};
render();
</script></body></html>
"""
