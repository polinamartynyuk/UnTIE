"""Blind prediction implementations for all predeclared real systems."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .contract import CELL_SPECS, PROPOSED
from .engine import CellEngine, LoadedCrossEncoder, runtime_as_unlabeled
from .generative import QwenBaseline
from .io_guard import read_json, read_jsonl, sha256, write_json, write_jsonl
from .modernbert import ModernBertReader
from .proposed import apply_runtime, load_policy


def _prediction(cell, system, item, selected_chunk_id, answer, *, seed, extra=None):
    metadata = dict(extra or {}); final_answer = str(answer or "")
    metadata.setdefault("candidate_chunk_ids", [])
    metadata.setdefault("answer_confidence", metadata.get("qa_confidence"))
    metadata.setdefault("fallback_reason", "EMPTY_ANSWER_RETAINED" if not final_answer.strip() else "NONE")
    metadata.setdefault("failure_reason", None)
    return {"runtime": {
        "dataset_cell": cell, "dataset": CELL_SPECS[cell].family, "split": "TEST",
        "corpus_family": CELL_SPECS[cell].family, "language": CELL_SPECS[cell].language,
        "system_id": system, "document_id": item.document_id, "canonical_source_document_id": item.document_id,
        "question_id": item.question_id, "aspect": item.aspect, "question": item.question,
        "evaluation_eligible": "PENDING_CANONICAL_ANSWERABLE_GOLD_SIDE_RULE",
        "selected_chunk_id": int(selected_chunk_id) if selected_chunk_id is not None else None,
        "final_answer": final_answer, "model_seed": seed, **metadata,
    }}


def _row_by_id(rows): return {int(x["chunk_id"]): x for x in rows}

def _confidence(value):
    return float(value.get("confidence", value.get("qa_confidence", 0.0))) if str(value.get("answer", value.get("qa_answer", ""))).strip() else -1e30


def baseline_predictions(cell, items, engine: CellEngine, feature_runtime, *, bge=None, qwen=None):
    grouped = defaultdict(list)
    for row in feature_runtime: grouped[str(row["question_id"])].append(row)
    outputs = {x: [] for x in ("B1", "B2", "B3", "B4a", "B4b", "G1")}
    for index, source in enumerate(items, 1):
        item = runtime_as_unlabeled(source); rows = sorted(grouped[item.question_id], key=lambda x: (x["b0_rank"], x["chunk_id"])); byid = _row_by_id(rows)
        top = rows[0]
        outputs["B2"].append(_prediction(cell, "B2", item, top["chunk_id"], top["qa_answer"], seed=0, extra={"ranker": "canonical_B0", "qa_confidence": top["qa_confidence"], "candidate_chunk_ids": [int(top["chunk_id"])]}))
        best = max(rows, key=lambda x: (_confidence(x), -int(x["b0_rank"]), -int(x["chunk_id"])))
        outputs["B3"].append(_prediction(cell, "B3", item, best["chunk_id"], best["qa_answer"], seed=0, extra={"ranker": "canonical_B0", "candidate_k": len(rows), "qa_confidence": best["qa_confidence"], "candidate_chunk_ids": [int(x["chunk_id"]) for x in rows]}))
        all_qa = [(chunk, engine.qa(item, chunk.chunk_id, item.question)) for chunk in item.chunks if chunk.eligible_for_qa]
        chosen_chunk, chosen_qa = max(all_qa, key=lambda x: (_confidence(x[1]), -int(x[0].chunk_id)))
        outputs["B1"].append(_prediction(cell, "B1", item, chosen_chunk.chunk_id, chosen_qa["answer"], seed=0, extra={"candidate_k": len(all_qa), "qa_confidence": chosen_qa["confidence"], "candidate_chunk_ids": [int(x[0].chunk_id) for x in all_qa]}))
        if bge is not None:
            scores = bge.predict([(item.question, chunk.text) for chunk in item.chunks]); ranked = sorted(zip(item.chunks, scores), key=lambda x: (-float(x[1]), int(x[0].chunk_id)))
            bge_top = ranked[: min(CELL_SPECS[cell].k, len(ranked))]; qas = [(chunk, score, engine.qa(item, chunk.chunk_id, item.question)) for chunk, score in bge_top]
            first = qas[0]
            outputs["B4a"].append(_prediction(cell, "B4a", item, first[0].chunk_id, first[2]["answer"], seed=0, extra={"bge_score": float(first[1]), "qa_confidence": first[2]["confidence"], "candidate_chunk_ids": [int(first[0].chunk_id)]}))
            bbest = max(qas, key=lambda x: (_confidence(x[2]), -int(x[0].chunk_id)))
            outputs["B4b"].append(_prediction(cell, "B4b", item, bbest[0].chunk_id, bbest[2]["answer"], seed=0, extra={"bge_score": float(bbest[1]), "qa_confidence": bbest[2]["confidence"], "candidate_k": len(qas), "candidate_chunk_ids": [int(x[0].chunk_id) for x in qas]}))
        if qwen is not None:
            try:
                answer = qwen.answer(item, CELL_SPECS[cell].language); failure = None
            except Exception as exc:
                answer = {"answer": "", "prompt_tokens": 0, "output_tokens": 0}; failure = f"{type(exc).__name__}: {exc}"
            outputs["G1"].append(_prediction(cell, "G1", item, None, answer["answer"], seed=0, extra={"context_regime": "full_document", "failure_reason": failure, **answer}))
        engine.progress.count(index, len(items), item=f"{cell} blind baseline predictions")
    return outputs


def proposed_predictions(cell, items, runtime, policy_root: Path, seeds):
    by_item = {str(x.question_id): runtime_as_unlabeled(x) for x in items}; by_rows = defaultdict(list)
    for row in runtime: by_rows[str(row["question_id"])].append(row)
    output = defaultdict(list)
    systems = ("P1", "P2", "P3", "P4", "P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK")
    for seed in seeds:
        for system in systems:
            policy = load_policy(policy_root / system / f"seed_{seed}")
            selected = apply_runtime(runtime, policy, use_harm=system != "P2-NO-RISK")
            for qid, decision in selected.items():
                rows = _row_by_id(by_rows[qid]); cid = int(decision["selected_chunk_id"]); row = rows[cid]
                output[system].append(_prediction(cell, system, by_item[qid], cid, row["qa_answer"], seed=seed,
                    extra={"candidate_k": len(rows), "policy": decision, "qa_confidence": row["qa_confidence"], "candidate_chunk_ids": sorted(rows)}))
    return output


def modernbert_predictions(cell, items, engine, b0_scorer, model_root: Path, seeds):
    output = []
    for seed in seeds:
        reader = ModernBertReader(model_root / f"seed_{seed}", engine.device)
        for item_runtime in items:
            item = runtime_as_unlabeled(item_runtime); scores = engine.scores(b0_scorer, item)
            top = sorted(item.chunks, key=lambda x: (-scores[x.chunk_id], x.chunk_id))[: engine.spec.k]
            candidates = [(chunk, reader.answer(item.question, chunk.text)) for chunk in top]
            chunk, answer = max(candidates, key=lambda x: (_confidence(x[1]), -int(x[0].chunk_id)))
            output.append(_prediction(cell, "B5-EN", item, chunk.chunk_id, answer["answer"], seed=seed,
                                      extra={"candidate_k": len(top), "qa_confidence": answer["confidence"], "candidate_chunk_ids": [int(x.chunk_id) for x in top], "failure_reason": answer.get("error")}))
    return output


def _available(capabilities, cell, system):
    row = next(x for x in capabilities["rows"] if x["cell"] == cell and x["system"] == system)
    return row["status"] == "RUN"


def run_blind_cell(config, cell, items, engine, b0_scorer, runtime, progress):
    capabilities = read_json(config.run_dir / "manifests/system_capability_matrix.json")
    provenance = read_json(config.run_dir / "manifests/model_provenance.json")
    bge = None
    if _available(capabilities, cell, "B4a"):
        b = provenance["BGE"]; bge = LoadedCrossEncoder(b.get("resolved_path") or b["model_id"], engine.device, local_files_only=not config.allow_downloads, revision=None if b.get("resolved_path") else b.get("resolved_revision"))
    qwen = None
    if _available(capabilities, cell, "G1"):
        q = provenance["QWEN"]; qwen = QwenBaseline(q.get("resolved_path") or q["model_id"], None if q.get("resolved_path") else q.get("resolved_revision"), engine.device, config.allow_downloads, progress=progress)
    progress.log("starting blind baseline predictions", cell=cell, questions=len(items))
    outputs = baseline_predictions(cell, items, engine, runtime, bge=bge, qwen=qwen)
    outputs.update(proposed_predictions(cell, items, runtime, config.run_dir / "models" / cell / "proposed", config.seeds))
    if CELL_SPECS[cell].language == "en" and _available(capabilities, cell, "B5-EN"):
        outputs["B5-EN"] = modernbert_predictions(cell, items, engine, b0_scorer, config.run_dir / "models" / cell / "B5-EN", config.seeds)
    hashes = {}
    stochastic = {"B5-EN", "P1", "P2", "P3", "P4", "P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK"}
    for system, rows in outputs.items():
        if not rows or (system not in {"P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK"} and not _available(capabilities, cell, system)):
            continue
        expected = len(items) * (len(config.seeds) if system in stochastic else 1)
        keys = [(str(x["runtime"]["question_id"]), int(x["runtime"]["model_seed"])) for x in rows]
        if len(rows) != expected or len(set(keys)) != expected:
            raise RuntimeError(f"{cell}/{system}: blind coverage {len(rows)} unique={len(set(keys))} expected={expected}")
        path = config.run_dir / "test_predictions" / system / f"{cell}.jsonl"
        if config.resume and path.is_file():
            existing = read_jsonl(path)
            existing_keys = {(str(x["runtime"]["question_id"]), int(x["runtime"]["model_seed"])) for x in existing}
            if len(existing) != expected or existing_keys != set(keys):
                raise RuntimeError(f"{cell}/{system}: existing immutable prediction coverage is invalid")
            progress.log("preserving existing immutable predictions", cell=cell, system=system, rows=len(existing))
            hashes[system] = {"rows": len(existing), "sha256": sha256(path)}
            continue
        write_jsonl(path, rows, immutable=True); hashes[system] = {"rows": len(rows), "sha256": sha256(path)}
    required = [x["system"] for x in capabilities["rows"] if x["cell"] == cell and x["status"] == "RUN"]
    missing = sorted(set(required) - set(hashes))
    if missing:
        raise RuntimeError(f"{cell}: blind prediction systems incomplete: {missing}")
    write_json(config.run_dir / "test_predictions" / f"{cell}_COMPLETE.json", {"cell": cell, "systems": hashes, "required": required}, immutable=True)
    return hashes
