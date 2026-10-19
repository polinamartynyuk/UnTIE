"""Fresh-process end-to-end inference resource measurement."""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from experiments.qa_chunk_selection.progress import ProgressLogger

from .config import Config
from .contract import CELL_SPECS
from .data import load_test_runtime
from .engine import CellEngine, LoadedCrossEncoder, runtime_as_unlabeled
from .generative import QwenBaseline
from .io_guard import read_json, write_json
from .modernbert import ModernBertReader
from .proposed import apply_runtime, load_policy


def _sync():
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except ImportError: pass


def _parameters(*objects):
    seen, total = set(), 0
    for obj in objects:
        model = getattr(obj, "model", None)
        if model is None: continue
        for parameter in model.parameters():
            if id(parameter) not in seen: seen.add(id(parameter)); total += parameter.numel()
    return int(total)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--cell", required=True); parser.add_argument("--system", required=True); parser.add_argument("--questions", type=int, default=100); parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv); manifest = read_json(args.run_dir / "manifest.json"); raw = manifest["config"]
    config = Config(run_dir=args.run_dir, cells=(args.cell,), seeds=tuple(raw["seeds"]), device=args.device, resume=True, unlock_test=True, allow_downloads=raw["allow_downloads"])
    progress = ProgressLogger(args.run_dir / "resources" / f"{args.cell}_{args.system}.progress.jsonl")
    engine = CellEngine(config, args.cell, progress)
    selection = read_json(args.run_dir / "development" / args.cell / "frozen_selection.json")
    engine.select_questions((), frozen=selection["question_mapping"])
    b0 = engine.load_b0(args.run_dir / "models" / args.cell / "B0_final") if args.system in {"B2", "B3", "B5-EN", "P2"} else None
    items = list(engine.apply_questions(load_test_runtime(args.cell)))[: args.questions]; extra = []
    bge = qwen = modern = policy = None
    provenance = read_json(args.run_dir / "manifests/model_provenance.json")
    if args.system == "B4b": b = provenance["BGE"]; bge = LoadedCrossEncoder(b.get("resolved_path") or b["model_id"], engine.device, local_files_only=not config.allow_downloads, revision=None if b.get("resolved_path") else b.get("resolved_revision")); extra.append(bge)
    if args.system == "G1": q = provenance["QWEN"]; qwen = QwenBaseline(q.get("resolved_path") or q["model_id"], None if q.get("resolved_path") else q.get("resolved_revision"), engine.device, config.allow_downloads, progress=progress); extra.append(qwen)
    if args.system == "B5-EN": modern = ModernBertReader(args.run_dir / "models" / args.cell / "B5-EN/seed_42", engine.device); extra.append(modern)
    if args.system == "P2": policy = load_policy(args.run_dir / "models" / args.cell / "proposed/P2/seed_42")

    def infer(source):
        item = runtime_as_unlabeled(source)
        if args.system == "B1":
            values = [engine.qa(item, x.chunk_id, item.question) for x in item.chunks]; return max(values, key=lambda x: x["confidence"])["answer"]
        if args.system == "G1": return qwen.answer(item, CELL_SPECS[args.cell].language)["answer"]
        if args.system == "B4b":
            scores = bge.predict([(item.question, x.text) for x in item.chunks]); top = sorted(zip(item.chunks, scores), key=lambda x: (-x[1], x[0].chunk_id))[: CELL_SPECS[args.cell].k]
            values = [engine.qa(item, x.chunk_id, item.question) for x, _ in top]; return max(values, key=lambda x: x["confidence"])["answer"]
        scores = engine.scores(b0, item); top = sorted(item.chunks, key=lambda x: (-scores[x.chunk_id], x.chunk_id))[: CELL_SPECS[args.cell].k]
        if args.system == "B2": return engine.qa(item, top[0].chunk_id, item.question)["answer"]
        if args.system == "B5-EN":
            values = [modern.answer(item.question, x.text) for x in top]; return max(values, key=lambda x: x["confidence"])["answer"]
        if args.system == "B3":
            values = [engine.qa(item, x.chunk_id, item.question) for x in top]; return max(values, key=lambda x: x["confidence"])["answer"]
        runtime, _, _ = engine.materialize([source], b0, 42, role="resource", with_targets=False)
        selected = apply_runtime(runtime, policy)[item.question_id]["selected_chunk_id"]
        return next(x["qa_answer"] for x in runtime if int(x["chunk_id"]) == int(selected))

    warmup = min(10, len(items));
    for item in items[:warmup]: infer(item)
    _sync()
    samples = []
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except ImportError: pass
    for repeat in range(3):
        for item in items:
            start = time.perf_counter(); infer(item); _sync(); samples.append(time.perf_counter() - start)
    peak_vram = None
    try:
        import torch
        if torch.cuda.is_available(): peak_vram = int(torch.cuda.max_memory_allocated())
    except ImportError: pass
    avg_chunks = statistics.mean(len(x.chunks) for x in items); k = CELL_SPECS[args.cell].k
    forward_counts = {
        "B1": {"qa": avg_chunks, "reranker": 0, "q_answer_reranker": 0, "generator": 0},
        "B2": {"qa": 1, "reranker": avg_chunks, "q_answer_reranker": 0, "generator": 0},
        "B3": {"qa": k, "reranker": avg_chunks, "q_answer_reranker": 0, "generator": 0},
        "B4b": {"qa": k, "reranker": avg_chunks, "q_answer_reranker": 0, "generator": 0},
        "B5-EN": {"qa": k, "reranker": avg_chunks, "q_answer_reranker": 0, "generator": 0},
        "G1": {"qa": 0, "reranker": 0, "q_answer_reranker": 0, "generator": 1},
        "P2": {"qa": k, "reranker": avg_chunks, "q_answer_reranker": k, "generator": 0},
    }[args.system]
    result = {"cell": args.cell, "system": args.system, "questions": len(items), "warmup_questions": warmup, "repeats": 3,
              "median_latency_s": statistics.median(samples), "p95_latency_s": sorted(samples)[max(0, int(.95 * len(samples)) - 1)],
              "throughput_questions_s": len(samples) / sum(samples), "peak_vram_bytes": peak_vram,
              "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "loaded_neural_parameters": _parameters(b0, *extra, *((engine.reader.answerer,) if engine.reader.answerer is not None else ())),
              "trainable_parameters_at_inference": 0, "mean_neural_forward_counts": forward_counts,
              "scope": "question plus preprocessed canonical chunks to final answer; model loading excluded"}
    write_json(args.run_dir / "resources" / args.system / f"{args.cell}.json", result); print(json.dumps(result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
