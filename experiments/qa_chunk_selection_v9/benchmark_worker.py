"""One fresh process per V9 cell/system."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np

from experiments.qa_chunk_selection.answerability import AnswerabilityModel
from experiments.qa_chunk_selection.v2.schemas import to_runtime
from experiments.qa_chunk_selection.v3.data import records_from_jsonl
from experiments.qa_chunk_selection.v3.train import score_fn_for
from experiments.qa_chunk_selection.v5_question_conditioned_residual.contract import CELLS
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language
from .io_guard import read_jsonl


def _sync(torch, device) -> None:
    if torch is not None and device is not None:
        torch.cuda.synchronize(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--system", choices=["S0", "S1", "S2"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--v5-run", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    spec = CELLS[args.cell]
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
    device = choose_device(args.device)[0]
    try:
        import torch
        cuda = torch.device(device) if str(device).startswith("cuda") and torch.cuda.is_available() else None
        if cuda is not None:
            torch.cuda.reset_peak_memory_stats(cuda)
    except ImportError:
        torch, cuda = None, None
    v5_seed = args.v5_run / args.cell / "outer_fold_0/seed_42"
    ranker = AnswerabilityModel(str(v5_seed / "eval/outer_model/model"), device, 512)
    factory = ModelFactory(PipelineConfig(profile=profile_for_language(spec.language), device=device))
    answerer = factory.answerer
    wanted = json.loads(args.sample_manifest.read_text())["cells"][args.cell]["question_ids"]
    items = {str(x.question_id): x for x in records_from_jsonl(spec.records)}
    records = [items[x] for x in wanted]
    question_policy = json.loads((v5_seed / "eval/question_policy.json").read_text())
    q5 = json.loads(spec.questions_q5.read_text()) if spec.questions_q5 else {}
    policy, columns = None, None
    if args.system != "S0":
        model_root = args.run_root / "models" / args.system / args.cell / "outer_fold_0/seed_42"
        policy = {
            "utility": joblib.load(model_root / "utility_model.joblib"),
            "harm": joblib.load(model_root / "harm_model.joblib"),
        }
        columns = json.loads((model_root / "selection.json").read_text())["feature_columns"]
    feature_path = args.run_root / "cache/canonical_features" / args.cell / "outer_fold_0/seed_42/eval_hrr_targets.jsonl"
    cached = read_jsonl(feature_path)
    by_question = defaultdict(list)
    for row in cached:
        by_question[str(row["question_id"])].append(row)

    def request(item):
        runtime = to_runtime(item)
        scores = {int(c.chunk_id): float(v) for c, v in zip(runtime.chunks, score_fn_for(ranker)(runtime))}
        top = sorted(scores, key=lambda cid: (-scores[cid], cid))[:min(spec.k, len(scores))]
        variant = question_policy.get(item.aspect)
        question = item.question if variant is None else q5[item.aspect][int(variant)]
        ids = top[:1] if args.system == "S0" else top
        answers = {cid: str(answerer(question=question, context=item.chunk_by_id(cid).text).get("answer") or "") for cid in ids}
        if args.system != "S0":
            ranker.predict([(question, answers[cid]) for cid in ids])
        if policy and by_question.get(str(item.question_id)):
            import pandas as pd
            frame = pd.DataFrame(by_question[str(item.question_id)])[columns]
            policy["utility"].predict(frame)
            policy["harm"].predict_proba(frame)
        return len(ids), len(ids) if args.system != "S0" else 0

    warm = (records * ((args.warmup + len(records) - 1) // max(len(records), 1)))[:args.warmup]
    for item in warm:
        request(item)
    latencies, qa, qa_ranker, repeats = [], [], [], []
    per_question_counts = {}
    for repeat in range(args.repeats):
        current = []
        for item in records:
            _sync(torch, cuda)
            started = time.perf_counter()
            q, r = request(item)
            _sync(torch, cuda)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            current.append(elapsed)
            qa.append(q)
            qa_ranker.append(r)
            if repeat == 0:
                per_question_counts[str(item.question_id)] = {"qa_forwards": q, "q_a_ranker_forwards": r}
        repeats.append({"repeat": repeat, "median_seconds": float(np.median(current)), "questions": len(current)})
    parameters = sum(x.numel() for x in ranker.model.parameters()) + sum(x.numel() for x in factory.qa_model.parameters())
    meta_parameters = 0
    if policy:
        for model in policy.values():
            if hasattr(model, "named_steps"):
                estimator = model.named_steps["model"]
                meta_parameters += int(np.size(estimator.coef_))
                meta_parameters += int(np.size(getattr(estimator, "intercept_", [])))
    disk = sum(x.stat().st_size for root in (v5_seed / "eval/outer_model/model", spec.qa_model) for x in root.rglob("*") if x.is_file())
    result = {
        "cell": args.cell, "system": args.system, "questions": len(records),
        "warmup": args.warmup, "repeats": args.repeats, "sample_question_ids": wanted,
        "median_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
        "throughput_questions_per_second": float(len(latencies) / sum(latencies)),
        "mean_qa_forwards": float(np.mean(qa)), "max_qa_forwards": int(max(qa, default=1)),
        "mean_q_a_ranker_forwards": float(np.mean(qa_ranker)),
        "per_question_forward_counts": per_question_counts,
        "peak_gpu_vram_bytes": int(torch.cuda.max_memory_allocated(cuda)) if cuda is not None else None,
        "peak_cpu_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "loaded_parameter_count": int(parameters), "trainable_meta_parameter_count": meta_parameters,
        "weights_disk_bytes": int(disk),
        "device": device, "fresh_process": True, "warmup_excluded": True,
        "neural_inference_fresh": True, "policy_feature_values_from_frozen_v9_cache": bool(policy),
        "repeat_summary": repeats,
    }
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

