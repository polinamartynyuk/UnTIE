"""Measured ranker-only and real QA-count end-to-end resource benchmark."""

from pathlib import Path
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import time

from experiments.qa_chunk_selection.io_util import write_json
from experiments.qa_chunk_selection.v4.shortlist import top_k_ids
from experiments.qa_chunk_selection.v4.v6_reader import agreement_features
from experiments.qa_chunk_selection.v4_4_ru.encoder import build_encoder
from experiments.qa_chunk_selection.v4_3.encoder import l2_normalize
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl
from experiments.qa_chunk_selection.v4_4_ru_ablation.features import adaptive_semantic_pack, routing_available
from experiments.qa_chunk_selection.v4_1.ranking import v6_fused_score
from experiments.qa_chunk_selection.v4_3.ranking import isolate_with_b0_tiebreak
from experiments.qa_chunk_selection.v4_4_ru_ranker_factorial.cross_encoder import FrozenRevisionCrossEncoder
from untie.config import PipelineConfig
from untie.models import ModelFactory, profile_for_language

from .contract import RANKERS, SOURCE_OVERLAP


def _sync(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _reset_peak(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return None


def _summary(values):
    ordered = sorted(float(value) for value in values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"n": len(ordered), "median_seconds": statistics.median(ordered),
            "mean_seconds": statistics.mean(ordered),
            "std_seconds": statistics.pstdev(ordered), "p95_seconds": p95,
            "raw_seconds": ordered}


def _environment(device):
    payload = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "device": str(device), "precision": "model/runtime dependent; recorded per component",
        "matched_batch_policy": "prediction batch 8 for both rankers",
        "best_feasible_policy": "model-specific frozen batch from config",
    }
    try:
        import torch
        payload.update(torch=torch.__version__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(torch.device(device))
            payload["gpu_total_bytes"] = torch.cuda.get_device_properties(torch.device(device)).total_memory
            try:
                payload["nvidia_driver"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True, timeout=10).splitlines()[0].strip()
            except Exception:
                payload["nvidia_driver"] = None
    except ImportError:
        payload["torch"] = None
    try:
        import transformers
        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["transformers"] = None
    try:
        import psutil
        payload["ram_total_bytes"] = psutil.virtual_memory().total
    except ImportError:
        payload["ram_total_bytes"] = None
    return payload


def _sample_pairs(state, limit):
    records = sorted(state.records, key=lambda item: str(item.question_id))[:limit]
    pairs = [(item.question, chunk.text) for item in records for chunk in item.chunks]
    lengths = [len(question.split()) + len(chunk.split()) for question, chunk in pairs]
    return records, pairs, lengths


def _ranker_benchmark(spec, device, pairs, *, batch_size, warmup, repetitions, local_files_only):
    scorer = FrozenRevisionCrossEncoder(spec, device, 512, batch_size, local_files_only=local_files_only)
    params = sum(parameter.numel() for parameter in scorer.model.parameters())
    weights = sum(parameter.numel() * parameter.element_size() for parameter in scorer.model.parameters())
    token_lengths = [len(scorer.tokenizer(question, passage, truncation=True, max_length=512)["input_ids"])
                     for question, passage in pairs]
    sorted_lengths = sorted(token_lengths)
    warm = pairs[:max(1, min(batch_size, len(pairs)))]
    for _ in range(warmup):
        scorer.predict(warm)
    _reset_peak(device)
    values = []
    for _ in range(repetitions):
        _sync(device)
        started = time.perf_counter()
        scorer.predict(pairs)
        _sync(device)
        values.append(time.perf_counter() - started)
    result = _summary(values)
    result.update({
        "pairs": len(pairs), "batch_size": batch_size,
        "throughput_pairs_per_second": len(pairs) / result["median_seconds"],
        "parameter_count": int(params), "weights_bytes": int(weights),
        "peak_gpu_allocated_bytes": _peak(device),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "model": spec.model_name, "revision": spec.revision,
        "sequence_length_tokens": {"min": min(sorted_lengths), "median": statistics.median(sorted_lengths),
                                   "p95": sorted_lengths[min(len(sorted_lengths)-1, int(round(.95*(len(sorted_lengths)-1))))],
                                   "max": max(sorted_lengths)},
        "weights_scope": "frozen base checkpoint; OOF fine-tuning preserves architecture and inference resource shape",
        "inference_precision": "fp32 for both rankers",
    })
    del scorer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def _answer(answerer, item, cid):
    chunk = item.chunk_by_id(int(cid))
    raw = answerer(question=item.question, context=chunk.text)
    return {"qa_answer": str(raw.get("answer") or ""), "qa_chunk_id": int(cid),
            "qa_chunk_start_char": int(raw.get("start") or 0), "qa_chunk_end_char": int(raw.get("end") or 0)}


def _pipeline_benchmark(state, ranker_key, device, output, config, *, use_stage2=False):
    spec = RANKERS[ranker_key]
    records = sorted(state.records, key=lambda item: str(item.question_id))[:config.benchmark_questions]
    scorer = FrozenRevisionCrossEncoder(spec, device, config.max_length,
                                        config.prediction_batch_size,
                                        local_files_only=config.local_files_only)
    factory = ModelFactory(PipelineConfig(profile=profile_for_language("ru"), device=device))
    answerer = factory.answerer
    encoder = None
    tuning = None
    if use_stage2:
        encoder = build_encoder(Path(output) / "runtime_semantic_cache" / state.spec.key,
                                device=device, smoke=config.smoke, progress=None)
        tuning_path = Path(output).parent / state.spec.key / "stage2/global_deployable_parameters.json"
        frozen = json.loads(tuning_path.read_text(encoding="utf-8"))
        tuning = {"v6": frozen["V6"], "sem": frozen["SEM"], "router": frozen["router"]}

    def execute(item):
        pairs = [(item.question, chunk.text) for chunk in item.chunks]
        scores = scorer.predict(pairs)
        order = [cid for _score, cid in sorted(zip(scores, [int(chunk.chunk_id) for chunk in item.chunks]),
                                                key=lambda pair: (-pair[0], pair[1]))]
        required = order[:state.spec.k] if use_stage2 else order[:1]
        qa = {cid: _answer(answerer, item, cid) for cid in required}
        if use_stage2:
            row = {"ranked_ids": order, "scores": {str(cid): score for cid, score in zip([int(c.chunk_id) for c in item.chunks], scores)}}
            features = agreement_features(item, row, qa, k=state.spec.k, min_overlap=SOURCE_OVERLAP)
            answers = {cid: qa[cid]["qa_answer"] for cid in required}
            valid_answers = [(cid, text) for cid, text in answers.items() if text.strip()]
            vectors = encoder.backend.encode([text for _cid, text in valid_answers]) if valid_answers else []
            embeddings = {cid: l2_normalize(vectors[index]) for index, (cid, _text) in enumerate(valid_answers)}
            pack = adaptive_semantic_pack(answers, embeddings, required)
            v6_scores = {cid: v6_fused_score(features[cid], **tuning["v6"]) for cid in required}
            sem_scores = {cid: float(tuning["sem"]["lambda_m"]) * float(features[cid]["majority"]) +
                          float(tuning["sem"]["lambda_s"]) * float(pack["sem_rel"][str(cid)]) for cid in required}
            v6_order = isolate_with_b0_tiebreak(item, row, v6_scores, k=state.spec.k)[0]
            sem_order = isolate_with_b0_tiebreak(item, row, sem_scores, k=state.spec.k)[0]
            confidence = pack.get("consensus_conf")
            tau = tuning["router"]["tau"]
            use_sem = bool(tau is not None and routing_available(pack, sem_order[0]) and
                           confidence is not None and float(confidence) >= float(tau))
            _final_order = sem_order if use_sem else v6_order
        return len(pairs), len(required)

    warm_item = records[0]
    for _ in range(config.benchmark_warmup):
        execute(warm_item)
    _reset_peak(device)
    observations, total_pairs, qa_forwards = [], 0, 0
    for _ in range(config.benchmark_repetitions):
        _sync(device)
        started = time.perf_counter()
        pairs = forwards = 0
        for item in records:
            p, q = execute(item)
            pairs += p
            forwards += q
        _sync(device)
        observations.append(time.perf_counter() - started)
        total_pairs, qa_forwards = pairs, forwards
    result = _summary(observations)
    result.update({
        "questions_per_repetition": len(records),
        "latency_seconds_per_question": result["median_seconds"] / max(len(records), 1),
        "throughput_questions_per_second": len(records) / result["median_seconds"],
        "ranker_forward_pairs_per_repetition": total_pairs,
        "qa_forward_count_per_repetition": qa_forwards,
        "qa_forward_count_per_question": qa_forwards / max(len(records), 1),
        "peak_gpu_allocated_bytes": _peak(device),
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "includes": ["all-candidate ranker", "real frozen QA forwards"] +
                    (["top-K stage2 features", "semantic router feature", "adaptive decision"] if use_stage2 else ["top1 final answer"]),
    })
    del scorer, factory, answerer, encoder
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def run_resource_benchmark(config, output, states, progress):
    progress.stage("resource-benchmark", "ranker-only and end-to-end measured comparison")
    from experiments.qa_chunk_selection.v4_4_ru_q5_b0.pipeline import choose_device
    device, _ = choose_device(config.device)
    root = Path(output) / "resource_benchmark"
    write_json(root / "environment.json", _environment(device))
    ranker_only, end_to_end = {}, {}
    for dataset, state in states.items():
        records, pairs, lengths = _sample_pairs(state, config.benchmark_questions)
        ranker_only[dataset] = {"sequence_length_words": lengths, "questions": len(records)}
        for ranker_key, spec in RANKERS.items():
            progress.log("ranker-only benchmark", dataset=dataset, ranker=ranker_key, pairs=len(pairs))
            ranker_only[dataset][ranker_key] = {
                "matched_operational_batch": _ranker_benchmark(
                    spec, device, pairs, batch_size=config.prediction_batch_size,
                    warmup=config.benchmark_warmup, repetitions=config.benchmark_repetitions,
                    local_files_only=config.local_files_only),
                "best_feasible_throughput": _ranker_benchmark(
                    spec, device, pairs,
                    batch_size=config.dity_throughput_batch if ranker_key == "dity" else config.bge_throughput_batch,
                    warmup=config.benchmark_warmup, repetitions=config.benchmark_repetitions,
                    local_files_only=config.local_files_only),
            }
        end_to_end[dataset] = {}
        for system, ranker_key, use_stage2 in (("S1_dity_b0", "dity", False),
                                                ("S3_dity_stage2", "dity", True),
                                                ("S4_bge_b0", "bge", False)):
            progress.log("end-to-end benchmark", dataset=dataset, pipeline=system)
            end_to_end[dataset][system] = _pipeline_benchmark(
                state, ranker_key, device, root, config, use_stage2=use_stage2)
    write_json(root / "ranker_only/results.json", ranker_only)
    write_json(root / "end_to_end/results.json", end_to_end)
    write_json(root / "raw/observations.json", {"ranker_only": ranker_only, "end_to_end": end_to_end})
    summary = {
        "status": "complete", "device": str(device),
        "ranker_only": {dataset: {ranker: values["matched_operational_batch"] for ranker, values in rows.items() if ranker in RANKERS}
                        for dataset, rows in ranker_only.items()},
        "end_to_end": end_to_end,
        "interpretation": "dimension-wise quality/resource comparison; no scalar utility and no unregistered non-inferiority margin",
        "weights_scope": "resource timing uses the exact frozen base revisions; quality uses newly fine-tuned OOF models. Fine-tuning changes values, not architecture/parameter count.",
        "cpu_rss_note": "ru_maxrss is a monotonic process-level peak; compare with caution after earlier components",
    }
    write_json(root / "summary.json", summary)
