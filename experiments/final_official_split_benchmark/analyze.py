"""Gold-side TEST evaluation, paired bootstrap, and final result tables."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.metrics import qa_row
from experiments.qa_chunk_selection.v4_4_ru.metrics_ru import rouge_l_f1 as unicode_rouge_l
from experiments.qa_chunk_selection.v4_4_ru_ablation.lemma import RussianLemmatizer

from .contract import CELL_SPECS, PROPOSED
from .data import load_test_gold
from .io_guard import read_json, read_jsonl, sha256, write_json, write_jsonl, write_text


def _prediction_paths(root: Path, cell: str):
    # ``_runtime_features`` contains raw feature rows, not prediction envelopes.
    # Only system directories are valid evaluation inputs.
    return sorted(path for path in root.glob(f"test_predictions/*/{cell}.jsonl")
                  if not path.parent.name.startswith("_"))


def assert_blind_complete(config) -> dict:
    capabilities = read_json(config.run_dir / "manifests/system_capability_matrix.json")
    report = {}
    for cell in config.cells:
        marker = config.run_dir / "test_predictions" / f"{cell}_COMPLETE.json"
        if not marker.is_file(): raise RuntimeError(f"TEST evaluation blocked: blind marker missing for {cell}")
        state = read_json(marker); required = {x["system"] for x in capabilities["rows"] if x["cell"] == cell and x["status"] == "RUN"}
        if not required <= set(state["systems"]): raise RuntimeError(f"{cell}: applicable real-system prediction set is incomplete")
        for system, value in state["systems"].items():
            path = config.run_dir / "test_predictions" / system / f"{cell}.jsonl"
            if sha256(path) != value["sha256"]: raise RuntimeError(f"blind prediction mutated: {path}")
        report[cell] = state
    return report


def _seed_average(rows):
    byq = defaultdict(list)
    for row in rows: byq[row["question_id"]].append(row)
    output = []
    for qid, values in byq.items():
        first = values[0]
        output.append({"question_id": qid, "document_id": first["document_id"],
            **{key: float(np.mean([x[key] for x in values])) for key in ("token_f1", "char_f1", "unicode_rouge_l", "exact_match")},
            "lemma_f1": float(np.mean([x["lemma_f1"] for x in values])) if values[0]["lemma_f1"] is not None else None,
            "hit@1": float(np.mean([x["hit@1"] for x in values])) if values[0]["hit@1"] is not None else None,
            "hit@k": float(np.mean([x["hit@k"] for x in values])) if values[0].get("hit@k") is not None else None,
            "hn@1": float(np.mean([x["hn@1"] for x in values])) if values[0]["hn@1"] is not None else None,
            "safe_rate": float(np.mean([x["safe_rate"] for x in values])) if values[0].get("safe_rate") is not None else None,
            "unsafe_rate": float(np.mean([x["unsafe_rate"] for x in values])) if values[0].get("unsafe_rate") is not None else None,
            "f1_selected_positive": float(np.mean([x["f1_selected_positive"] for x in values if x["f1_selected_positive"] is not None])) if any(x.get("f1_selected_positive") is not None for x in values) else None,
            "seeds": len(values)})
    return output


def evaluate(config, progress) -> dict:
    assert_blind_complete(config); lemmatizer = RussianLemmatizer(smoke=False)
    availability = read_json(config.run_dir / "manifests/evidence_label_availability.json")
    summaries, question_metrics = {}, {}
    for cell in config.cells:
        progress.stage(f"evaluate:{cell}", "unlock TEST gold and evaluate immutable blind predictions")
        gold = {str(x.question_id): x for x in load_test_gold(cell, unlocked=True)}
        population = set(gold); summaries[cell] = {}; question_metrics[cell] = {}
        for path in _prediction_paths(config.run_dir, cell):
            system = path.parent.name; predictions = read_jsonl(path); qids = {str(x["runtime"]["question_id"]) for x in predictions}
            missing = population - qids
            if missing: raise RuntimeError(f"{cell}/{system}: missing canonical answerable TEST questions: {sorted(missing)[:5]}")
            stochastic = system in {"B5-EN", "P1", "P2", "P3", "P4", "P2-NO-FULLLEX", "P2-NO-PAIRLEX", "P2-NO-RISK"}
            expected_seeds = set(config.seeds) if stochastic else {0}
            by_question_seeds = defaultdict(set)
            for value in predictions:
                by_question_seeds[str(value["runtime"]["question_id"])].add(int(value["runtime"].get("model_seed", 0)))
            bad = {qid: sorted(by_question_seeds[qid]) for qid in population if by_question_seeds[qid] != expected_seeds}
            if bad: raise RuntimeError(f"{cell}/{system}: incomplete seed coverage: {list(bad.items())[:3]}")
            rows = []
            for prediction in predictions:
                runtime = prediction["runtime"]; qid = str(runtime["question_id"])
                if qid not in gold: continue
                item = gold[qid]; answer = str(runtime.get("final_answer") or ""); metrics = qa_row(answer, item.gold_answers)
                selected = runtime.get("selected_chunk_id"); chunk = item.chunk_by_id(int(selected)) if selected is not None else None
                row = {"dataset_cell": cell, "system_id": system, "document_id": item.document_id, "question_id": qid,
                       "seed": int(runtime.get("model_seed", 0)), "selected_chunk_id": selected, "final_answer": answer,
                       "token_f1": metrics["token_f1"], "char_f1": metrics["char_f1"],
                       "unicode_rouge_l": max((unicode_rouge_l(answer, x) for x in item.gold_answers), default=0.0),
                       "exact_match": metrics["exact_match"],
                       "lemma_f1": lemmatizer.gold_f1(answer, item.gold_answers) if CELL_SPECS[cell].language == "ru" else None,
                       "hit@1": int(chunk.label == "positive") if chunk is not None and availability[cell]["positive_labels"] else None,
                       "hit@k": int(any(item.chunk_by_id(int(cid)).label == "positive" for cid in runtime.get("candidate_chunk_ids", []) if any(c.chunk_id == int(cid) for c in item.chunks))) if runtime.get("candidate_chunk_ids") and availability[cell]["positive_labels"] else None,
                       "hn@1": int(chunk.label == "hard_negative") if chunk is not None and availability[cell]["hard_negative_labels"] else None,
                       "safe_rate": int(chunk.label == "positive") if chunk is not None and availability[cell]["positive_labels"] else None,
                       "unsafe_rate": int(chunk.label == "hard_negative") if chunk is not None and availability[cell]["hard_negative_labels"] else None,
                       "f1_selected_positive": metrics["token_f1"] if chunk is not None and chunk.label == "positive" else None,
                       "selected_evidence": chunk.label if chunk is not None else None}
                rows.append(row)
            averaged = _seed_average(rows); question_metrics[cell][system] = averaged
            mean = lambda key: float(np.mean([x[key] for x in averaged if x[key] is not None])) if any(x[key] is not None for x in averaged) else None
            summaries[cell][system] = {"questions": len(averaged), "prediction_rows": len(rows), "seeds_per_question": sorted({x["seeds"] for x in averaged}),
                **{key: mean(key) for key in ("token_f1", "char_f1", "unicode_rouge_l", "exact_match", "lemma_f1", "hit@1", "hit@k", "hn@1", "safe_rate", "unsafe_rate", "f1_selected_positive")}}
            write_jsonl(config.run_dir / "test_analysis" / system / f"{cell}.jsonl", rows, immutable=True)
        # Gold-side diagnostic oracles are deliberately created only after every
        # real-system blind prediction is locked. They never enter real ranking.
        features = defaultdict(list)
        for value in read_jsonl(config.run_dir / "test_predictions/_runtime_features" / f"{cell}.jsonl"):
            features[str(value["question_id"])].append(value)
        for system in ("D1", "D2"):
            rows = []
            for qid, item in gold.items():
                candidates = sorted(features[qid], key=lambda x: (int(x["b0_rank"]), int(x["chunk_id"])))
                if system == "D1":
                    selected = max(candidates, key=lambda x: qa_row(str(x.get("qa_answer") or ""), item.gold_answers)["token_f1"])
                else:
                    positive = [x for x in candidates if item.chunk_by_id(int(x["chunk_id"])).label == "positive"]
                    selected = positive[0] if positive else candidates[0]
                answer = str(selected.get("qa_answer") or ""); metric = qa_row(answer, item.gold_answers); chunk = item.chunk_by_id(int(selected["chunk_id"]))
                rows.append({"dataset_cell": cell, "system_id": system, "document_id": item.document_id, "question_id": qid, "seed": 0,
                    "selected_chunk_id": int(selected["chunk_id"]), "final_answer": answer, "token_f1": metric["token_f1"], "char_f1": metric["char_f1"],
                    "unicode_rouge_l": max((unicode_rouge_l(answer, x) for x in item.gold_answers), default=0.0), "exact_match": metric["exact_match"],
                    "lemma_f1": lemmatizer.gold_f1(answer, item.gold_answers) if CELL_SPECS[cell].language == "ru" else None,
                    "hit@1": int(chunk.label == "positive"), "hit@k": int(any(item.chunk_by_id(int(x["chunk_id"])).label == "positive" for x in candidates)),
                    "hn@1": int(chunk.label == "hard_negative") if availability[cell]["hard_negative_labels"] else None,
                    "safe_rate": int(chunk.label == "positive"), "unsafe_rate": int(chunk.label == "hard_negative") if availability[cell]["hard_negative_labels"] else None,
                    "f1_selected_positive": metric["token_f1"] if chunk.label == "positive" else None,
                    "selected_evidence": chunk.label, "diagnostic_oracle": True})
            averaged = _seed_average(rows); question_metrics[cell][system] = averaged
            mean = lambda key: float(np.mean([x[key] for x in averaged if x[key] is not None])) if any(x[key] is not None for x in averaged) else None
            summaries[cell][system] = {"questions": len(averaged), "prediction_rows": len(rows), "seeds_per_question": [1],
                **{key: mean(key) for key in ("token_f1", "char_f1", "unicode_rouge_l", "exact_match", "lemma_f1", "hit@1", "hn@1")}, "diagnostic_only": True}
            write_jsonl(config.run_dir / "test_analysis" / system / f"{cell}.jsonl", rows, immutable=True)
    write_json(config.run_dir / "test_analysis/primary_metrics.json", summaries, immutable=True)
    write_json(config.run_dir / "test_analysis/question_metrics.json", question_metrics, immutable=True)
    return {"summaries": summaries, "question_metrics": question_metrics}


def family_balanced(summaries: dict) -> dict:
    systems = sorted(set.intersection(*[set(summaries[cell]) for cell in summaries]) - {"D1", "D2"})
    result = {}
    for system in systems:
        scier = np.mean([summaries[c][system]["token_f1"] for c in ("scier_128", "scier_384") if c in summaries])
        family = {"SciREX": summaries["scirex"][system]["token_f1"], "SciER": float(scier),
                  "RuSERRC": summaries["russerrc"][system]["token_f1"], "SciMDIX": summaries["scimdix"][system]["token_f1"]}
        result[system] = {"families": family, "ALL_STANDARDIZED": float(np.mean(list(family.values())))}
    return result


def paired_bootstrap(left, right, iterations, seed=42):
    li = {x["question_id"]: x for x in left}; ri = {x["question_id"]: x for x in right}; common = sorted(set(li) & set(ri)); bydoc = defaultdict(list)
    for qid in common: bydoc[li[qid]["document_id"]].append(float(li[qid]["token_f1"] - ri[qid]["token_f1"]))
    docs = sorted(bydoc); observed = float(np.mean([x for d in docs for x in bydoc[d]])) if docs else 0.0
    rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations):
        draw = rng.choice(docs, len(docs), replace=True) if docs else []
        samples.append(float(np.mean([x for d in draw for x in bydoc[d]])) if len(draw) else 0.0)
    return {"delta_token_f1": observed, "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)),
            "documents": len(docs), "questions": len(common), "iterations": iterations, "cluster": "source document", "seed": seed}


def family_paired_bootstrap(question_metrics, challenger, baseline, iterations, cells, seed=42):
    """Equal-family bootstrap with one coupled source-document draw for both SciER cells."""
    diffs = {}
    for cell in cells:
        left = {x["question_id"]: x for x in question_metrics[cell][challenger]}
        right = {x["question_id"]: x for x in question_metrics[cell][baseline]}
        by_doc = defaultdict(list)
        for qid in set(left) & set(right):
            by_doc[left[qid]["document_id"]].append(float(left[qid]["token_f1"] - right[qid]["token_f1"]))
        diffs[cell] = by_doc
    families = []
    if "scirex" in cells: families.append(("SciREX", ("scirex",)))
    scier = tuple(x for x in ("scier_128", "scier_384") if x in cells)
    if scier: families.append(("SciER", scier))
    if "russerrc" in cells: families.append(("RuSERRC", ("russerrc",)))
    if "scimdix" in cells: families.append(("SciMDIX", ("scimdix",)))

    def observed_family(member_cells):
        return float(np.mean([np.mean([v for values in diffs[cell].values() for v in values]) for cell in member_cells]))
    observed = float(np.mean([observed_family(members) for _, members in families]))
    rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations):
        family_values = []
        for family, members in families:
            if family == "SciER":
                docs = sorted(set.intersection(*[set(diffs[cell]) for cell in members])); draw = rng.choice(docs, len(docs), replace=True)
                family_values.append(float(np.mean([np.mean([v for doc in draw for v in diffs[cell][doc]]) for cell in members])))
            else:
                cell = members[0]; docs = sorted(diffs[cell]); draw = rng.choice(docs, len(docs), replace=True)
                family_values.append(float(np.mean([v for doc in draw for v in diffs[cell][doc]])))
        samples.append(float(np.mean(family_values)))
    return {"delta_token_f1": observed, "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5)),
            "iterations": iterations, "families": [x[0] for x in families], "family_weights": "equal", "scier_coupled": True,
            "seed_handling": "average model seeds within question before document bootstrap"}


def statistics(config, evaluated: dict) -> dict:
    q = evaluated["question_metrics"]; contrasts = {}
    universal = ("B1", "B2", "B3", "B4b")
    for cell in config.cells:
        if "P2" not in q[cell]: continue
        contrasts[cell] = {}
        for baseline in universal:
            if baseline in q[cell]: contrasts[cell][f"P2_vs_{baseline}"] = paired_bootstrap(q[cell]["P2"], q[cell][baseline], config.bootstrap)
        if CELL_SPECS[cell].language == "en" and "B5-EN" in q[cell]:
            contrasts[cell]["P2_vs_B5-EN"] = paired_bootstrap(q[cell]["P2"], q[cell]["B5-EN"], config.bootstrap)
        if "G1" in q[cell]: contrasts[cell]["P2_vs_G1"] = paired_bootstrap(q[cell]["P2"], q[cell]["G1"], config.bootstrap)
        for challenger in ("P1", "P3", "P4"):
            if challenger in q[cell]: contrasts[cell][f"{challenger}_vs_P2"] = paired_bootstrap(q[cell][challenger], q[cell]["P2"], config.bootstrap)
    balanced = family_balanced(evaluated["summaries"])
    family_contrasts = {}
    all_cells = tuple(config.cells)
    for baseline in ("B1", "B2", "B3", "B4b"):
        if all(baseline in q[cell] and "P2" in q[cell] for cell in all_cells):
            family_contrasts[f"P2_vs_{baseline}"] = family_paired_bootstrap(q, "P2", baseline, config.bootstrap, all_cells)
    for challenger in ("P1", "P3", "P4"):
        if all(challenger in q[cell] and "P2" in q[cell] for cell in all_cells):
            family_contrasts[f"{challenger}_vs_P2"] = family_paired_bootstrap(q, challenger, "P2", config.bootstrap, all_cells)
    en_cells = tuple(x for x in ("scirex", "scier_128", "scier_384") if x in q)
    if en_cells and all("B5-EN" in q[cell] and "P2" in q[cell] for cell in en_cells):
        family_contrasts["P2_vs_B5-EN_EN_ONLY"] = family_paired_bootstrap(q, "P2", "B5-EN", config.bootstrap, en_cells)
    exposure = read_json(config.run_dir / "test_exposure/exposure.json")
    untouched = [c for c, value in exposure.items() if value["status"] == "UNTOUCHED"]
    confirmation_families = {CELL_SPECS[c].family for c in untouched}
    scope = {"untouched_cells": untouched, "independent_untouched_families": sorted(confirmation_families),
             "global_confirmation_mean_reported": len(confirmation_families) >= 2,
             "rule": "global UNTOUCHED_CONFIRMATION mean requires at least two independent untouched corpus families"}
    untouched_official = tuple(c for c in untouched if CELL_SPECS[c].split_mode != "NO_OFFICIAL_TEST")
    untouched_metrics = {}
    if len(confirmation_families) >= 2 and untouched_official:
        common_systems = set.intersection(*[set(evaluated["summaries"][cell]) for cell in untouched_official]) - {"D1", "D2"}
        for system in sorted(common_systems):
            grouped = defaultdict(list)
            for cell in untouched_official: grouped[CELL_SPECS[cell].family].append(evaluated["summaries"][cell][system]["token_f1"])
            families = {family: float(np.mean(values)) for family, values in grouped.items()}
            untouched_metrics[system] = {"families": families, "UNTOUCHED_CONFIRMATION": float(np.mean(list(families.values())))}
    confirmation_contrasts = {}
    if len(confirmation_families) >= 2:
        for baseline in ("B1", "B2", "B3", "B4b"):
            if all(baseline in q[cell] and "P2" in q[cell] for cell in untouched_official):
                confirmation_contrasts[f"P2_vs_{baseline}"] = family_paired_bootstrap(q, "P2", baseline, config.bootstrap, untouched_official)
    claims = {}
    simple = [family_contrasts.get(f"P2_vs_{name}") for name in ("B1", "B2", "B3")]
    simple_cell_deltas = [contrasts[cell][f"P2_vs_{name}"]["delta_token_f1"] for cell in config.cells for name in ("B1", "B2", "B3") if f"P2_vs_{name}" in contrasts.get(cell, {})]
    claims["H2_BEATS_SIMPLE_BASELINES_STANDARDIZED"] = bool(simple and all(x and x["ci_low"] > 0 for x in simple) and simple_cell_deltas and min(simple_cell_deltas) >= -1.0)
    bge = family_contrasts.get("P2_vs_B4b")
    bge_cells = [contrasts[cell]["P2_vs_B4b"]["delta_token_f1"] for cell in config.cells if "P2_vs_B4b" in contrasts.get(cell, {})]
    claims["H2_BEYOND_STRONG_RERANKER_STANDARDIZED"] = bool(bge and bge["ci_low"] > 0 and bge_cells and min(bge_cells) >= -1.0)
    b5 = family_contrasts.get("P2_vs_B5-EN_EN_ONLY")
    claims["H2_NONINFERIOR_TO_MODERNBERT_EN"] = bool(b5 and b5["ci_low"] > -1.0)
    proposed_scores = {system: balanced[system]["ALL_STANDARDIZED"] for system in ("P1", "P2", "P3", "P4") if system in balanced}
    claims["BEST_PROPOSED_OBSERVED"] = max(proposed_scores, key=proposed_scores.get) if proposed_scores else None
    result = {"per_cell_contrasts": contrasts, "family_balanced": balanced, "family_balanced_contrasts": family_contrasts,
              "untouched_confirmation_metrics": untouched_metrics, "untouched_confirmation_contrasts": confirmation_contrasts,
              "confirmation_scope": scope, "claim_audit": claims,
              "scier_bootstrap_coupling": "one resampled source-document multiset is applied to SciER128 and SciER384"}
    write_json(config.run_dir / "statistics/statistics.json", result, immutable=True)
    write_json(config.run_dir / "test_analysis/paired_bootstrap.json", contrasts, immutable=True)
    write_json(config.run_dir / "test_analysis/family_bootstrap.json", {"standardized": family_contrasts, "confirmation": confirmation_contrasts}, immutable=True)
    write_json(config.run_dir / "test_analysis/primary_contrasts.json", claims, immutable=True)
    write_json(config.run_dir / "test_analysis/proposed_family.json", {k: v for k, v in balanced.items() if k in {"P1", "P2", "P3", "P4"}}, immutable=True)
    return result


def report(config, evaluated, stats) -> Path:
    summaries = evaluated["summaries"]; exposure = read_json(config.run_dir / "test_exposure/exposure.json")
    lines = ["# Final Official-Split Benchmark / H2 Family Confirmation", "",
             "This report was generated only after all applicable real-system blind predictions were complete and hash-locked.", "",
             "## Test independence", "", "| Cell | Split mode | Exposure |", "|---|---|---|"]
    for cell in config.cells: lines.append(f"| {cell} | {CELL_SPECS[cell].split_mode} | {exposure[cell]['status']} |")
    lines += ["", "## Primary TEST results", "", "Token F1 is averaged over model seeds within each question before dataset aggregation.", "",
              "| Cell | System | Questions | Token F1 | Char F1 | Unicode ROUGE-L | EM | Lemma F1 RU | Hit@1 | HN@1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.4f}"
    for cell in config.cells:
        for system, value in sorted(summaries[cell].items()):
            lines.append(f"| {cell} | {system} | {value['questions']} | {fmt(value['token_f1'])} | {fmt(value['char_f1'])} | {fmt(value['unicode_rouge_l'])} | {fmt(value['exact_match'])} | {fmt(value['lemma_f1'])} | {fmt(value['hit@1'])} | {fmt(value['hn@1'])} |")
    lines += ["", "## Family-balanced standardized result", "", "SciER contributes the mean of its 128/24 and 384/50 cells; SciREX, SciER, RuSERRC and SciMDIX then receive equal weight.", "", "| System | SciREX | SciER | RuSERRC | SciMDIX | ALL_STANDARDIZED |", "|---|---:|---:|---:|---:|---:|"]
    for system, value in sorted(stats["family_balanced"].items()):
        f = value["families"]; lines.append(f"| {system} | {f['SciREX']:.4f} | {f['SciER']:.4f} | {f['RuSERRC']:.4f} | {f['SciMDIX']:.4f} | {value['ALL_STANDARDIZED']:.4f} |")
    lines += ["", "## Evidence diagnostics", "", "Missing exact evidence labels are shown as — and are never replaced with zero.", "",
              "| Cell | System | Hit@1 | Hit@K | HN@1 | SAFE | UNSAFE | F1 given selected positive |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cell in config.cells:
        for system, value in sorted(summaries[cell].items()):
            lines.append(f"| {cell} | {system} | {fmt(value.get('hit@1'))} | {fmt(value.get('hit@k'))} | {fmt(value.get('hn@1'))} | {fmt(value.get('safe_rate'))} | {fmt(value.get('unsafe_rate'))} | {fmt(value.get('f1_selected_positive'))} |")
    lines += ["", "## Family-balanced paired contrasts", "", "SciER128 and SciER384 use the same resampled source-document multiset.", "",
              "| Contrast | Scope | Δ Token F1 | 95% CI |", "|---|---|---:|---:|"]
    for name, value in stats.get("family_balanced_contrasts", {}).items():
        lines.append(f"| {name} | STANDARDIZED | {value['delta_token_f1']:.4f} | [{value['ci_low']:.4f}, {value['ci_high']:.4f}] |")
    for name, value in stats.get("untouched_confirmation_contrasts", {}).items():
        lines.append(f"| {name} | UNTOUCHED_CONFIRMATION | {value['delta_token_f1']:.4f} | [{value['ci_low']:.4f}, {value['ci_high']:.4f}] |")
    lines += ["", "## Predeclared paired contrasts", ""]
    for cell, comparisons in stats["per_cell_contrasts"].items():
        lines += [f"### {cell}", "", "| Contrast | Δ Token F1 | 95% CI | Documents |", "|---|---:|---:|---:|"]
        for name, value in comparisons.items(): lines.append(f"| {name} | {value['delta_token_f1']:.4f} | [{value['ci_low']:.4f}, {value['ci_high']:.4f}] | {value['documents']} |")
        lines.append("")
    if stats.get("untouched_confirmation_metrics"):
        lines += ["", "## Untouched official-test confirmation", "", "| System | Family-balanced Token F1 |", "|---|---:|"]
        for system, value in sorted(stats["untouched_confirmation_metrics"].items()):
            lines.append(f"| {system} | {value['UNTOUCHED_CONFIRMATION']:.4f} |")
    lines += ["", "## Frozen claim audit", ""]
    for name, value in stats.get("claim_audit", {}).items(): lines.append(f"- {name}: **{value}**")
    lines.append("")
    scope = stats["confirmation_scope"]
    resource_path = config.run_dir / "resources/summary.json"
    if resource_path.is_file():
        lines += ["## Resource-quality measurements", "", "Fresh process per cell/system; model loading excluded; ten warmup questions and three measured repeats.", "",
                  "| Cell | System | Median s/q | P95 s/q | q/s | Peak VRAM bytes | Peak RSS KiB | Parameters |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in read_json(resource_path)["rows"]:
            lines.append(f"| {row['cell']} | {row['system']} | {row['median_latency_s']:.6f} | {row['p95_latency_s']:.6f} | {row['throughput_questions_s']:.4f} | {row.get('peak_vram_bytes') or '—'} | {row['peak_cpu_rss_kib']} | {row['loaded_neural_parameters']} |")
        lines.append("")
    lines += ["## Claim scope", "", f"Untouched cells: {', '.join(scope['untouched_cells']) or 'none'}.", "",
              ("A global UNTOUCHED_CONFIRMATION mean is permitted because at least two independent untouched families remain." if scope["global_confirmation_mean_reported"] else "A global confirmation mean is not reported because fewer than two independent untouched corpus families remain."), "",
              "RuSERRC is a standardized nonofficial holdout and is excluded from official-test confirmation claims.", "",
              "## Interpretation rules", "", "- D1/D2 are diagnostics and do not enter the real-system ranking.", "- Qwen uses a different full-document context regime.", "- B5-EN is an English-only reader contrast and has no five-cell aggregate.", "- Missing evidence metrics are left missing; they are never replaced by zero.", ""]
    path = config.run_dir / "reports/final_benchmark_results.md"; write_text(path, "\n".join(lines), immutable=True)
    statistical = ["# Statistical analysis", "", "All stochastic seeds are averaged within question. Bootstrap resamples source documents; SciER128/384 use a coupled draw.", ""]
    for name, value in stats.get("family_balanced_contrasts", {}).items(): statistical.append(f"- {name}: Δ={value['delta_token_f1']:.4f}, 95% CI [{value['ci_low']:.4f}, {value['ci_high']:.4f}]")
    write_text(config.run_dir / "reports/statistical_analysis.md", "\n".join(statistical) + "\n", immutable=True)
    resources = ["# Resource analysis", "", "Fresh-process measurements exclude one-time model loading and dataset parsing.", ""]
    resource_file = config.run_dir / "resources/summary.json"
    if resource_file.is_file():
        resource_payload = read_json(resource_file)
        for row in resource_payload["rows"]: resources.append(f"- {row['cell']} / {row['system']}: median {row['median_latency_s']:.6f}s, P95 {row['p95_latency_s']:.6f}s, {row['throughput_questions_s']:.4f} q/s")
        resources += ["", "## Final refit training"]
        for cell, value in resource_payload.get("training", {}).items(): resources.append(f"- {cell}: {value.get('training_wall_seconds', 0):.1f}s wall time; {value.get('training_gpu_hours_upper_bound_one_gpu', 0):.4f} one-GPU-hours upper bound")
    else: resources.append("- Resource stage was not run.")
    write_text(config.run_dir / "reports/resource_analysis.md", "\n".join(resources) + "\n", immutable=True)
    reproducibility = ["# Reproducibility", "", f"Frozen contract: `{config.run_dir / 'frozen_contract.yaml'}`", f"Manifest: `{config.run_dir / 'manifest.json'}`", f"System registry: `{config.run_dir / 'audit/system_registry.json'}`", "", "Historical V1–V10 artifacts are hash-checked before and after the run. TEST predictions are immutable and evaluated only after full real-system completion."]
    write_text(config.run_dir / "reports/reproducibility.md", "\n".join(reproducibility) + "\n", immutable=True)
    return path
