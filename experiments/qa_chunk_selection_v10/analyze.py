"""V10 scientific aggregation, diagnostics, bootstrap and reports."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from .contract import FAMILIES, SYSTEMS
from .io_guard import read_jsonl, write_json, write_text


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _load(root, system, cell):
    return [row for path in sorted((root / "predictions" / system / cell).glob("outer_fold_*/seed_*.jsonl")) for row in read_jsonl(path)]


def _primary(rows):
    evaluation = [x["evaluation"] for x in rows]
    return {
        "n_prediction_rows": len(rows), "n_questions": len({x["runtime"]["question_id"] for x in rows}),
        "token_f1": _mean(x["token_f1"] for x in evaluation),
        "char_f1": _mean(x.get("char_f1", 0.0) for x in evaluation),
        "rouge_l_f1": _mean(x.get("rouge_l_f1", 0.0) for x in evaluation),
        "exact_match": _mean(x.get("exact_match", 0.0) for x in evaluation),
        "hit@1": _mean(x["hit@1"] for x in evaluation), "hn@1": _mean(x["hn@1"] for x in evaluation),
        "hit@k": _mean(x["hit@k"] for x in evaluation), "hn@k": _mean(x["hn@k"] for x in evaluation),
        "selected_safe_evidence": _mean(x.get("selected_safe_evidence", x["hit@1"]) for x in evaluation),
        "selected_unsafe_evidence": _mean(x.get("selected_unsafe_evidence", x["hn@1"]) for x in evaluation),
        "switch_rate": _mean(x["changed_top1"] for x in evaluation),
    }


def _question_seed_average(rows, metric):
    grouped = defaultdict(list); documents = {}
    for row in rows:
        qid = row["runtime"]["question_id"]; grouped[qid].append(float(metric(row))); documents[qid] = row["runtime"]["canonical_document_id"]
    return {qid: _mean(values) for qid, values in grouped.items()}, documents


def _bootstrap(left, right, iterations, seed=42):
    left_values, docs = _question_seed_average(left, lambda x: x["evaluation"]["token_f1"])
    right_values, right_docs = _question_seed_average(right, lambda x: x["evaluation"]["token_f1"])
    keys = sorted(set(left_values) & set(right_values))
    if not keys: raise RuntimeError("empty paired bootstrap")
    if any(docs[x] != right_docs[x] for x in keys): raise AssertionError("bootstrap document mismatch")
    by_doc = defaultdict(list)
    for key in keys: by_doc[docs[key]].append(left_values[key] - right_values[key])
    document_ids = sorted(by_doc); rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations):
        chosen = rng.choice(document_ids, size=len(document_ids), replace=True)
        samples.append(_mean(value for document in chosen for value in by_doc[document]))
    return {
        "delta": _mean(left_values[x] - right_values[x] for x in keys),
        "lcb95": float(np.percentile(samples, 2.5)), "ucb95": float(np.percentile(samples, 97.5)),
        "iterations": iterations, "cluster": "canonical_document_id", "seed": seed,
        "seeds_averaged_within_question": True,
    }


def _global_bootstrap(data, cells, iterations, seed=42):
    """Family-balanced bootstrap with one coupled SciER document sample."""
    deltas = {}; documents = {}
    for cell in cells:
        left, left_docs = _question_seed_average(data[cell]["S2"], lambda x: x["evaluation"]["token_f1"])
        right, right_docs = _question_seed_average(data[cell]["S1"], lambda x: x["evaluation"]["token_f1"])
        keys = sorted(set(left) & set(right))
        deltas[cell] = {key: left[key] - right[key] for key in keys}
        documents[cell] = {key: left_docs[key] for key in keys}
        if any(left_docs[key] != right_docs[key] for key in keys):
            raise AssertionError("global bootstrap document mismatch")
    families = sorted(set(FAMILIES[x] for x in cells)); rng = np.random.default_rng(seed); samples = []
    for _ in range(iterations):
        effects = []
        for family in families:
            family_cells = [x for x in cells if FAMILIES[x] == family]
            document_ids = sorted({doc for cell in family_cells for doc in documents[cell].values()})
            sampled = rng.choice(document_ids, size=len(document_ids), replace=True)
            cell_effects = []
            for cell in family_cells:
                by_doc = defaultdict(list)
                for key, delta in deltas[cell].items():
                    by_doc[documents[cell][key]].append(delta)
                values = [value for document in sampled for value in by_doc.get(document, [])]
                if values:
                    cell_effects.append(_mean(values))
            effects.append(_mean(cell_effects))
        samples.append(_mean(effects))
    family_points = {
        family: _mean(_mean(deltas[cell].values()) for cell in cells if FAMILIES[cell] == family)
        for family in families
    }
    return {
        "family_effects": family_points, "family_balanced_delta_S2_S1": _mean(family_points.values()),
        "lcb95": float(np.percentile(samples, 2.5)), "ucb95": float(np.percentile(samples, 97.5)),
        "iterations": iterations, "seed": seed, "scier_coupled_document_resampling": True,
    }


def _binary_metrics(labels, scores):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    if not labels or len(set(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None, "brier": None, "log_loss": None}
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)), "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def _module_diagnostics(rows):
    switch_labels = []; switch_scores = []; answer_labels = []; answer_scores = []
    evidence_labels = []; evidence_scores = []; pair_labels = []; pair_scores = []
    regrets = []; top_f1 = []; oracle_capture = []
    for row in rows:
        runtime = row["runtime"]; evaluation = row["evaluation"]
        f1 = {int(k): float(v) for k, v in evaluation["candidate_f1"].items()}
        candidate_ids = [int(x) for x in runtime["candidate_ids"]]; base = candidate_ids[0]
        if len(candidate_ids) < 2: continue
        switch_labels.append(int(evaluation["positive_opportunity"])); switch_scores.append(float(runtime["p_switch"]))
        for chunk, score in runtime["p_answer_harm"].items():
            chunk = int(chunk); answer_labels.append(int(f1[chunk] < f1[base])); answer_scores.append(float(score))
        evidence = {int(k): v for k, v in evaluation["candidate_evidence"].items()}
        for chunk, score in runtime["p_evidence_harm"].items():
            chunk = int(chunk)
            if evidence[chunk] == "EXCLUDE": continue
            evidence_labels.append(int(evidence[chunk] == "UNSAFE_EVIDENCE")); evidence_scores.append(float(score))
        for pair in runtime["pairwise_probabilities"]:
            left, right = int(pair["left"]), int(pair["right"])
            if abs(f1[left] - f1[right]) < 1.0: continue
            pair_labels.append(int(f1[left] > f1[right])); pair_scores.append(float(pair["p_left_gt_right"]))
        top = int(runtime["rank_order"][0]); oracle = max(candidate_ids[1:], key=lambda x: (f1[x], -x))
        top_f1.append(f1[top]); regrets.append(f1[oracle] - f1[top]); oracle_capture.append(int(top == oracle))
    switch = _binary_metrics(switch_labels, switch_scores); answer = _binary_metrics(answer_labels, answer_scores); evidence = _binary_metrics(evidence_labels, evidence_scores)
    thresholded_switch = [int(score >= 0.5) for score in switch_scores]
    switch.update({
        "opportunity_recall_at_0.5": sum(p and y for p, y in zip(thresholded_switch, switch_labels)) / max(sum(switch_labels), 1),
        "opportunity_precision_at_0.5": sum(p and y for p, y in zip(thresholded_switch, switch_labels)) / max(sum(thresholded_switch), 1),
    })
    rank = _binary_metrics(pair_labels, pair_scores)
    rank.update({"top_ranked_alternative_f1": _mean(top_f1), "oracle_best_alternative_capture": _mean(oracle_capture), "mean_rank_regret": _mean(regrets), "median_rank_regret": float(np.median(regrets)) if regrets else 0.0})
    return switch, rank, answer, evidence


def _mechanism(rows):
    opportunities = [x for x in rows if x["evaluation"].get("positive_opportunity", max(x["evaluation"].get("candidate_utilities", {"0": 0}).values()) > 0)]
    switches = [x for x in rows if x["evaluation"]["changed_top1"]]
    capture = _mean(x["evaluation"].get("positive_capture", x["evaluation"].get("actual_switch_utility", 0) > 0) for x in opportunities)
    stay = _mean(not x["evaluation"]["changed_top1"] for x in opportunities)
    harmful = _mean(x["evaluation"].get("harmful_switch", x["evaluation"].get("actual_switch_utility", 0) < 0) for x in switches)
    return {"positive_opportunities": len(opportunities), "final_positive_capture": capture, "stay_on_b0_despite_opportunity": stay, "harmful_switch_rate": harmful}


def _markdown(primary, paired, mechanism, decision):
    lines = ["# V10 / SRR — результаты", "", "V10 является development-экспериментом на пяти ранее использованных cells.", "", "## Основные метрики", "", "| Cell | S0 F1 | S1 H2 F1 | S2 SRR F1 | ΔS2−S1 | 95% CI | S2 Hit@1 | S2 HN@1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in primary:
        ci = paired[cell]["S2-S1"]
        lines.append(f"| {cell} | {primary[cell]['S0']['token_f1']:.3f} | {primary[cell]['S1']['token_f1']:.3f} | {primary[cell]['S2']['token_f1']:.3f} | {ci['delta']:+.3f} | [{ci['lcb95']:+.3f}, {ci['ucb95']:+.3f}] | {100*primary[cell]['S2']['hit@1']:.2f}% | {100*primary[cell]['S2']['hn@1']:.2f}% |")
    lines += ["", "## Механизм", "", "| Cell | H2 positive capture | SRR positive capture | Δ | SRR harmful switches | Stay B0 при opportunity |", "|---|---:|---:|---:|---:|---:|"]
    for cell in primary:
        h2, s2 = mechanism[cell]["S1"], mechanism[cell]["S2"]
        lines.append(f"| {cell} | {100*h2['final_positive_capture']:.2f}% | {100*s2['final_positive_capture']:.2f}% | {100*(s2['final_positive_capture']-h2['final_positive_capture']):+.2f} pp | {100*s2['harmful_switch_rate']:.2f}% | {100*s2['stay_on_b0_despite_opportunity']:.2f}% |")
    lines += ["", "## Решение по протоколу", "", f"**{decision['label']}**", "", json.dumps(decision, ensure_ascii=False, indent=2), ""]
    return "\n".join(lines)


def analyze_run(root: Path, bootstrap: int, progress):
    root = Path(root); progress.stage("analysis", "aggregate seeds by question, paired document bootstrap, module and bottleneck reports")
    cells = sorted(path.name for path in (root / "predictions/S2").iterdir() if path.is_dir())
    data = {cell: {system: _load(root, system, cell) for system in SYSTEMS} for cell in cells}
    primary = {cell: {system: _primary(rows) for system, rows in systems.items()} for cell, systems in data.items()}
    paired = {cell: {
        "S1-S0": _bootstrap(data[cell]["S1"], data[cell]["S0"], bootstrap),
        "S2-S0": _bootstrap(data[cell]["S2"], data[cell]["S0"], bootstrap),
        "S2-S1": _bootstrap(data[cell]["S2"], data[cell]["S1"], bootstrap),
        **{f"{system}-S1": _bootstrap(data[cell][system], data[cell]["S1"], bootstrap) for system in ("A0", "A1", "A2")},
    } for cell in cells}
    switch = {}; rank = {}; answer = {}; evidence = {}
    for cell in cells: switch[cell], rank[cell], answer[cell], evidence[cell] = _module_diagnostics(data[cell]["S2"])
    mechanism = {cell: {system: _mechanism(data[cell][system]) for system in ("S1", "S2")} for cell in cells}
    # Equal family weights; SciER contributes the mean of its two chunking cells.
    global_effect = _global_bootstrap(data, cells, bootstrap)
    bottleneck_cells = [x for x in ("scier_384", "scier_128", "russerrc", "scimdix") if x in cells]
    mechanism_passes = [x for x in bottleneck_cells if mechanism[x]["S2"]["final_positive_capture"] - mechanism[x]["S1"]["final_positive_capture"] >= 0.10 and mechanism[x]["S2"]["harmful_switch_rate"] - mechanism[x]["S1"]["harmful_switch_rate"] <= 0.02 and primary[x]["S2"]["hn@1"] - primary[x]["S0"]["hn@1"] <= 0.01]
    criteria = {
        "mechanism_success": len(mechanism_passes) >= 3,
        "overall_quality": global_effect["family_balanced_delta_S2_S1"] >= 0.5 and all(paired[x]["S2-S1"]["delta"] >= -0.25 for x in cells),
        "uncertainty_guard": all(paired[x]["S2-S1"]["lcb95"] > -0.5 for x in cells),
        "evidence_safety": all(primary[x]["S2"]["hit@1"] - primary[x]["S0"]["hit@1"] >= -0.005 and primary[x]["S2"]["hn@1"] - primary[x]["S0"]["hn@1"] <= 0.01 for x in cells),
        "non_degeneracy": all(
            sum(sum(row["evaluation"]["changed_top1"] for row in data[cell]["S2"] if row["runtime"]["question_id"] == qid) >= 2
                for qid in {row["runtime"]["question_id"] for row in data[cell]["S2"]})
            >= max(3, math.ceil(primary[cell]["S2"]["n_questions"] * .01))
            for cell in cells
        ),
    }
    decision = {"label": "V10_SRR_DEVELOPMENT_CANDIDATE" if all(criteria.values()) else "V10_SRR_DOES_NOT_REPLACE_H2", "criteria": criteria, "mechanism_cells_passing": mechanism_passes, "development_not_independent_confirmation": True}
    fold_stability = {cell: {f"fold_{fold}": {system: _primary([x for x in data[cell][system] if x["runtime"]["outer_fold"] == fold])["token_f1"] for system in ("S0", "S1", "S2")} for fold in range(5)} for cell in cells}
    seed_stability = {cell: {f"seed_{seed}": {system: _primary([x for x in data[cell][system] if x["runtime"]["seed"] == seed])["token_f1"] for system in ("S0", "S1", "S2")} for seed in (42, 43, 44)} for cell in cells}
    aspect = {cell: {aspect: _primary([x for x in data[cell]["S2"] if x["runtime"].get("canonical_aspect") == aspect]) for aspect in sorted({x["runtime"].get("canonical_aspect") for x in data[cell]["S2"]})} for cell in cells}
    k_analysis = {cell: {str(k): _primary([x for x in data[cell]["S2"] if len(x["runtime"]["candidate_ids"]) == k]) for k in sorted({len(x["runtime"]["candidate_ids"]) for x in data[cell]["S2"]})} for cell in cells}
    bottleneck = {}; b0_rank = {}
    for cell in cells:
        s0 = primary[cell]["S0"]["token_f1"]; s1 = primary[cell]["S1"]["token_f1"]; s2 = primary[cell]["S2"]["token_f1"]
        oracle = _mean(max(map(float, row["evaluation"]["candidate_f1"].values())) for row in data[cell]["S2"])
        denominator = oracle - s0
        bottleneck[cell] = {
            "hit@k": primary[cell]["S2"]["hit@k"], "qa_oracle_f1": oracle,
            "available_oracle_gain": denominator,
            "h2_used_oracle_gain_pct": 100 * (s1 - s0) / denominator if denominator > 1e-12 else None,
            "srr_used_oracle_gain_pct": 100 * (s2 - s0) / denominator if denominator > 1e-12 else None,
            "h2_positive_capture": mechanism[cell]["S1"]["final_positive_capture"],
            "srr_positive_capture": mechanism[cell]["S2"]["final_positive_capture"],
            "h2_harmful_switch_rate": mechanism[cell]["S1"]["harmful_switch_rate"],
            "srr_harmful_switch_rate": mechanism[cell]["S2"]["harmful_switch_rate"],
        }
        s1_by = {(x["runtime"]["question_id"], x["runtime"]["seed"]): x for x in data[cell]["S1"]}
        buckets = {}
        for rank_value in (2, 3, 4, 5):
            selected = [x for x in data[cell]["S2"] if x["evaluation"]["positive_opportunity"] and x["evaluation"]["oracle_rank"] == rank_value]
            buckets[str(rank_value)] = {
                "n": len(selected),
                "h2_capture": _mean(
                    s1_by[(x["runtime"]["question_id"], x["runtime"]["seed"])]["evaluation"]["actual_switch_utility"] > 0
                    for x in selected
                ),
                "srr_capture": _mean(x["evaluation"]["positive_capture"] for x in selected),
            }
        b0_rank[cell] = buckets
    for name, value in {
        "primary_metrics.json": primary, "paired_bootstrap.json": paired,
        "global_family_bootstrap.json": global_effect, "switch_metrics.json": switch,
        "rank_metrics.json": rank, "answer_risk_metrics.json": answer,
        "evidence_risk_metrics.json": evidence, "positive_capture.json": mechanism,
        "stay_on_b0.json": mechanism, "harmful_switches.json": mechanism,
        "oracle_capture.json": rank, "fold_stability.json": fold_stability,
        "bottleneck_analysis.json": bottleneck, "b0_rank_analysis.json": {"NOT_USED_FOR_MODEL_OR_POLICY": True, "data": b0_rank},
        "seed_stability.json": seed_stability, "aspect_analysis.json": {"NOT_USED_FOR_MODEL_OR_POLICY": True, "data": aspect},
        "k_analysis.json": {"NOT_USED_FOR_MODEL_OR_POLICY": True, "data": k_analysis},
        "development_decision.json": decision,
    }.items(): write_json(root / "analysis" / name, value)
    report = _markdown(primary, paired, mechanism, decision)
    write_text(root / "reports/v10_srr_results.md", report)
    write_text(root / "reports/v10_module_analysis.md", "# V10 module diagnostics\n\n```json\n" + json.dumps({"switch": switch, "rank": rank, "answer_risk": answer, "evidence_risk": evidence}, ensure_ascii=False, indent=2) + "\n```\n")
    write_text(root / "reports/reproducibility.md", "# Reproducibility\n\nAll outer predictions are sealed. Inner folds are grouped by canonical document and shared across modules/seeds. Model and feature hashes are stored per prediction. Historical V1–V9 paths are read-only.\n")
    progress.log("analysis complete", cells=len(cells), decision=decision["label"])
