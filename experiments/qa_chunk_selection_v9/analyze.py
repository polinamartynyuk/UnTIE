"""Primary metrics, paired uncertainty, selector diagnostics and V9 decision."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.progress import ProgressLogger
from experiments.qa_chunk_selection_v7.analyze import _activity, _paired, _secondary_answer_metrics, _summary, _switch
from .contract import CELLS, CONFIGS, LANGUAGES
from .io_guard import read_jsonl, write_json, write_text


def _load(root: Path) -> dict:
    data = defaultdict(lambda: defaultdict(list))
    for path in Path(root).glob("predictions/*/*/outer_fold_*/seed_*.jsonl"):
        system, cell = path.relative_to(Path(root) / "predictions").parts[:2]
        data[cell][system].extend(read_jsonl(path))
    return data


def _stable_difference(current: list[dict], base: list[dict]) -> dict:
    def index(rows):
        return {
            (str(x["runtime"]["question_id"]), int(x["runtime"]["seed"])): int(x["runtime"]["selected_chunk_id"])
            for x in rows
        }
    a, b = index(current), index(base)
    if set(a) != set(b):
        raise AssertionError("component activity coverage mismatch")
    grouped = defaultdict(list)
    for (question, seed), selected in a.items():
        grouped[question].append(selected != b[(question, seed)])
    stable = sum(sum(values) >= 2 for values in grouped.values())
    required = max(3, math.ceil(0.01 * len(grouped)))
    return {
        "questions": len(grouped),
        "stably_changed_vs_s1": stable,
        "required": required,
        "non_degenerate": stable >= required,
    }


def _question_delta(current: list[dict], base: list[dict]) -> dict[str, dict]:
    def grouped(rows):
        out = defaultdict(list)
        for row in rows:
            out[str(row["runtime"]["question_id"])].append(row)
        return out
    left, right = grouped(current), grouped(base)
    if set(left) != set(right):
        raise AssertionError("paired question coverage mismatch")
    result = {}
    for question in left:
        result[question] = {
            "delta": float(np.mean([x["evaluation"]["token_f1"] for x in left[question]]))
            - float(np.mean([x["evaluation"]["token_f1"] for x in right[question]])),
            "document": str(left[question][0]["runtime"]["canonical_document_id"]),
        }
    return result


def _family_bootstrap(data: dict, current: str, base: str, iterations: int) -> dict:
    by_cell = {cell: _question_delta(systems[current], systems[base]) for cell, systems in data.items()}
    cell_point = {cell: float(np.mean([x["delta"] for x in rows.values()])) for cell, rows in by_cell.items()}
    scier_cells = [cell for cell in cell_point if cell.startswith("scier_")]
    scier = float(np.mean([cell_point[x] for x in scier_cells])) if scier_cells else None
    family_point = [value for cell, value in cell_point.items() if not cell.startswith("scier_")]
    if scier is not None:
        family_point.append(scier)
    point = float(np.mean(family_point))
    rng = np.random.default_rng(42)
    samples = []
    ordinary = [cell for cell in by_cell if not cell.startswith("scier_")]
    doc_values = {}
    for cell, questions in by_cell.items():
        grouped = defaultdict(list)
        for value in questions.values():
            grouped[value["document"]].append(value["delta"])
        doc_values[cell] = dict(grouped)
    for _ in range(iterations):
        family_effects = []
        for cell in ordinary:
            docs = sorted(doc_values[cell])
            picked = rng.choice(docs, size=len(docs), replace=True)
            values = [value for doc in picked for value in doc_values[cell][str(doc)]]
            family_effects.append(float(np.mean(values)))
        if scier_cells:
            union = sorted({doc for cell in scier_cells for doc in doc_values[cell]})
            picked = rng.choice(union, size=len(union), replace=True)
            counts = Counter(str(x) for x in picked)
            scier_effects = []
            for cell in scier_cells:
                weighted = [
                    value
                    for doc, values in doc_values[cell].items()
                    for _ in range(counts.get(doc, 0))
                    for value in values
                ]
                if weighted:
                    scier_effects.append(float(np.mean(weighted)))
            family_effects.append(float(np.mean(scier_effects)))
        samples.append(float(np.mean(family_effects)))
    return {
        "current": current,
        "base": base,
        "cell_deltas": cell_point,
        "scier_family_delta": scier,
        "family_balanced_delta": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "iterations": iterations,
        "seed": 42,
        "scier_coupled_document_resampling": bool(scier_cells),
    }


def _selection_summary(rows: list[dict]) -> dict:
    counts = Counter(x["selected_config"] for x in rows)
    probabilities = [count / len(rows) for count in counts.values()] if rows else []
    entropy = -sum(p * math.log(p, 2) for p in probabilities if p > 0)
    return {
        "selection_events": len(rows),
        "frequency": {config: counts.get(config, 0) for config in CONFIGS},
        "entropy_bits": entropy,
    }


def _selection_analysis(root: Path, data: dict) -> tuple[dict, dict]:
    manifests = [
        json.loads(path.read_text())
        for path in sorted((root / "selections/manifests").glob("*/outer_fold_*.json"))
    ]
    analysis = {"by_cell": {}, "by_language": {}, "overall": _selection_summary(manifests)}
    regrets = []
    mapping = {"C0": "S1", "CA": "D-A", "CP": "D-P", "CAP": "D-AP"}
    for cell in sorted({x["cell"] for x in manifests}):
        rows = [x for x in manifests if x["cell"] == cell]
        analysis["by_cell"][cell] = {**_selection_summary(rows), "rows": rows}
        for manifest in rows:
            fold = int(manifest["outer_fold"])
            scores = {}
            for config, system in mapping.items():
                values = [x for x in data[cell][system] if int(x["runtime"]["outer_fold"]) == fold]
                scores[config] = float(np.mean([x["evaluation"]["token_f1"] for x in values]))
            selected = manifest["selected_config"]
            best = max(CONFIGS, key=lambda x: (scores[x], -CONFIGS.index(x)))
            regrets.append({
                "cell": cell,
                "outer_fold": fold,
                "selected_on_train": selected,
                "oracle_best_on_outer_eval_diagnostic_only": best,
                "selected_eval_f1": scores[selected],
                "oracle_eval_f1": scores[best],
                "selector_regret": scores[best] - scores[selected],
                "oracle_eligible_for_selection": False,
            })
    for language in sorted(set(LANGUAGES.values())):
        rows = [x for x in manifests if LANGUAGES[x["cell"]] == language]
        analysis["by_language"][language] = _selection_summary(rows)
    return analysis, {"rows": regrets, "oracle_is_diagnostic_only": True}


def _coefficient_analysis(root: Path) -> dict:
    rows = []
    for path in sorted((root / "models").glob("S[12]/*/outer_fold_*/seed_*/selection.json")):
        payload = json.loads(path.read_text())
        for feature, value in payload["selection"].get("standardized_coefficients", {}).items():
            rows.append({
                "path": str(path.relative_to(root)),
                "system": path.parts[-5],
                "cell": path.parts[-4],
                "config": payload["config_id"],
                "feature": feature,
                "value": float(value),
            })
    summary = {}
    for feature in sorted({x["feature"] for x in rows}):
        values = np.asarray([x["value"] for x in rows if x["feature"] == feature], float)
        summary[feature] = {
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "positive_rate": float(np.mean(values > 0)),
            "n": len(values),
        }
    return {"rows": rows, "summary": summary, "coefficients_are_standardized": True}


def analyze_run(root: Path, bootstrap: int = 5000, progress=None) -> dict:
    root = Path(root)
    progress = progress or ProgressLogger(root / "progress.jsonl")
    progress.stage("analysis", "paired bootstrap, family-aware global CI, selector regret and frozen decision")
    data = _load(root)
    if set(data) != set(CELLS) and not json.loads((root / "manifest.json").read_text())["config"].get("smoke"):
        raise RuntimeError(f"incomplete V9 cells: {sorted(data)}")

    primary, paired, activity, switches = {}, {}, {}, {}
    folds, seeds, aspects, ks = {}, {}, {}, {}
    for cell, systems in data.items():
        primary[cell] = {system: _summary(rows) for system, rows in systems.items()}
        paired[cell] = {
            "S1-S0": _paired(systems["S0"], systems["S1"], bootstrap),
            "S2-S0": _paired(systems["S0"], systems["S2"], bootstrap),
            "S2-S1": _paired(systems["S1"], systems["S2"], bootstrap),
        }
        for system in ("D-A", "D-P", "D-AP"):
            paired[cell][f"{system}-S1"] = _paired(systems["S1"], systems[system], bootstrap)
        activity[cell] = {system: _activity(rows) for system, rows in systems.items() if system != "S0"}
        activity[cell]["S2-S1"] = _stable_difference(systems["S2"], systems["S1"])
        switches[cell] = {system: _switch(rows) for system, rows in systems.items() if system != "S0"}
        folds[cell] = {
            system: {
                str(fold): _summary([x for x in rows if x["runtime"]["outer_fold"] == fold])
                for fold in sorted({int(x["runtime"]["outer_fold"]) for x in rows})
            }
            for system, rows in systems.items()
        }
        seeds[cell] = {
            system: {str(seed): _summary([x for x in rows if x["runtime"]["seed"] == seed]) for seed in sorted({x["runtime"]["seed"] for x in rows})}
            for system, rows in systems.items()
        }
        aspects[cell] = {
            system: {aspect: _summary([x for x in rows if x["runtime"]["canonical_aspect"] == aspect]) for aspect in sorted({x["runtime"]["canonical_aspect"] for x in rows})}
            for system, rows in systems.items()
        }
        ks[cell] = {
            system: {str(k): _summary([x for x in rows if x["runtime"]["effective_k"] == k]) for k in sorted({x["runtime"]["effective_k"] for x in rows})}
            for system, rows in systems.items()
        }

    write_json(root / "analysis/primary_metrics.json", primary)
    write_json(root / "analysis/paired_bootstrap.json", paired)
    write_json(root / "analysis/switch_analysis.json", switches)
    write_json(root / "analysis/fold_stability.json", folds)
    write_json(root / "analysis/seed_stability.json", seeds)
    write_json(root / "analysis/aspect_analysis.json", aspects)
    write_json(root / "analysis/k_analysis.json", ks)
    write_json(root / "analysis/non_degeneracy.json", activity)
    write_json(root / "analysis/secondary_answer_metrics.json", _secondary_answer_metrics({"local": data}))
    write_json(root / "analysis/coefficient_analysis.json", _coefficient_analysis(root))

    selection_analysis, regret = _selection_analysis(root, data)
    write_json(root / "analysis/component_selection_analysis.json", selection_analysis)
    write_json(root / "analysis/selector_regret.json", regret)
    write_json(root / "selections/selection_summary.json", selection_analysis)

    global_bootstrap = {
        "S1-S0": _family_bootstrap(data, "S1", "S0", bootstrap),
        "S2-S0": _family_bootstrap(data, "S2", "S0", bootstrap),
        "S2-S1": _family_bootstrap(data, "S2", "S1", bootstrap),
    }
    write_json(root / "analysis/global_family_bootstrap.json", global_bootstrap)

    cells = sorted(data)
    def universal(system: str, contrast: str) -> dict:
        deltas = {cell: paired[cell][contrast]["delta_token_f1"] for cell in cells}
        lcb = {cell: paired[cell][contrast]["ci_low"] for cell in cells}
        evidence = {
            cell: {
                "delta_hit1_pp": 100 * (primary[cell][system]["hit@1"] - primary[cell]["S0"]["hit@1"]),
                "delta_hn1_pp": 100 * (primary[cell][system]["hn@1"] - primary[cell]["S0"]["hn@1"]),
            }
            for cell in cells
        }
        safe = all(x["delta_hit1_pp"] >= -0.5 - 1e-12 and x["delta_hn1_pp"] <= 1.0 + 1e-12 for x in evidence.values())
        nondeg = all(activity[cell][system]["non_degenerate"] for cell in cells)
        return {
            "cell_deltas": deltas,
            "cell_lcb95": lcb,
            "point_positive_all": len(cells) == 5 and all(x > 0 for x in deltas.values()),
            "lcb_above_minus_0_5_all": len(cells) == 5 and all(x > -0.5 for x in lcb.values()),
            "strict_lcb_positive_all": len(cells) == 5 and all(x > 0 for x in lcb.values()),
            "evidence": evidence,
            "evidence_safe_all": safe,
            "non_degenerate_all": nondeg,
            "pass": len(cells) == 5 and all(x > 0 for x in deltas.values()) and all(x > -0.5 for x in lcb.values()) and safe and nondeg,
        }

    s1 = universal("S1", "S1-S0")
    s2 = universal("S2", "S2-S0")
    s2s1_delta = {cell: paired[cell]["S2-S1"]["delta_token_f1"] for cell in cells}
    s2s1_lcb = {cell: paired[cell]["S2-S1"]["ci_low"] for cell in cells}
    incremental_evidence = {
        cell: {
            "delta_hit1_pp": 100 * (primary[cell]["S2"]["hit@1"] - primary[cell]["S1"]["hit@1"]),
            "delta_hn1_pp": 100 * (primary[cell]["S2"]["hn@1"] - primary[cell]["S1"]["hn@1"]),
        }
        for cell in cells
    }
    preservation = (
        len(cells) == 5
        and all(x >= -1e-12 for x in s2s1_delta.values())
        and all(x > -0.5 for x in s2s1_lcb.values())
        and all(x["delta_hit1_pp"] >= -0.25 - 1e-12 and x["delta_hn1_pp"] <= 0.5 + 1e-12 for x in incremental_evidence.values())
        and all(activity[cell]["S2"]["non_degenerate"] for cell in cells)
    )
    added_value = (
        global_bootstrap["S2-S1"]["family_balanced_delta"] >= 0.2 - 1e-12
        and sum(x > 0 for x in s2s1_delta.values()) >= 2
    )
    if s2["pass"] and preservation and added_value:
        final_candidate, status = "S2/H2-ACS", "DEVELOPMENT_CANDIDATE_SELECTED"
    elif s1["pass"]:
        final_candidate, status = "S1/FIXED-H2", "DEVELOPMENT_CANDIDATE_SELECTED"
    else:
        final_candidate, status = None, "NO_UNIVERSAL_DEVELOPMENT_CANDIDATE"
    decision = {
        "status": status,
        "estimate": "DEVELOPMENT",
        "S1_universal": s1,
        "S2_universal": s2,
        "S2_preservation_vs_S1": {
            "pass": preservation,
            "cell_deltas": s2s1_delta,
            "cell_lcb95": s2s1_lcb,
            "incremental_evidence": incremental_evidence,
        },
        "S2_added_value": {
            "pass": added_value,
            "family_balanced_delta": global_bootstrap["S2-S1"]["family_balanced_delta"],
            "strictly_positive_cells": sum(x > 0 for x in s2s1_delta.values()),
        },
        "final_candidate": final_candidate,
        "diagnostics_are_ineligible": ["D-A", "D-P", "D-AP"],
        "h2_replay_is_not_independent_confirmation": True,
    }
    write_json(root / "analysis/global_summary.json", global_bootstrap)
    write_json(root / "analysis/development_decision.json", decision)

    lines = [
        "# V9 / H2-ACS results", "",
        "LOCAL canonical development protocol. These are development estimates, not independent confirmation.", "",
        "## Primary systems", "",
        "| Cell | S0 F1 | S1 F1 | Δ S1-S0 | 95% CI | S2 F1 | Δ S2-S0 | 95% CI | Δ S2-S1 | 95% CI |",
        "|---|---:|---:|---:|---|---:|---:|---|---:|---|",
    ]
    for cell in cells:
        a, b, c = paired[cell]["S1-S0"], paired[cell]["S2-S0"], paired[cell]["S2-S1"]
        lines.append(
            f"| {cell} | {primary[cell]['S0']['token_f1']:.4f} | {primary[cell]['S1']['token_f1']:.4f} | "
            f"{a['delta_token_f1']:+.4f} | [{a['ci_low']:+.4f}; {a['ci_high']:+.4f}] | "
            f"{primary[cell]['S2']['token_f1']:.4f} | {b['delta_token_f1']:+.4f} | "
            f"[{b['ci_low']:+.4f}; {b['ci_high']:+.4f}] | {c['delta_token_f1']:+.4f} | "
            f"[{c['ci_low']:+.4f}; {c['ci_high']:+.4f}] |"
        )
    lines += [
        "", "## Fixed diagnostics", "",
        "These system-level deltas include configuration-specific tuning/refit and are not pure causal feature effects.", "",
        "| Cell | D-A vs S1 | D-P vs S1 | D-AP vs S1 |",
        "|---|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell} | {paired[cell]['D-A-S1']['delta_token_f1']:+.4f} | "
            f"{paired[cell]['D-P-S1']['delta_token_f1']:+.4f} | "
            f"{paired[cell]['D-AP-S1']['delta_token_f1']:+.4f} |"
        )
    lines += [
        "", "## Family-balanced effects", "",
        "| Contrast | Mean ΔF1 | 95% CI |", "|---|---:|---|",
    ]
    for contrast, values in global_bootstrap.items():
        lines.append(
            f"| {contrast} | {values['family_balanced_delta']:+.4f} | "
            f"[{values['ci_low']:+.4f}; {values['ci_high']:+.4f}] |"
        )
    lines += ["", "## Frozen decision", "", "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", ""]
    write_text(root / "reports/v9_results.md", "\n".join(lines))

    selection_lines = [
        "# V9 component selection", "",
        "One TRAIN-only configuration is selected per `(cell, outer_fold)` and shared by all model seeds.", "",
        "| Cell | Fold | Selected | ΔF1 vs C0 | Worst SELECT fold | Stable changes | Oracle diagnostic | Regret |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    regret_index = {(x["cell"], x["outer_fold"]): x for x in regret["rows"]}
    for cell, summary in selection_analysis["by_cell"].items():
        for manifest in summary["rows"]:
            selected = manifest["selected_config"]
            metrics = manifest["candidates"][selected]
            oracle = regret_index[(cell, int(manifest["outer_fold"]))]
            selection_lines.append(
                f"| {cell} | {manifest['outer_fold']} | {selected} | "
                f"{metrics['delta_f1_vs_c0']:+.4f} | {metrics['worst_selection_fold_delta']:+.4f} | "
                f"{metrics['stable_changed_questions']} | {oracle['oracle_best_on_outer_eval_diagnostic_only']} | "
                f"{oracle['selector_regret']:+.4f} |"
            )
    selection_lines += ["", "## Full machine-readable summary", "", "```json", json.dumps(selection_analysis, ensure_ascii=False, indent=2), "```", ""]
    write_text(root / "reports/component_selection.md", "\n".join(selection_lines))
    write_text(
        root / "reports/reproducibility.md",
        "# V9 reproducibility\n\n"
        "V1–V8 are read-only and hash-checked. Component selection uses seed-averaged TRAIN-only "
        "SELECT_OOF with four document-grouped folds. Outer EVAL is generated only after one "
        "configuration is frozen per `(cell, outer_fold)`. SciER cells use coupled document "
        "resampling in the global family bootstrap.\n",
    )
    progress.log("analysis complete", status=status, final_candidate=final_candidate)
    return decision

