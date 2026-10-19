"""V6 quality, safety, universality and stability analysis."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.qa_chunk_selection.progress import ProgressLogger

from .contract import FAMILIES, VARIANTS
from .io_guard import read_jsonl, write_json, write_text


def _summary(rows):
    def mean(path):
        return float(np.mean([float(row[path[0]][path[1]]) for row in rows])) if rows else 0.0
    return {"rows": len(rows), "questions": len({r["runtime"]["question_id"] for r in rows}),
            "token_f1": mean(("evaluation", "token_f1")), "hit@1": mean(("evaluation", "hit@1")),
            "hn@1": mean(("evaluation", "hn@1")), "mrr_selected_first": mean(("evaluation", "mrr_selected_first")),
            "switch_rate": mean(("evaluation", "changed_top1")), "routing_rate": mean(("runtime", "gate_decision")),
            "qa_forwards": mean(("runtime", "qa_forwards")), "q_a_ranker_forwards": mean(("runtime", "q_a_ranker_forwards"))}


def _paired(base, other, bootstrap, seed=42):
    left = {(r["runtime"]["question_id"], r["runtime"]["seed"]): r for r in base}
    docs = defaultdict(list)
    for row in other:
        key = (row["runtime"]["question_id"], row["runtime"]["seed"])
        if key in left:
            docs[row["runtime"]["canonical_document_id"]].append(row["evaluation"]["token_f1"] - left[key]["evaluation"]["token_f1"])
    identifiers = sorted(docs); observed = float(np.mean([v for doc in identifiers for v in docs[doc]])) if identifiers else 0.0
    rng = np.random.default_rng(seed); samples = []
    for _ in range(bootstrap):
        draw = rng.choice(identifiers, len(identifiers), replace=True) if identifiers else []
        samples.append(float(np.mean([v for doc in draw for v in docs[doc]])) if len(draw) else 0.0)
    return {"delta_token_f1": observed, "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)), "cluster_unit": "canonical_document_id", "documents": len(identifiers)}


def _load(root: Path):
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in root.glob("predictions/*/*/*/outer_fold_*/seed_*/*.jsonl"):
        parts = path.relative_to(root / "predictions").parts
        mode, scope, cell = parts[:3]; values[mode][cell][path.stem].extend(read_jsonl(path))
    return values


def feature_columns_for_report(variant: str):
    from .contract import feature_columns
    return feature_columns(variant)


def analyze_run(root: Path, bootstrap: int, progress=None):
    root = Path(root); progress = progress or ProgressLogger(root / "progress.jsonl")
    progress.stage("analysis", "quality, safety, universality, stability and report")
    data = _load(root); primary = {}; comparisons = {}; stability = {}; switches = {}
    for mode, cells in data.items():
        primary[mode] = {}; comparisons[mode] = {}; stability[mode] = {}; switches[mode] = {}
        for cell, systems in cells.items():
            if "S0" not in systems: continue
            primary[mode][cell] = {name: _summary(rows) for name, rows in systems.items()}
            comparisons[mode][cell] = {name: _paired(systems["S0"], rows, bootstrap) for name, rows in systems.items() if name != "S0"}
            stability[mode][cell] = {}
            switches[mode][cell] = {}
            for name, rows in systems.items():
                stability[mode][cell][name] = {
                    "by_seed": {str(seed): _summary([r for r in rows if int(r["runtime"]["seed"]) == seed]) for seed in sorted({r["runtime"]["seed"] for r in rows})},
                    "by_fold": {str(fold): _summary([r for r in rows if int(r["runtime"]["outer_fold"]) == fold]) for fold in sorted({r["runtime"]["outer_fold"] for r in rows})},
                    "by_aspect": {aspect: _summary([r for r in rows if r["runtime"]["canonical_aspect"] == aspect]) for aspect in sorted({r["runtime"]["canonical_aspect"] for r in rows})},
                    "by_k": {str(k): _summary([r for r in rows if int(r["runtime"]["effective_k"]) == k]) for k in sorted({r["runtime"]["effective_k"] for r in rows})},
                }
                if name != "S0":
                    delta = [r["evaluation"]["token_f1"] - r["evaluation"]["baseline_token_f1"] for r in rows if r["evaluation"]["changed_top1"]]
                    switches[mode][cell][name] = {"good": sum(v > 0 for v in delta), "bad": sum(v < 0 for v in delta),
                                                        "neutral": sum(v == 0 for v in delta), "mean_delta_changed": float(np.mean(delta)) if delta else 0.0}
    write_json(root / "analysis/primary_metrics.json", primary); write_json(root / "analysis/paired_bootstrap.json", comparisons)
    write_json(root / "analysis/fold_seed_aspect_k_stability.json", stability); write_json(root / "analysis/switch_analysis.json", switches)
    universality = {}
    for mode in ("pooled", "local"):
        if mode not in comparisons: continue
        universality[mode] = {}
        for variant in ("S2", "S3", "S4"):
            deltas = {cell: comparisons[mode][cell][variant]["delta_token_f1"] for cell in comparisons[mode] if variant in comparisons[mode][cell]}
            scier = np.mean([v for cell, v in deltas.items() if cell.startswith("scier_")]) if any(c.startswith("scier_") for c in deltas) else 0.0
            family = [v for c, v in deltas.items() if not c.startswith("scier_")] + [float(scier)]
            evidence_safe = all(
                100 * (primary[mode][c][variant]["hit@1"] - primary[mode][c]["S0"]["hit@1"]) >= -0.5 - 1e-12
                and 100 * (primary[mode][c][variant]["hn@1"] - primary[mode][c]["S0"]["hn@1"]) <= 1.0 + 1e-12
                for c in deltas)
            universality[mode][variant] = {"cell_deltas": deltas, "global_family_equal_delta": float(np.mean(family)) if family else 0.0,
                                           "point_positive_all_cells": bool(deltas) and all(v > 0 for v in deltas.values()),
                                           "lcb_above_minus_0_5_all": bool(deltas) and all(comparisons[mode][c][variant]["ci_low"] > -0.5 for c in deltas),
                                           "evidence_safe_all_cells": evidence_safe}
            universality[mode][variant]["eligible_for_freeze"] = (universality[mode][variant]["point_positive_all_cells"]
                and universality[mode][variant]["lcb_above_minus_0_5_all"] and evidence_safe)
    write_json(root / "analysis/universality_summary.json", universality)
    eligible = []
    for mode in ("pooled", "local"):
        for variant, row in universality.get(mode, {}).items():
            if row["eligible_for_freeze"]:
                cost = float(np.mean([primary[mode][cell][variant]["qa_forwards"] for cell in primary[mode]]))
                eligible.append({"mode": mode, "variant": variant, "global_delta": row["global_family_equal_delta"], "qa_forwards": cost})
        if eligible: break
    selected = max(eligible, key=lambda row: (row["global_delta"], -row["qa_forwards"], -len(feature_columns_for_report(row["variant"])))) if eligible else None
    write_json(root / "analysis/selected_development_candidate.json", {"selected": selected, "eligible_candidates": eligible,
               "status": "GO_TO_FREEZE" if selected else "NO_UNIVERSAL_CANDIDATE", "estimate": "DEVELOPMENT"})
    lines = ["# V6 / UAR-G — development results", "", "Outer CV is development evidence; it is not an untouched confirmation.", "",
             "| Mode | Cell | System | Token F1 | Δ vs S0 | 95% CI | Hit@1 | HN@1 | QA forwards | Routing |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in sorted(primary):
        for cell in sorted(primary[mode]):
            for name in sorted(primary[mode][cell]):
                value = primary[mode][cell][name]; comp = comparisons[mode][cell].get(name, {"delta_token_f1": 0, "ci_low": 0, "ci_high": 0})
                lines.append(f"| {mode} | {cell} | {name} | {value['token_f1']:.4f} | {comp['delta_token_f1']:+.4f} | [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}] | {value['hit@1']:.4f} | {value['hn@1']:.4f} | {value['qa_forwards']:.3f} | {value['routing_rate']:.1%} |")
    lines += ["", "S1 is the V5 evidence-target interaction control. S2 uses positive Token-F1 utility. S3 adds the pre-QA gate. S4 adds historical source and LEX signals. S5 is skipped unless exact historical SEM and QA-confidence can be replayed on all cells."]
    write_text(root / "reports/results_v6.md", "\n".join(lines) + "\n")
    write_text(root / "reports/reproducibility.md", "# V6 reproducibility\n\nSee manifest.json, frozen_contract.yaml, audit/, meta_models/, predictions/, and analysis/. Historical V4/V5 inputs are read-only and are verified by before/after hashes.\n")
    progress.log("analysis complete", modes=sorted(primary))
    return primary


def parser():
    p = argparse.ArgumentParser(description="Analyze a completed V6 development run")
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--cluster-key", default="document_id"); p.add_argument("--include-resources", action="store_true")
    p.add_argument("--include-switch-analysis", action="store_true"); p.add_argument("--include-interactions", action="store_true")
    p.add_argument("--include-universality", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv); analyze_run(args.run_dir, args.bootstrap); return 0


if __name__ == "__main__": raise SystemExit(main())

