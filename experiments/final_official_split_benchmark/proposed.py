"""Exact V7/V8/V9 H2-family meta policies over freshly produced features."""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import joblib

from experiments.qa_chunk_selection_v8.contract import H1_COLUMNS, H2_COLUMNS, H3_COLUMNS
from experiments.qa_chunk_selection_v8.features import hrr_rows, normalize_weights
from experiments.qa_chunk_selection_v8.models import apply_policy as apply_v8_policy, tune_policy
from experiments.qa_chunk_selection_v7.models import apply_policy as apply_v7_policy, tune_policy as tune_v7_policy
from experiments.qa_chunk_selection_v9.contract import CONFIG_COLUMNS, CONFIGS
from experiments.qa_chunk_selection_v9.selection import make_select_split_manifest, select_configuration, split_rows

from .io_guard import write_json

NO_FULLLEX_COLUMNS = tuple(x for x in H2_COLUMNS if "fulllex" not in x)
NO_PAIRLEX_COLUMNS = tuple(x for x in H2_COLUMNS if "pairlex" not in x)
POLICY_COLUMNS = {"P1": tuple(H1_COLUMNS), "P2": tuple(H2_COLUMNS), "P4": tuple(H3_COLUMNS),
                  "P2-NO-FULLLEX": NO_FULLLEX_COLUMNS, "P2-NO-PAIRLEX": NO_PAIRLEX_COLUMNS, "P2-NO-RISK": tuple(H2_COLUMNS)}


def fit_policy(rows: list[dict], system: str, seed: int, folds: int):
    values = copy.deepcopy(rows); normalize_weights(values, pooled=False)
    if system == "P1":
        return tune_v7_policy(values, "S2", seed, folds)
    return tune_policy(values, system, POLICY_COLUMNS[system], seed, folds, use_harm=system != "P2-NO-RISK")

def apply_labeled(rows, policy, *, use_harm=True):
    if hasattr(policy, "admission_model"):
        choices, predictions = apply_v7_policy(rows, policy)
        return choices, predictions
    return apply_v8_policy(rows, policy, use_harm=use_harm)


def save_policy(path: Path, policy, *, extra: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy.utility_model, path / "utility_model.joblib")
    joblib.dump(policy.harm_model, path / "harm_model.joblib")
    joblib.dump(policy, path / "policy.joblib")
    write_json(path / "selection.json", {"system": policy.system, "feature_columns": list(policy.columns), "selection": policy.selection, **(extra or {})})


def load_policy(path: Path):
    return joblib.load(Path(path) / "policy.joblib")


def runtime_alternatives(runtime: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in runtime:
        grouped[(row["dataset_cell"], row["question_id"])].append(row)
    out = []
    for values in grouped.values():
        values = sorted(values, key=lambda x: (x["b0_rank"], x["chunk_id"]))
        fake = {"baseline_f1": 0.0, "candidate_f1": {str(x["chunk_id"]): 0.0 for x in values},
                "label_tier": {str(x["chunk_id"]): "unresolved_for_training" for x in values},
                "hard_negative": {str(x["chunk_id"]): 0 for x in values}}
        out.extend(hrr_rows(values, fake))
    return out


def apply_runtime(runtime: list[dict], policy, *, use_harm: bool = True) -> dict[str, dict]:
    grouped_runtime = defaultdict(list)
    for row in runtime:
        grouped_runtime[str(row["question_id"])].append(row)
    rows = runtime_alternatives(runtime)
    if rows:
        choices, predicted = apply_labeled(rows, policy, use_harm=use_harm)
        utility, harm = predicted[0], predicted[1]
    else:
        choices, utility, harm = [], [], []
    predictions = {(r["question_id"], int(r["chunk_id"])): {"utility": float(u), "harm": float(h)} for r, u, h in zip(rows, utility, harm)}
    result = {}
    thresholds = {"tau_utility": policy.selection["tau_utility"], "tau_harm": policy.selection["tau_harm"]}
    for choice in choices:
        qid = str(choice["question_id"]); selected = int(choice["selected_chunk_id"])
        result[qid] = {"selected_chunk_id": selected, "candidate_scores": {str(cid): value for (question, cid), value in predictions.items() if question == qid},
                       "fold_policy": policy.selection["fold_policy"], "thresholds": thresholds}
    # A question with one canonical candidate has no alternative row for a
    # meta-policy to score. Its only valid decision is the B0 top-1 itself.
    for qid, values in grouped_runtime.items():
        if qid in result:
            continue
        top = min(values, key=lambda x: (int(x["b0_rank"]), int(x["chunk_id"])))
        result[qid] = {"selected_chunk_id": int(top["chunk_id"]), "candidate_scores": {},
                       "fold_policy": "B0_FALLBACK_NO_ALTERNATIVE", "thresholds": thresholds,
                       "fallback_reason": "NO_ALTERNATIVE_CANDIDATE"}
    return result


def _select_oof_prediction(rows: list[dict], policy, seed: int, held_fold: int, config: str) -> list[dict]:
    choices, _ = apply_v8_policy(rows, policy, use_harm=True)
    by_question = defaultdict(list)
    for row in rows: by_question[str(row["question_id"])].append(row)
    outputs = []
    for choice in choices:
        qid = str(choice["question_id"]); values = by_question[qid]; first = values[0]
        outputs.append({
            "runtime": {"question_id": qid, "seed": seed, "selected_chunk_id": int(choice["selected_chunk_id"]),
                        "canonical_document_id": first["canonical_document_id"], "select_fold": held_fold, "split_role": "SELECT_OOF", "config": config},
            "evaluation": {"token_f1": float(choice["f1"]), "baseline_token_f1": float(choice["base_f1"]),
                           "hit@1": int(choice["hit1"]), "baseline_hit@1": int(choice["base_hit1"]),
                           "hn@1": int(choice["hn1"]), "baseline_hn@1": int(choice["base_hn1"])},
        })
    return outputs


def select_acs(rows_by_seed: dict[int, list[dict]], seeds: tuple[int, ...], folds: int, root: Path, progress) -> dict:
    """Exact V9 selector: four document-grouped SELECT_OOF folds, then seed averaging."""
    reference = rows_by_seed[seeds[0]]
    manifest = make_select_split_manifest(reference); write_json(root / "select_split_manifest.json", manifest)
    predictions = {config: [] for config in CONFIGS}
    for seed in seeds:
        rows = rows_by_seed[seed]
        docs = {x["canonical_document_id"] for x in rows}
        if docs != {x["canonical_document_id"] for x in reference}:
            raise RuntimeError("ACS seed feature rows have different document universes")
        for held in range(4):
            train, valid = split_rows(rows, manifest, held)
            progress.log("H2-ACS SELECT_OOF", seed=seed, select_fold=held, train_rows=len(train), valid_rows=len(valid))
            c0 = tune_policy(copy.deepcopy(train), "C0", CONFIG_COLUMNS["C0"], seed, folds, True)
            ca = tune_policy(copy.deepcopy(train), "CA", CONFIG_COLUMNS["CA"], seed, folds, True)
            for config, policy in (("C0", c0), ("CA", ca), ("CP", c0), ("CAP", ca)):
                predictions[config].extend(_select_oof_prediction(valid, policy, seed, held, config))
    selection = select_configuration(predictions, seeds)
    write_json(root / "component_selection.json", selection)
    return selection


def fit_final_family(rows_by_seed: dict[int, list[dict]], seeds: tuple[int, ...], folds: int, root: Path, progress) -> dict:
    acs = select_acs(rows_by_seed, seeds, folds, root / "P3", progress)
    selected = acs["selected_config"]
    outputs = {"P3_selected_config": selected, "seeds": {}}
    for seed in seeds:
        rows = rows_by_seed[seed]
        outputs["seeds"][str(seed)] = {}
        for system in POLICY_COLUMNS:
            progress.log("final meta refit", system=system, seed=seed, rows=len(rows))
            policy = fit_policy(rows, system, seed, folds)
            save_policy(root / system / f"seed_{seed}", policy)
            outputs["seeds"][str(seed)][system] = policy.selection["fold_policy"]
        policy = tune_policy(copy.deepcopy(rows), "P3", CONFIG_COLUMNS[selected], seed, folds, True)
        save_policy(root / "P3" / f"seed_{seed}", policy, extra={"selected_config": selected, "selector": acs})
        outputs["seeds"][str(seed)]["P3"] = policy.selection["fold_policy"]
    write_json(root / "final_family_fit.json", outputs)
    return outputs
