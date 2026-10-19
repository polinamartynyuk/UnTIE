"""Fold-stability selection over gold-bearing source folds; outer EVAL is never read."""

import math


def metrics(keys, top_id, utility):
    rows = [utility[key][int(top_id(key))] for key in keys]
    return {"n": len(rows), "token_f1": sum(x["f1"] for x in rows) / max(len(rows), 1),
            "hit@1": sum(x["positive"] for x in rows) / max(len(rows), 1),
            "hn@1": sum(x["hn"] for x in rows) / max(len(rows), 1)}


def select_stable(keys, candidates, top_id_for, utility, fold_of, *, baseline_top,
                  epsilon_hn=.02, hit_tolerance=.005, require_stable=True, progress=None,
                  label="weights"):
    base = metrics(keys, baseline_top, utility)
    folds = sorted({fold_of[key] for key in keys})
    base_fold = {fold: metrics([key for key in keys if fold_of[key] == fold], baseline_top, utility)
                 for fold in folds}
    tried = []
    for index, params in enumerate(candidates, 1):
        fn = lambda key, p=params: top_id_for(key, p)
        value = metrics(keys, fn, utility)
        deltas = []
        for fold in folds:
            subset = [key for key in keys if fold_of[key] == fold]
            result = metrics(subset, fn, utility)
            deltas.append(result["token_f1"] - base_fold[fold]["token_f1"])
        row = {"params": dict(params), "metrics": value,
               "delta_f1": value["token_f1"] - base["token_f1"],
               "fold_deltas": deltas, "positive_folds": sum(x > 1e-9 for x in deltas),
               "worst_fold_delta": min(deltas) if deltas else 0.0}
        row["safe"] = (value["hn@1"] <= base["hn@1"] + epsilon_hn + 1e-12 and
                       value["hit@1"] >= base["hit@1"] - hit_tolerance - 1e-12)
        row["stable"] = bool(row["safe"] and row["delta_f1"] > 1e-9 and
                             row["positive_folds"] >= math.ceil(len(folds) / 2))
        tried.append(row)
        if progress:
            progress.count(index, len(candidates), item=label)
    eligible = [row for row in tried if row["stable"]] if require_stable else [row for row in tried if row["safe"]]
    selected = max(eligible, key=lambda row: (row["delta_f1"], row["positive_folds"],
                                              row["worst_fold_delta"],
                                              -sum(abs(float(x)) for x in row["params"].values()))) if eligible else None
    return {"baseline": base, "selected": selected, "fallback": selected is None,
            "criterion": "source-fold majority improvement + HN/Hit safety" if require_stable else "source mean F1 + HN/Hit safety",
            "source_folds": folds, "n_candidates": len(tried), "tried": tried}

