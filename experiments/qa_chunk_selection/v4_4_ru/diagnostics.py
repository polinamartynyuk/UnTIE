"""Post-EVAL diagnostics; never called by the feature-only router."""
from collections import defaultdict
import numpy as np


def rates(rows):
    n = len(rows)
    used = sum(r["reader_only_used"] for r in rows)
    return {"n": n, "routing_available_rate": sum(r["routing_available"] for r in rows) / max(n, 1),
            "reader_only_used_rate": used / max(n, 1), "fallback_rate": (n - used) / max(n, 1)}


def grouped_rates(rows):
    result = {"all": rates(rows)}
    for field in ("cv_run", "seed", "aspect"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[field] = {key: rates(value) for key, value in groups.items()}
    return result


def routing_audit(rows):
    disagreements = [r for r in rows if r["v6_chunk_id"] != r["reader_chunk_id"]]
    switched = [r for r in disagreements if r["reader_only_used"]]
    better = [r for r in disagreements if r["reader_f1"] > r["v6_f1"] + 1e-9]
    good = [r for r in switched if r["reader_f1"] > r["v6_f1"] + 1e-9]
    bad = [r for r in switched if r["reader_f1"] < r["v6_f1"] - 1e-9]
    return {"n_disagreement": len(disagreements), "n_reader_switch_on_disagreement": len(switched),
            "reader_better": len(better),
            "v6_better": sum(r["reader_f1"] < r["v6_f1"] - 1e-9 for r in disagreements),
            "tie": sum(abs(r["reader_f1"] - r["v6_f1"]) <= 1e-9 for r in disagreements),
            "good_switch": len(good), "bad_switch": len(bad),
            "unnecessary_switch": len(switched) - len(good) - len(bad),
            "missed_opportunity": len(better) - len(good),
            "routing_precision": len(good) / len(switched) if switched else None,
            "routing_recall": len(good) / len(better) if better else None}


def confidence_bins(rows, field="consensus_conf"):
    finite = [r for r in rows if r.get(field) is not None and np.isfinite(r[field])]
    if not finite:
        return {"bins": [], "monotone_reader_win_rate": None}
    edges = np.quantile([r[field] for r in finite], [.2, .4, .6, .8])
    bins = []
    for index in range(5):
        group = [r for r in finite if int(np.searchsorted(edges, r[field], side="right")) == index]
        dis = [r for r in group if r["v6_chunk_id"] != r["reader_chunk_id"]]
        delta = [r["reader_f1"] - r["v6_f1"] for r in dis]
        bins.append({"bin": f"{index*20}-{(index+1)*20}%", "n": len(group), "n_disagreements": len(dis),
                     "reader_better_rate": float(np.mean(np.array(delta) > 1e-9)) if delta else None,
                     "v6_better_rate": float(np.mean(np.array(delta) < -1e-9)) if delta else None,
                     "mean_delta_f1": float(np.mean(delta)) if delta else None})
    values = [b["reader_better_rate"] for b in bins if b["reader_better_rate"] is not None]
    return {"edges": edges.tolist(), "bins": bins, "ties": "equal confidence stays in one bin",
            "monotone_reader_win_rate": all(a <= b for a, b in zip(values, values[1:])) if len(values) > 1 else None,
            "diagnostic_only": True}


def error_bucket(row):
    if row["v6_chunk_id"] == row["reader_chunk_id"]:
        return None
    delta = row["reader_f1"] - row["v6_f1"]
    if row["reader_only_used"]:
        return "good_switch" if delta > 1e-9 else "bad_switch" if delta < -1e-9 else "unnecessary_switch"
    return "missed_opportunity" if delta > 1e-9 else None
