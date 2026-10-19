"""Gold-free routing and calibration: inputs deliberately contain no labels/metrics."""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from experiments.qa_chunk_selection.v4_3.ranking import routing_available
from .contract import MIN_ROUTING_CAL_ROWS, MIN_SUPPORT_CAL_ROWS


@dataclass(frozen=True)
class RouteFeature:
    question_id: str
    document_id: str
    seed: int
    role: str
    confidence: float | None
    available: bool
    reader_id: int
    v6_id: int
    support_rel: float | None = None


def available(pack, reader_id):
    conf = pack.get("consensus_conf")
    return bool(conf is not None and math.isfinite(float(conf)) and routing_available(
        adapt_answers_ok=int(pack["n_valid"]) >= 4,
        selected_cid=reader_id, best_triple=pack["best_triple"], consensus_conf=conf,
    ))


def calibrate(rows: Sequence[RouteFeature], *, support=False, minimum=None):
    minimum = minimum if minimum is not None else (MIN_SUPPORT_CAL_ROWS if support else MIN_ROUTING_CAL_ROWS)
    eligible = [r for r in rows if r.role in {"train", "cal"} and r.available
                and (not support or r.reader_id != r.v6_id)]
    values, keys = [], []
    for row in eligible:
        value = row.support_rel if support else row.confidence
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
            keys.append([row.question_id, row.seed])
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("Duplicate calibration (question_id, seed)")
    enabled = len(values) >= minimum
    return {
        "enabled": enabled, "tau": float(np.quantile(values, .2 if support else .7)) if enabled else None,
        "quantile": .2 if support else .7, "minimum": minimum,
        "n_threshold_source_rows": len(values), "source_keys": keys,
        "source_roles": ["train", "cal"], "uses_gold": False,
        "reason": "ok" if enabled else "insufficient calibration rows; exact fallback",
    }


def use_reader(feature: RouteFeature, threshold) -> bool:
    return bool(threshold["enabled"] and feature.available and feature.confidence is not None
                and math.isfinite(feature.confidence) and feature.confidence >= threshold["tau"])


def support_veto(feature: RouteFeature, threshold, *, switched: bool) -> bool:
    return bool(switched and threshold["enabled"] and feature.support_rel is not None
                and math.isfinite(feature.support_rel) and feature.support_rel < threshold["tau"])


def choose_prediction(v6, reader, feature, threshold):
    """Copy the chosen component exactly, including its full isolated ranking."""
    used = use_reader(feature, threshold)
    return dict(reader if used else v6), used
