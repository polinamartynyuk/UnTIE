"""Hierarchical development gates. Diagnostics cannot enter the candidate list."""


def primary_gate(v6, asem, delta):
    checks = {
        "positive_delta": delta["mean_delta"] > 0,
        "development_ci_above_zero": delta["ci95_low"] > 0,
        "hn_ok": asem["hn@1"] <= v6["hn@1"] + .01 + 1e-12,
        "hit1_ok": asem["hit@1"] >= v6["hit@1"] - .005 - 1e-12,
    }
    return {"pass": all(checks.values()), "checks": checks, "development_delta": delta}


def select_candidate(v6, asem, alex, h1, delta):
    if not h1["pass"]:
        return {"preferred": "V6-K5", "h1": h1, "h2": {"tested": False, "reason": "H1 failed"}}
    superiority = (delta["mean_delta"] > 0 and delta["ci95_low"] > 0
                   and alex["hn@1"] <= v6["hn@1"] + .01 + 1e-12)
    noninferiority = (delta["ci95_low"] > -.5
                      and alex["hn@1"] <= asem["hn@1"] + 1e-12
                      and alex["hit@1"] >= asem["hit@1"] - .005 - 1e-12)
    return {"preferred": "ALEX" if superiority or noninferiority else "ASEM", "h1": h1,
            "h2": {"tested": True, "superiority": superiority, "safety_noninferiority": noninferiority,
                   "development_delta": delta}}
