"""Deployment draft. It must never masquerade as a fitted/test-ready pipeline."""
from .contract import V6_GLOBAL, SEM_FIXED, LEX_FIXED, FINAL_SEED, STATUS


def build_draft(preferred, threshold, hashes, *, smoke=False):
    if preferred not in {"V6-K5", "ASEM", "ALEX"}:
        raise ValueError("Diagnostic cannot be frozen")
    pending = ["Fit B0-final on unlocked data using frozen V3 B0 recipe; record fitted model hash",
               "Verify the assembled single-seed runtime pipeline before any separately authorized locked test"]
    if preferred != "V6-K5" and (not threshold or not threshold["enabled"]):
        pending.append("Insufficient canonical-seed threshold rows: resolve deployable routing before test")
    if smoke:
        pending.append("Synthetic smoke artifacts cannot be used for deployment")
    return {
        "status": "DRAFT; not deployable; INTERNAL_TEST locked", "development_status": STATUS,
        "preferred_architecture": preferred, "ready_for_internal_test": False,
        "runtime": {"primary_k": 5, "v6_weights": dict(V6_GLOBAL),
                    "reader_weights": dict(SEM_FIXED if preferred == "ASEM" else LEX_FIXED) if preferred != "V6-K5" else None,
                    "routing_threshold": threshold, "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "seed_policy": {"kind": "single", "canonical_seed": FINAL_SEED, "chosen_before_replay": True},
                    "b0_recipe": "frozen V3 B0 / V2-B2a; no EVAL seed selection"},
        "b0_final_trained": False, "internal_test": "locked", "smoke": smoke,
        "pending": pending, "source_sha256": hashes,
    }
