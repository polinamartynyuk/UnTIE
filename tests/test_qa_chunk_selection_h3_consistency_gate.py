from experiments.qa_chunk_selection_h3_consistency_gate.pipeline import consistency_decision


def test_exact_consistency_accepts_endpoint_or_intermediate_lambda():
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        decision = consistency_decision(value, value)
        assert decision["accepted_lambda"] == value
        assert decision["fallback_to_h3"] is False
        assert decision["confidence_interval_gate"] is False


def test_any_train_refit_mismatch_falls_back_to_h3():
    for development, refit in ((1.0, 0.5), (0.75, 1.0), (0.0, 0.5)):
        decision = consistency_decision(development, refit)
        assert decision["accepted_lambda"] == 1.0
        assert decision["fallback_to_h3"] is True
        assert decision["reason"] == "UNSTABLE_SELECTION_FALLBACK_H3"
