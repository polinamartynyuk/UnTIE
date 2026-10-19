from experiments.qa_chunk_selection.v4_4_ru_ablation.features import adaptive_semantic_pack
from experiments.qa_chunk_selection.v4_4_ru_ablation.lemma import RussianLemmatizer
from experiments.qa_chunk_selection.v4_4_ru_ablation.tuning import select_stable


def test_k3_confidence_leaves_one_answer_out():
    answers = {0: "метод", 1: "метод", 2: "вывод"}
    embeddings = {0: [1.0, 0.0], 1: [.99, .01], 2: [0.0, 1.0]}
    pack = adaptive_semantic_pack(answers, embeddings, [0, 1, 2])
    assert pack["core_size"] == 2
    assert pack["best_core"] == [0, 1]
    assert pack["consensus_conf"] > 0


def test_russian_lemma_metric_is_secondary_and_inflection_aware():
    lemma = RussianLemmatizer()
    assert lemma.f1("метод", "методом") == 100.0
    assert lemma.f1("метод", "вывод") == 0.0


def test_stability_selector_can_fallback_to_baseline():
    keys = [("a", 42), ("b", 42)]
    utility = {key: {0: {"f1": 100.0, "positive": 1.0, "hn": 0.0},
                     1: {"f1": 0.0, "positive": 0.0, "hn": 1.0}} for key in keys}
    result = select_stable(keys, [{"x": 1}], lambda key, params: 1, utility,
                           {keys[0]: 0, keys[1]: 1}, baseline_top=lambda key: 0)
    assert result["fallback"] is True

