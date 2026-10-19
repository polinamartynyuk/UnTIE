from collections import Counter, defaultdict
from pathlib import Path
import json

from experiments.qa_chunk_selection.io_util import write_json
from experiments.qa_chunk_selection.v4_1.evaluate import row_key, summarize_rows, transitions
from experiments.qa_chunk_selection.v4_4_ru.evaluate import paired_delta
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl
from experiments.qa_chunk_selection.v4_4_ru_ablation.lemma import RussianLemmatizer

from .contract import BOOTSTRAP_SEED, SYSTEMS


def _load(root, dataset, system):
    return read_jsonl(Path(root) / dataset / "oof" / system / "predictions_scored.jsonl")


def _observed_delta(left, right, metric="token_f1"):
    base = {row_key(row): row for row in left}
    by_question = defaultdict(list)
    for row in right:
        key = row_key(row)
        by_question[str(row["question_id"])].append(float(row[metric]) - float(base[key][metric]))
    values = [sum(rows) / len(rows) for rows in by_question.values()]
    return sum(values) / max(len(values), 1)


def _comparison(left, right, config, metric="token_f1"):
    result = paired_delta(left, right, metric=metric,
                          n_boot=50 if config.smoke else config.n_bootstrap,
                          seed=BOOTSTRAP_SEED)
    result["observed_mean_delta"] = _observed_delta(left, right, metric)
    result["bootstrap_mean_delta"] = result.pop("mean_delta")
    return result


def _subset(rows, key, value):
    return [row for row in rows if row.get(key) == value]


def _switch_analysis(base_rows, candidate_rows):
    base = {row_key(row): row for row in base_rows}
    counts = Counter()
    gain = defaultdict(float)
    missed = 0
    for row in candidate_rows:
        previous = base[row_key(row)]
        same = int(previous["selected_chunk_id"]) == int(row["selected_chunk_id"])
        delta = float(row["token_f1"]) - float(previous["token_f1"])
        if same:
            label = "same_top1"
        elif delta > 1e-9:
            label = "good_switch"
        elif delta < -1e-9:
            label = "bad_switch"
        else:
            label = "neutral_switch"
        counts[label] += 1
        gain[label] += delta
        if row.get("router_branch_used") == "V6":
            missed += int(False)  # Gold-based opportunity is reported only when both branch rows exist.
    useful = counts["good_switch"] + counts["bad_switch"]
    return {
        "counts": dict(counts), "token_f1_delta_sum": dict(gain),
        "useful_switch_precision": counts["good_switch"] / useful if useful else None,
        "actual_top1_change_rate": 1.0 - counts["same_top1"] / max(len(candidate_rows), 1),
        "missed_opportunity": None,
        "missed_opportunity_note": "not inferred without a separately materialized counterfactual branch utility",
    }



def _router_opportunity(rows):
    available = [row for row in rows if row.get("router_available")]
    opportunities = [row for row in available if float(row["sem_token_f1"]) > float(row["v6_token_f1"]) + 1e-9]
    used = [row for row in available if row.get("router_used_sem")]
    good = [row for row in used if float(row["sem_token_f1"]) > float(row["v6_token_f1"]) + 1e-9]
    bad = [row for row in used if float(row["sem_token_f1"]) < float(row["v6_token_f1"]) - 1e-9]
    neutral = [row for row in used if abs(float(row["sem_token_f1"]) - float(row["v6_token_f1"])) <= 1e-9]
    decisive = len(good) + len(bad)
    return {
        "available_rows": len(available), "reader_branch_rows": len(used),
        "good_router_switches": len(good), "bad_router_switches": len(bad),
        "neutral_router_switches": len(neutral),
        "missed_opportunities": len([row for row in opportunities if not row.get("router_used_sem")]),
        "useful_switch_precision": len(good) / decisive if decisive else None,
        "useful_switch_recall": len(good) / len(opportunities) if opportunities else None,
        "opportunity_definition": "SEM counterfactual Token F1 > V6 Token F1; post-hoc gold diagnostic only",
    }


def _router_analysis(rows):
    available = sum(bool(row.get("router_available")) for row in rows)
    reader = sum(row.get("router_branch_used") == "SEM" for row in rows)
    return {
        "n": len(rows), "available": available,
        "availability_rate": available / max(len(rows), 1),
        "reader_branch_used": reader, "reader_branch_use_rate": reader / max(len(rows), 1),
        "fallback_count": len(rows) - reader, "fallback_rate": (len(rows) - reader) / max(len(rows), 1),
        "runtime_features_gold_free": True,
    }


def _metrics_with_lemma(rows, state, lemmatizer):
    result = summarize_rows(rows)
    lemma = []
    for row in rows:
        item = state.items[str(row["question_id"])]
        lemma.append(lemmatizer.gold_f1(str(row.get("qa_answer") or ""), item.gold_answers))
    result["lemma_f1"] = sum(lemma) / max(len(lemma), 1)
    return result


def write_analysis(config, output, states, progress):
    progress.stage("analysis", "paired quality, safety, stability, routing and trade-off")
    output = Path(output)
    primary, comparisons, seeds, folds, aspects = {}, {}, {}, {}, {}
    switches, evidence, routers, residual = {}, {}, {}, {}
    lemmatizer = RussianLemmatizer(smoke=config.smoke)
    for dataset, state in states.items():
        rows = {system: _load(output, dataset, system) for system in SYSTEMS}
        primary[dataset] = {system: _metrics_with_lemma(value, state, lemmatizer) for system, value in rows.items()}
        comparisons[dataset] = {
            "S3_vs_S1_primary": _comparison(rows["S1_dity_b0"], rows["S3_adaptive"], config),
            "S2_vs_S1": _comparison(rows["S1_dity_b0"], rows["S2_v6_support"], config),
            "S3_vs_S2": _comparison(rows["S2_v6_support"], rows["S3_adaptive"], config),
            "S1_vs_S0": _comparison(rows["S0_historical_b0"], rows["S1_dity_b0"], config),
            "S3_vs_S4_descriptive": _comparison(rows["S4_bge_b0"], rows["S3_adaptive"], config),
        }
        base_metrics, candidate_metrics = primary[dataset]["S1_dity_b0"], primary[dataset]["S3_adaptive"]
        comparisons[dataset]["safety"] = {
            "hit1_delta_pp": 100 * (candidate_metrics["hit@1"] - base_metrics["hit@1"]),
            "hn1_delta_pp": 100 * (candidate_metrics["hn@1"] - base_metrics["hn@1"]),
            "hit1_pass": candidate_metrics["hit@1"] >= base_metrics["hit@1"] - 0.005,
            "hn1_pass": candidate_metrics["hn@1"] <= base_metrics["hn@1"] + 0.01,
            "convention": "inherited V4.4 development safety margins",
        }
        seeds[dataset] = {}
        for seed in config.seeds:
            seeds[dataset][str(seed)] = {
                system: _metrics_with_lemma(_subset(value, "seed", seed), state, lemmatizer)
                for system, value in rows.items() if system in {"S1_dity_b0", "S2_v6_support", "S3_adaptive", "S4_bge_b0"}
            }
            seeds[dataset][str(seed)]["S3_minus_S1"] = _observed_delta(
                _subset(rows["S1_dity_b0"], "seed", seed), _subset(rows["S3_adaptive"], "seed", seed))
        folds[dataset] = {}
        fold_deltas = []
        for fold in ((0,) if config.smoke else range(5)):
            left, right = _subset(rows["S1_dity_b0"], "cv_run", fold), _subset(rows["S3_adaptive"], "cv_run", fold)
            delta = _observed_delta(left, right)
            fold_deltas.append(delta)
            lm, rm = summarize_rows(left), summarize_rows(right)
            folds[dataset][str(fold)] = {
                "delta_f1": delta, "delta_hit1_pp": 100 * (rm["hit@1"] - lm["hit@1"]),
                "delta_hn1_pp": 100 * (rm["hn@1"] - lm["hn@1"]), "n": len(right),
            }
        fold_net_sums = []
        for fold in ((0,) if config.smoke else range(5)):
            left_index = {row_key(row): row for row in _subset(rows["S1_dity_b0"], "cv_run", fold)}
            fold_net_sums.append(sum(float(row["token_f1"]) - float(left_index[row_key(row)]["token_f1"])
                                     for row in _subset(rows["S3_adaptive"], "cv_run", fold)))
        total_fold_net = sum(fold_net_sums)
        folds[dataset]["summary"] = {
            "positive_fold_count": sum(value > 1e-9 for value in fold_deltas),
            "negative_fold_count": sum(value < -1e-9 for value in fold_deltas),
            "range": [min(fold_deltas), max(fold_deltas)],
            "fold_deltas": fold_deltas, "fold_net_delta_sums": fold_net_sums,
            "fraction_of_total_net_gain_by_fold": [value / total_fold_net if abs(total_fold_net) > 1e-12 else None for value in fold_net_sums],
        }
        aspects[dataset] = {}
        for aspect in sorted({item.aspect for item in state.records}):
            aspects[dataset][aspect] = {
                "N_questions": len({str(row["question_id"]) for row in rows["S1_dity_b0"] if row["aspect"] == aspect}),
                "S1": summarize_rows([row for row in rows["S1_dity_b0"] if row["aspect"] == aspect]),
                "S3": summarize_rows([row for row in rows["S3_adaptive"] if row["aspect"] == aspect]),
                "delta_f1": _observed_delta([row for row in rows["S1_dity_b0"] if row["aspect"] == aspect],
                                            [row for row in rows["S3_adaptive"] if row["aspect"] == aspect]),
            }
        switches[dataset] = _switch_analysis(rows["S1_dity_b0"], rows["S3_adaptive"])
        evidence[dataset] = transitions(rows["S1_dity_b0"], rows["S3_adaptive"])
        routers[dataset] = _router_analysis(rows["S3_adaptive"])
        counterfactual_path = output / dataset / "stage2/counterfactual_router_utility.jsonl"
        routers[dataset]["opportunity"] = _router_opportunity(read_jsonl(counterfactual_path))
        residual[dataset] = {
            system: {key: primary[dataset][system].get(key) for key in ("token_f1", "hit@1", "hn@1", "mrr")}
            for system in ("S0_historical_b0", "S1_dity_b0", "S2_v6_support", "S3_adaptive", "S4_bge_b0")
        }
        progress.log("analysis complete", dataset=dataset,
                     primary_delta=comparisons[dataset]["S3_vs_S1_primary"]["observed_mean_delta"])
    analysis = output / "analysis"
    write_json(analysis / "primary_metrics.json", primary)
    write_json(analysis / "paired_bootstrap.json", comparisons)
    write_json(analysis / "seed_stability.json", seeds)
    write_json(analysis / "fold_stability.json", folds)
    write_json(analysis / "aspect_metrics.json", aspects)
    write_json(analysis / "switch_analysis.json", switches)
    write_json(analysis / "evidence_transitions.json", evidence)
    write_json(analysis / "router_analysis.json", routers)
    write_json(analysis / "residual_error_analysis.json", residual)
    resource_path = output / "resource_benchmark/summary.json"
    resources = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.is_file() else {"status": "not_run"}
    tradeoff = {
        "quality": {dataset: {
            "S3_token_f1": primary[dataset]["S3_adaptive"]["token_f1"],
            "BGE_token_f1": primary[dataset]["S4_bge_b0"]["token_f1"],
            "S3_minus_BGE": comparisons[dataset]["S3_vs_S4_descriptive"]["observed_mean_delta"],
        } for dataset in states},
        "resources": resources,
        "rule": "S3 is Pareto-dominated if it is worse in Token F1, peak VRAM and latency; no scalar F1/GB criterion",
        "non_inferiority_margin": None,
    }
    write_json(analysis / "quality_resource_tradeoff.json", tradeoff)
    selection = {}
    for dataset in states:
        safety = comparisons[dataset]["safety"]
        delta = comparisons[dataset]["S3_vs_S1_primary"]["observed_mean_delta"]
        stable = folds[dataset]["summary"]["positive_fold_count"] >= 3
        candidate = "S3_adaptive" if safety["hit1_pass"] and safety["hn1_pass"] and delta > 0 and stable else "S1_dity_b0"
        dominated = None
        if resources.get("status") == "complete":
            end = resources["end_to_end"][dataset]
            chosen_resource = end["S3_dity_stage2"] if candidate == "S3_adaptive" else end["S1_dity_b0"]
            bge_resource = end["S4_bge_b0"]
            chosen_f1 = primary[dataset][candidate]["token_f1"]
            bge_f1 = primary[dataset]["S4_bge_b0"]["token_f1"]
            chosen_vram = chosen_resource.get("peak_gpu_allocated_bytes")
            bge_vram = bge_resource.get("peak_gpu_allocated_bytes")
            dominated = None if chosen_vram is None or bge_vram is None else bool(
                chosen_f1 < bge_f1 and
                chosen_resource["latency_seconds_per_question"] >= bge_resource["latency_seconds_per_question"] and
                chosen_vram >= bge_vram)
            if dominated is True:
                candidate = "S4_bge_b0"
        selection[dataset] = {
            "selected_development_candidate": candidate,
            "hierarchy": ["runtime validity", "evidence safety", "stage2 stability", "resource trade-off", "Token F1", "simplicity"],
            "S3_safety_pass": bool(safety["hit1_pass"] and safety["hn1_pass"]),
            "S3_positive_fold_majority": stable, "S3_observed_delta": delta,
            "pareto_dominated_by_BGE": dominated,
            "independent_confirmation": False,
        }
    write_json(analysis / "development_selection.json", selection)
    freeze = {
        "status": "draft_frozen_for_V4.5_B; independent test unavailable",
        "architecture_selected_on": "V4.5-A reused development OOF",
        "ranker": {"model": "DiTy/cross-encoder-russian-msmarco", "revision": "9029bab08103ad171724b510d312befa5b476293"},
        "final_ranker_recipe": "contract_frozen.json training_recipe; final seed 42",
        "final_ranker_training": "pending until a prospective test is authorized and provenance is untouched",
        "datasets": {}, "question": "canonical q0", "Q5_selection": False, "Q5_veto": False,
        "QA_reader": "frozen Russian reader from contract_frozen.json",
        "runtime_preprocessing": "exact historical dataset-specific chunking",
        "metrics": ["Token F1", "Lemma F1", "Character F1", "Unicode ROUGE-L", "EM", "MRR", "Hit/HN"],
        "post_test_changes_forbidden": ["K", "alpha/beta/gamma", "router", "q0", "Q5", "seed", "SEM/LEX family", "QA", "preprocessing"],
        "independent_test_opened": False,
    }
    for dataset in states:
        params = json.loads((output / dataset / "stage2/global_deployable_parameters.json").read_text(encoding="utf-8"))
        freeze["datasets"][dataset] = {"development_selection": selection[dataset], "global_runtime_parameters": params}
    from .io_guard import canonical_hash
    freeze["freeze_sha256"] = canonical_hash(freeze)
    write_json(output / "final_freeze_manifest_draft.json", freeze)
    _write_report(output, primary, comparisons, folds, switches, routers, resources)


def _write_report(output, metrics, comparisons, folds, switches, routers, resources):
    lines = [
        "# V4.5-RU compact refinement: результаты", "",
        "> V4.5-A является development characterization на уже использованных данных, а не независимым подтверждением.", "",
        "## Основные результаты", "",
        "| Датасет | S0 old B0 | S1 DiTy | S2 V6-support | S3 adaptive | S4 BGE | ΔS3−S1 | 95% development CI | Safety |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset in metrics:
        row, cmp = metrics[dataset], comparisons[dataset]
        delta, safety = cmp["S3_vs_S1_primary"], cmp["safety"]
        lines.append(
            f"| {dataset} | {row['S0_historical_b0']['token_f1']:.4f} | {row['S1_dity_b0']['token_f1']:.4f} | "
            f"{row['S2_v6_support']['token_f1']:.4f} | {row['S3_adaptive']['token_f1']:.4f} | "
            f"{row['S4_bge_b0']['token_f1']:.4f} | {delta['observed_mean_delta']:+.4f} | "
            f"[{delta['ci95_low']:+.4f}; {delta['ci95_high']:+.4f}] | "
            f"Hit={'pass' if safety['hit1_pass'] else 'fail'}, HN={'pass' if safety['hn1_pass'] else 'fail'} |"
        )
    lines += ["", "## Интерпретация", "",
              "Primary comparison — S3 против S1. SciMDIX является frozen reproduction/stability check; RuSERRC — новая development characterization той же заранее замороженной процедуры.", "",
              "BGE — strong tested reference among the tested rankers. Сравнение S3 с BGE является описательным trade-off: formal non-inferiority margin заранее не задан.", "",
              "Параметры stage2 и router выбраны только на TRAIN+CAL каждого outer fold. Q0 и K были заморожены до EVAL; Q5 selection и Q5-veto не используются.", "",
              "## Устойчивость и переключения", ""]
    for dataset in metrics:
        fs, sw, rt = folds[dataset]["summary"], switches[dataset], routers[dataset]
        lines += [f"### {dataset}", "",
                  f"- Положительных/отрицательных folds: {fs['positive_fold_count']}/{fs['negative_fold_count']}.",
                  f"- Диапазон fold ΔF1: [{fs['range'][0]:+.4f}; {fs['range'][1]:+.4f}].",
                  f"- Router availability: {100*rt['availability_rate']:.2f}%; reader branch use: {100*rt['reader_branch_use_rate']:.2f}%.",
                  f"- Switch counts: {sw['counts']}.", ""]
    lines += ["## Resource benchmark", "", f"Status: {resources.get('status', 'complete')}", "",
              "Полные environment, raw observations, ranker-only и end-to-end component measurements находятся в resource_benchmark/. End-to-end сравнение учитывает top-K QA для S3 и top-1 QA для BGE.", "",
              "## Ограничения", "",
              "- Все CI development; датасеты уже использовались в V4.4.",
              "- Лучший seed не выбирается; три seeds усредняются внутри вопроса, bootstrap ресэмплирует документы.",
              "- В отчёте не используется исторически повреждённое protocol score mapping; V4.5 сериализует scores строго и проверяет конечность.",
              "- Ресурсный вывод делается по измерениям отдельных измерений и Pareto rule, без искусственного общего scalar.", ""]
    report = output / "reports/results_v4_5_ru.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
