"""Factor decomposition and human-readable reports for the factorial run."""

import json
from pathlib import Path

from experiments.qa_chunk_selection.io_util import write_json
from experiments.qa_chunk_selection.v4_1.evaluate import summarize_rows
from experiments.qa_chunk_selection.v4_4_ru.evaluate import paired_delta
from experiments.qa_chunk_selection.v4_4_ru.io_guard import read_jsonl

from .contract import RANKER_SPECS, STRATEGIES, SYSTEMS


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta(left, right, config):
    return paired_delta(left, right, n_boot=50 if config.smoke else config.n_bootstrap, seed=42)


def _system_rows(dataset_root, ranker, cell, system):
    path = Path(dataset_root) / "protocol" / ranker / "systems" / cell / system / "predictions_scored.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing protocol output: {path}")
    return read_jsonl(path)


def write_protocol_report(root, dataset, model_id, metrics, comparisons, tuning):
    path = Path(root) / "reports/results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# V4.4-RU: {dataset} / {model_id}", "",
        "Все вопросы, веса и пороги подобраны только по train+calibration соответствующего внешнего фолда; outer EVAL при подборе не использовался.", "",
        "| Ячейка | Система | Token F1 | Lemma F1 | Hit@1 | HN@1 | Delta F1 к своему B0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(metrics):
        cell, system = key.split("/")
        row = metrics[key]
        delta = 0.0 if system == "B0" else comparisons[cell][system]["mean_delta"]
        lines.append(
            f"| {cell} | {system} | {row.get('token_f1', 0):.4f} | {row.get('lemma_f1', 0):.4f} | "
            f"{100 * row.get('hit@1', 0):.2f}% | {100 * row.get('hn@1', 0):.2f}% | {delta:+.4f} |"
        )
    lines += [
        "", "## Ограничения интерпретации", "",
        "- B0 внутри ячейки — выбранный ранкер без второго компонента.",
        "- q0 и fold-local selected проверяются раздельно.",
        "- Для каждого K и внешнего фолда коэффициенты V6/SEM/LEX/LEX-LEM обучаются заново на train+calibration.",
        "- ASEM и ALEX получают отдельные Q70-пороги. ASEM-Q5 и ALEX-Q5 добавляют симметричный gold-free Q30 veto.",
        "- Лемматизация участвует только в LEX-LEM и в отдельной метрике Lemma F1.",
        "- Наборы уже использовались при разработке; интервалы являются development CI.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_analysis(dataset_root, dataset, config, current_b0, b0_sets):
    dataset_root = Path(dataset_root)
    ranker_effect = {ranker: _delta(current_b0, rows, config) for ranker, rows in b0_sets.items()}
    component_effect, total_effect, question_effect, metric_tables = {}, {}, {}, {}
    expected_cells = [f"{strategy}_k{k}" for strategy in STRATEGIES for k in config.k_values]
    for ranker in config.rankers:
        component_effect[ranker], total_effect[ranker] = {}, {}
        question_effect[ranker], metric_tables[ranker] = {}, {}
        for cell in expected_cells:
            base = _system_rows(dataset_root, ranker, cell, "B0")
            component_effect[ranker][cell], total_effect[ranker][cell], metric_tables[ranker][cell] = {}, {}, {}
            for system in SYSTEMS:
                rows = _system_rows(dataset_root, ranker, cell, system)
                metric_tables[ranker][cell][system] = summarize_rows(rows)
                component_effect[ranker][cell][system] = None if system == "B0" else _delta(base, rows, config)
                total_effect[ranker][cell][system] = _delta(current_b0, rows, config)
        for k in config.k_values:
            q0 = _system_rows(dataset_root, ranker, f"q0_k{k}", "B0")
            selected = _system_rows(dataset_root, ranker, f"selected_k{k}", "B0")
            question_effect[ranker][f"k{k}"] = _delta(q0, selected, config)

    cross_ranker = {}
    if len(config.rankers) > 1:
        for left_index, left in enumerate(config.rankers):
            for right in config.rankers[left_index + 1:]:
                comparison = f"{right}_minus_{left}"
                cross_ranker[comparison] = {}
                for cell in expected_cells:
                    cross_ranker[comparison][cell] = {}
                    for system in SYSTEMS:
                        left_rows = _system_rows(dataset_root, left, cell, system)
                        right_rows = _system_rows(dataset_root, right, cell, system)
                        cross_ranker[comparison][cell][system] = _delta(left_rows, right_rows, config)

    joint_success = {}
    for ranker in config.rankers:
        ranker_supported = ranker_effect[ranker]["ci95_low"] > 0
        joint_success[ranker] = {}
        for cell in expected_cells:
            joint_success[ranker][cell] = {}
            for system in SYSTEMS:
                component = component_effect[ranker][cell][system]
                total = total_effect[ranker][cell][system]
                joint_success[ranker][cell][system] = {
                    "ranker_ci_above_zero": ranker_supported,
                    "component_ci_above_zero": bool(
                        component is not None and component["ci95_low"] > 0
                    ),
                    "total_ci_above_zero": total["ci95_low"] > 0,
                    "both_ranker_and_component_supported": bool(
                        ranker_supported and component is not None
                        and component["ci95_low"] > 0
                    ),
                }

    decomposition = {
        "dataset": dataset,
        "factor_definition": {
            "ranker_effect": "new ranker B0(q0) minus previous dataset B0(q0)",
            "question_effect": "selected-question B0 minus q0 B0 for the same new ranker",
            "component_effect": "V4.4 system minus B0 in the same ranker/question/K cell",
            "total_effect": "complete system minus previous dataset B0(q0)",
            "cross_ranker": "right ranker minus left ranker in the same cell/system",
        },
        "ranker_effect": ranker_effect, "question_effect": question_effect,
        "component_effect": component_effect, "total_effect": total_effect,
        "cross_ranker": cross_ranker,
        "joint_success": joint_success,
        "pairing": "qa_instance_id + seed; clustered bootstrap by document",
        "selection_uses_outer_eval": False,
    }
    write_json(dataset_root / "analysis/factor_decomposition.json", decomposition)
    write_json(dataset_root / "analysis/all_metrics.json", metric_tables)

    lines = [
        f"# Факторный V4.4-RU: {dataset}", "",
        "Эксперимент разделяет вклад нового ранкера и вклад второго компонента V4.4.", "",
        "## Вклад ранкера", "",
        "| Ранкер | Delta Token F1 к прежнему B0 | 95% CI |", "|---|---:|---:|",
    ]
    for ranker in config.rankers:
        delta = ranker_effect[ranker]
        lines.append(f"| {ranker} | {delta['mean_delta']:+.4f} | [{delta['ci95_low']:+.4f}; {delta['ci95_high']:+.4f}] |")
    lines += ["", "## Вклад второго компонента", "",
        "| Ранкер | Ячейка | Система | Token F1 | Delta к своему B0 | 95% CI |",
        "|---|---|---|---:|---:|---:|"]
    for ranker in config.rankers:
        for cell in expected_cells:
            for system in SYSTEMS:
                metric = metric_tables[ranker][cell][system]
                effect = component_effect[ranker][cell][system]
                if effect is None:
                    delta_text, ci_text = "+0.0000", "—"
                else:
                    delta_text = f"{effect['mean_delta']:+.4f}"
                    ci_text = f"[{effect['ci95_low']:+.4f}; {effect['ci95_high']:+.4f}]"
                lines.append(f"| {ranker} | {cell} | {system} | {metric.get('token_f1', 0):.4f} | {delta_text} | {ci_text} |")
    lines += ["",
        "Положительный вклад второго компонента считается подтверждённым для заранее заданной ячейки, если весь 95% development CI выше нуля. Выбор лучшей строки после просмотра всей таблицы остаётся описательным.", ""]
    report_path = dataset_root / "reports/factorial_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return decomposition


def write_root_report(root, config):
    root = Path(root)
    lines = [
        "# V4.4-RU: факторный эксперимент по ранкерам", "",
        "Два новых русских cross-encoder ранкера проверяются независимо на RuSERRC и SciMDIX. Все параметры второго компонента повторно обучаются внутри CV каждого датасета и каждого ранкера.", "",
        "| Датасет | Ранкер | Модель | Delta F1 ранкера к прежнему B0 | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for dataset in config.datasets:
        data = _load_json(root / dataset / "analysis/factor_decomposition.json")
        for ranker in config.rankers:
            row, spec = data["ranker_effect"][ranker], RANKER_SPECS[ranker]
            lines.append(f"| {dataset} | {ranker} | {spec.model_name} | {row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}; {row['ci95_high']:+.4f}] |")
    lines += ["", "Полные таблицы абсолютных метрик, вклада второго компонента, общего эффекта и прямого сравнения ранкеров находятся в каталогах каждого датасета.", ""]
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.md").write_text("\n".join(lines), encoding="utf-8")
