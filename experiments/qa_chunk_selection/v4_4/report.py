from .contract import STATUS


def write_report(output, selection, summaries, efficiency):
    path = output / "reports/results_v4_4.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# V4.4: автоматически сформированный отчёт", "", STATUS, "",
             "Это development stability replay на повторно использованном пуле. CI не устраняет selection bias.", "",
             f"Smoke: `{selection['smoke']}`. Preferred candidate: **{selection['preferred']}**.",
             f"H1 development gate: **{selection['h1']['pass']}**. H2 выполнен: **{selection['h2']['tested']}**.", "",
             "| Система | n | Token F1 | Hit@1 | HN@1 |", "|---|---:|---:|---:|---:|"]
    for name in ("B0-live", "V6-K5", "SEM-FIX", "LEX-FIX", "ASEM", "ALEX", "ASOFT", "PREF-SUPPORT"):
        if name in summaries:
            m = summaries[name]
            lines.append(f"| {name} | {m.get('n', '')} | {m['token_f1']:.4f} | {100*m['hit@1']:.2f}% | {100*m['hn@1']:.2f}% |")
    d = selection["h1"]["development_delta"]
    lines += ["", f"ASEM−V6: ΔF1={d['mean_delta']:.4f}; 95% development CI [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}].",
              "", "Always-controls, ASOFT и PREF-SUPPORT не участвуют в выборе финального кандидата.",
              "При H1 fail ALEX остаётся описательным результатом; H2 не проверяется.",
              "", "INTERNAL_TEST закрыт. B0-final не обучался. Конфигурация deployment — только черновик.",
              "", f"Время replay до отчёта: {efficiency['total_elapsed_seconds']:.1f} с. Новых QA forwards: 0.",
              "Полные сравнения: final/development_selection.json; диагностики: analysis/; thresholds: confidence/.",
              "", "Для научных выводов нужен отдельный разбор результатов в hypothesis/.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
