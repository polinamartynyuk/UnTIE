def write_report(output, selection, summaries, efficiency, aspects):
    path = output / "reports/results_v4_4_ru.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V4.4-RU: автоматически сформированный отчёт", "",
        "Это cross-fitted development replay на русскоязычных answerable-инстансах RusErrC.",
        "Независимая тестовая выборка не открывалась и не создавалась.", "",
        f"Предпочтительный development-кандидат: **{selection['preferred']}**.",
        f"H1: **{selection['h1']['pass']}**; H2 выполнен: **{selection['h2']['tested']}**.", "",
        "| Система | n | Token F1 | ROUGE-L RU | Hit@1 | HN@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("B0-live", "V6-K5", "SEM-FIX", "LEX-FIX", "ASEM", "ALEX", "PREF-SUPPORT"):
        if name in summaries:
            row = summaries[name]
            lines.append(f"| {name} | {row.get('n','')} | {row.get('token_f1',0):.4f} | {row.get('rouge_l_f1',0):.4f} | {100*row.get('hit@1',0):.2f}% | {100*row.get('hn@1',0):.2f}% |")
    delta = selection["h1"]["development_delta"]
    lines += ["", f"ASEM−V6: Δ Token F1={delta['mean_delta']:.4f}; 95% development CI [{delta['ci95_low']:.4f}, {delta['ci95_high']:.4f}].",
              "", f"Аспекты: {', '.join(aspects)}.",
              "V6-веса выбирались отдельно для каждого прогона только по CAL; Q70 — только по TRAIN+CAL.",
              "Ответы переиспользованы из канонического Q5-кэша; новых QA forward passes: 0.",
              "ROUGE-L рассчитан Unicode-совместимой word-LCS реализацией.",
              f"Время: {efficiency['total_elapsed_seconds']:.1f} с.", "",
              "Полные метрики находятся в analysis/, веса — в v6_weights/, пороги — в confidence/.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
