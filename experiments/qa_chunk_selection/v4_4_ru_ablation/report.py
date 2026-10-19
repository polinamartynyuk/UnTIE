def write_report(root, model_id, metrics, comparisons, tuning):
    path = root / "reports/results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# V4.4-RU ablation: {model_id}", "",
             "Cross-fitted development results. Outer EVAL was excluded from question, weight and routing selection.", "",
             "| Cell | System | Token F1 | Lemma F1 | Hit@1 | HN@1 | ΔF1 vs B0 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(metrics):
        cell, system = key.split("/")
        row = metrics[key]
        delta = 0.0 if system == "B0" else comparisons[cell][system]["mean_delta"]
        lines.append(f"| {cell} | {system} | {row.get('token_f1',0):.4f} | {row.get('lemma_f1',0):.4f} | "
                     f"{100*row.get('hit@1',0):.2f}% | {100*row.get('hn@1',0):.2f}% | {delta:+.4f} |")
    lines += ["", "## Interpretation guardrails", "",
              "- `q0` and per-aspect `selected` are separate ablations.",
              "- K=3 uses a coherent pair versus the remaining answer; K=4/5 retain a coherent triple versus outside answers.",
              "- Lemma F1 is secondary. `LEX-LEM` is the separate ranking ablation.",
              "- `ASEM-Q5` is exploratory: a gold-free Q30 veto measures agreement of all five question formulations.",
              "- Related aspects remain separate; pooled family metrics are diagnostic only.",
              "- V6 falls back to exact B0 ordering when source-fold stability and safety checks fail.",
              "- These are repeatedly used development data; no independent test conclusion is permitted.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

