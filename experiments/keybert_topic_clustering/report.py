"""Write summary.md, plots, and aggregated tables after the run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io_util import write_csv


def pca_scatter(path: Path, vectors: np.ndarray, labels: np.ndarray, title: str) -> None:
    from sklearn.decomposition import PCA
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(vectors) < 3:
        return
    points = PCA(n_components=2, random_state=42).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=28)
    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def size_histogram(path: Path, labels: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(int(item)) for item in unique], counts)
    ax.set_xlabel("cluster")
    ax.set_ylabel("train documents")
    ax.set_title("Cluster size distribution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# KeyBERT topic clustering experiment",
        "",
        "## Dataset",
        "",
        f"- documents: {payload.get('n_documents')}",
        f"- language: {payload.get('language')}",
        f"- fields: title/abstract parsed from body when metadata empty ({payload.get('metadata_empty_rate')})",
        f"- splits: train {payload.get('n_train')} / dev {payload.get('n_dev')} / test {payload.get('n_test')}",
        f"- mean abstract words: {payload.get('mean_abstract_words')}",
        "",
        "## Environment",
        "",
        f"- KeyBERT: {payload.get('keybert_version')}",
        f"- sentence-transformers: {payload.get('st_version')}",
        f"- sklearn: {payload.get('sklearn_version')}",
        f"- topic encoder: {payload.get('topic_encoder')}",
        f"- CE: {payload.get('ce')}",
        f"- hardware: {payload.get('hardware')}",
        "",
        "## Clustering configurations",
        "",
        payload.get("clustering_table_md", ""),
        "",
        "## Selected clustering",
        "",
        f"- C_plain: {payload.get('c_plain')}",
        f"- C_keybert: {payload.get('c_keybert')}",
        "",
        "## Downstream metrics",
        "",
        payload.get("downstream_table_md", ""),
        "",
        "## Ablations",
        "",
        payload.get("ablations_md", ""),
        "",
        "## Conclusion",
        "",
        payload.get("conclusion_md", "Filled after metrics are computed."),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([line, sep, *body])


def conclude(metrics: Mapping[str, Mapping[str, float]]) -> str:
    def get(*keys: str) -> float | None:
        for key in keys:
            if key in metrics:
                return float(metrics[key]["mean_char_f1"])
        return None

    q0 = get("Q0")
    q1 = get("Q1")
    lines = []
    for clustering in ("plain", "keybert"):
        d0 = get(f"{clustering}/D0")
        d1 = get(f"{clustering}/D1")
        d2 = get(f"{clustering}/D2")
        if d0 is not None and d1 is not None:
            lines.append(f"- D1-D0 ({clustering}): {d1 - d0:+.4f}")
        if d1 is not None and d2 is not None:
            lines.append(f"- D2-D1 ({clustering}): {d2 - d1:+.4f}")
        if d2 is not None and q1 is not None:
            lines.append(f"- D2-Q1 ({clustering}): {d2 - q1:+.4f}")
    p2 = get("plain/D2")
    k2 = get("keybert/D2")
    if p2 is not None and k2 is not None:
        lines.append(f"- D2(KeyBERT)-D2(plain): {k2 - p2:+.4f}")
    lines.append("")
    lines.append("Answers to the prompt checklist are taken from these deltas; no result is assumed in advance.")
    return "\n".join(lines)
