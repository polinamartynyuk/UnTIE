from __future__ import annotations

import json
from pathlib import Path


def test_hierarchical_topic_notebook_code_cells_compile() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "notebooks"
        / "09_Hierarchical_topic_reranking_en.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
