from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT


SCIMDIX_REVISION = "69170fbdd15a4e19e247a9eb9554356d018b6f57"
SCIER_REVISION = "db347813de379eb200e0813bbfee350d67e7c701"

SCIMDIX_Q5: dict[str, tuple[str, ...]] = {
    "goal": (
        "Какова цель исследования?", "Какую цель ставят авторы?",
        "Для чего выполнено исследование?", "Чего стремятся достичь авторы?",
        "Какова основная цель работы?",
    ),
    "material": (
        "Какой материал или набор данных используется?",
        "Какие данные или материалы рассматриваются?",
        "На каком материале выполнено исследование?",
        "Какой исследовательский материал указан в работе?",
        "Какие данные служат материалом исследования?",
    ),
    "method": (
        "Какой метод используется или предлагается?", "Какой подход применяют авторы?",
        "С помощью какого метода выполнено исследование?",
        "Какой метод или модель описаны в работе?", "Какая методика используется?",
    ),
    "result": (
        "Какой результат получен в работе?", "Что получили авторы исследования?",
        "Каков основной результат работы?", "Какие результаты представлены?",
        "Что было установлено или разработано?",
    ),
    "task": (
        "Какая задача решается в работе?", "Какую задачу рассматривают авторы?",
        "Решению какой задачи посвящено исследование?",
        "Какая исследовательская задача поставлена?", "Какую проблему решает работа?",
    ),
    "tool": (
        "Какой инструмент используется в работе?", "Какое средство применяют авторы?",
        "С помощью какого инструмента выполнено исследование?",
        "Какое программное или техническое средство используется?",
        "Какой инструмент указан в работе?",
    ),
    "use": (
        "Для чего используется описанный подход?", "Каково применение метода или средства?",
        "Для какой цели применяется описанное решение?",
        "Как авторы используют представленный подход?", "Каково назначение описанного решения?",
    ),
}

SCIER_Q5: dict[str, tuple[str, ...]] = {
    "material": (
        "What datasets or materials are mentioned in the paper?",
        "Which datasets are named in the document?",
        "What data resources are discussed in the paper?",
        "Which datasets or corpora occur in the document?",
        "What experimental data or benchmarks are mentioned?",
    ),
    "method": (
        "What methods or models are mentioned in the paper?",
        "Which methods are named in the document?",
        "What models or approaches are discussed?",
        "Which algorithms or systems occur in the paper?",
        "What methodological approaches are mentioned?",
    ),
    "task": (
        "What tasks are mentioned in the paper?", "Which research tasks are named?",
        "What problems or tasks are discussed in the document?",
        "Which application tasks occur in the paper?", "What tasks does the paper mention?",
    ),
}

SCIMDIX_NATIVE_MAP = {
    "AIM": "goal", "MATERIAL": "material", "METHOD": "method",
    "RESULT": "result", "TASK": "task", "TOOL": "tool", "USAGE": "use",
}
SCIER_NATIVE_MAP = {"Dataset": "material", "Method": "method", "Task": "task"}


@dataclass(frozen=True)
class ExternalDataConfig:
    output_dir: Path
    datasets: tuple[str, ...] = ("scimdix", "scier")
    scimdix_source: Path | None = None
    scier_source: Path | None = None
    download: bool = True
    scimdix_chunk_max_tokens: int = 128
    scimdix_overlap_tokens: int = 24
    scier_chunk_max_tokens: int = 384
    scier_overlap_tokens: int = 50
    n_folds: int = 5
    seed: int = 42
    device: str = "auto"
    hard_negative_top_n: int = 5
    gap_bootstrap: int = 10
    limit_docs: int = 0

    def chunking(self, dataset: str) -> tuple[int, int]:
        if dataset == "scimdix":
            return self.scimdix_chunk_max_tokens, self.scimdix_overlap_tokens
        if dataset == "scier":
            return self.scier_chunk_max_tokens, self.scier_overlap_tokens
        raise ValueError(f"unknown dataset: {dataset}")

    @property
    def english_tokenizer(self) -> Path:
        return PROJECT_ROOT / "scripts/models_processing/models/bert_eng_qa_baseroberta_model"

    @property
    def russian_tokenizer(self) -> Path:
        return PROJECT_ROOT / "scripts/models_processing/models/rubert_ru_qa_model"

    def validate(self) -> None:
        unknown = set(self.datasets) - {"scimdix", "scier"}
        if unknown or not self.datasets:
            raise ValueError(f"datasets must be a non-empty subset of scimdix,scier: {sorted(unknown)}")
        for dataset in ("scimdix", "scier"):
            chunk_max_tokens, overlap_tokens = self.chunking(dataset)
            if chunk_max_tokens < 16:
                raise ValueError(f"{dataset}_chunk_max_tokens must be at least 16")
            if not 0 <= overlap_tokens < chunk_max_tokens / 2:
                raise ValueError(f"{dataset}_overlap_tokens must be less than half chunk size")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.hard_negative_top_n < 1:
            raise ValueError("hard_negative_top_n must be positive")
        if self.gap_bootstrap < 1:
            raise ValueError("gap_bootstrap must be positive")
        if not self.download:
            if "scimdix" in self.datasets and self.scimdix_source is None:
                raise ValueError("--scimdix-source is required with --no-download")
            if "scier" in self.datasets and self.scier_source is None:
                raise ValueError("--scier-source is required with --no-download")

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["output_dir"] = str(self.output_dir.resolve())
        row["scimdix_source"] = str(self.scimdix_source.resolve()) if self.scimdix_source else None
        row["scier_source"] = str(self.scier_source.resolve()) if self.scier_source else None
        row["english_tokenizer"] = str(self.english_tokenizer.resolve())
        row["russian_tokenizer"] = str(self.russian_tokenizer.resolve())
        row["source_revisions"] = {"scimdix": SCIMDIX_REVISION, "scier": SCIER_REVISION}
        row["questions_q5"] = {"scimdix": SCIMDIX_Q5, "scier": SCIER_Q5}
        return row
