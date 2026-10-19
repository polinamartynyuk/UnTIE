"""Frozen configuration for the isolated Russian V4.4 data preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT


# Native RuSERRC aspect names are kept. They are not forced into SciREX's four labels.
ASPECT_SPECS: dict[str, tuple[str, str]] = {
    "Task": ("task", "Какая задача решается в работе?"),
    "Goal": ("goal", "Какова цель работы?"),
    "Contrib": ("contribution", "Каков основной вклад работы?"),
    "Method": ("method", "Какой метод используется или предлагается в работе?"),
    "Tool": ("tool", "Какой инструмент используется в работе?"),
    "Use": ("use", "Для чего используется описанный подход?"),
    "Adv": ("advantage", "Какое преимущество описано в работе?"),
    "Example": ("example", "Какой пример приведён в работе?"),
    "Conc": ("conclusion", "Какой вывод сделан в работе?"),
}

OFFICIAL_REPOSITORY = "https://github.com/iis-research-team/ruserrc-dataset"
OFFICIAL_REF = "master"
OFFICIAL_PARTS = tuple(f"aspects_data_part_{index}.csv" for index in range(1, 4))


@dataclass(frozen=True)
class RuDataConfig:
    output_dir: Path
    source_dir: Path | None = None
    structured_source: Path | None = None
    download: bool = True
    device: str = "auto"
    seed: int = 42
    chunk_max_tokens: int = 128
    overlap_tokens: int = 24
    hard_negative_top_n: int = 5
    gap_bootstrap: int = 10
    n_folds: int = 5
    train_fraction: float = 0.60
    calibration_fraction: float = 0.15
    development_fraction: float = 0.15
    internal_test_fraction: float = 0.10
    limit_docs: int = 0
    aspects: tuple[str, ...] = field(
        default_factory=lambda: tuple(spec[0] for spec in ASPECT_SPECS.values())
    )

    @property
    def fractions(self) -> tuple[float, float, float, float]:
        return (
            self.train_fraction,
            self.calibration_fraction,
            self.development_fraction,
            self.internal_test_fraction,
        )

    def validate(self) -> None:
        known = {spec[0] for spec in ASPECT_SPECS.values()}
        unknown = set(self.aspects) - known
        if unknown:
            raise ValueError(f"unknown RuSERRC aspects: {sorted(unknown)}")
        if not self.aspects:
            raise ValueError("at least one aspect is required")
        source_count = int(self.download) + int(self.source_dir is not None) + int(self.structured_source is not None)
        if source_count != 1:
            raise ValueError("choose exactly one source: --download, --source-dir, or --structured-source")
        if self.chunk_max_tokens < 8:
            raise ValueError("chunk-max-tokens must be at least 8")
        if not 0 <= self.overlap_tokens < self.chunk_max_tokens / 2:
            raise ValueError("overlap-tokens must be less than half the chunk size")
        if abs(sum(self.fractions) - 1.0) > 1e-9:
            raise ValueError("document split fractions must sum to 1")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["source_dir"] = str(self.source_dir) if self.source_dir else None
        payload["structured_source"] = str(self.structured_source) if self.structured_source else None
        payload["fractions"] = list(self.fractions)
        payload["official_repository"] = OFFICIAL_REPOSITORY
        payload["official_ref"] = OFFICIAL_REF
        payload["questions"] = {
            aspect: question for _, (aspect, question) in ASPECT_SPECS.items()
        }
        return payload


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "qa_chunk_selection_v4_4_ru_data"
