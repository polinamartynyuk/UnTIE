from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_json_dataframe(path: str | Path, *, orient: str = "records") -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    try:
        return pd.read_json(source, orient=orient, lines=True)
    except ValueError:
        return pd.read_json(source, orient=orient)


def save_json_dataframe(
    dataframe: pd.DataFrame, path: str | Path, *, orient: str = "records"
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_json(destination, orient=orient, force_ascii=False, indent=2)


def replace_underscores(
    dataframe: pd.DataFrame,
    column: str,
    *,
    destination_column: str | None = None,
) -> pd.DataFrame:
    if column not in dataframe:
        raise ValueError(f"Missing column: {column}")
    result = dataframe.copy()
    destination = destination_column or f"{column}_cleaned"
    result[destination] = result[column].apply(
        lambda values: [value.replace("_", " ") for value in values]
        if isinstance(values, list)
        else values
    )
    return result
