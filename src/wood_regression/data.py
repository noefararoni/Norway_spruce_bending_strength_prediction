"""CSV reading and cleaning with stable specimen tracking."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, TARGET


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read comma or semicolon CSVs, including European decimal commas."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin1")
    if not text.strip():
        raise ValueError(f"Dataset is empty: {path}")
    separator = ";" if ";" in text.splitlines()[0] else ","
    df = pd.read_csv(StringIO(text), sep=separator)
    df.columns = df.columns.str.strip()
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep model features and provenance; discard invalid numeric rows."""
    required = FEATURES + [TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    tracking = [c for c in ("source_file", "source_row", "Specimen ID") if c in df]
    result = df[required + tracking].copy()
    for column in required:
        result[column] = pd.to_numeric(
            result[column].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if result.empty:
        raise ValueError("No complete numeric specimens remain after cleaning.")
    if not result.index.is_unique:
        raise ValueError("Specimen indices must be unique.")
    return result


def merge_datasets(paths: list[Path]) -> pd.DataFrame:
    """Merge sources without silently dropping required columns."""
    frames = []
    for path in paths:
        frame = read_dataset(path)
        missing = set(FEATURES + [TARGET]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks required columns: {sorted(missing)}")
        frame["source_file"] = path.stem
        frame["source_row"] = frame.index
        frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one input dataset.")
    return clean_dataset(pd.concat(frames, ignore_index=True))
