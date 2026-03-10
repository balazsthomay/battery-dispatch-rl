"""Parquet-based data caching."""

from pathlib import Path

import pandas as pd


def cache_path(data_dir: str, zone: str, data_type: str, year: int) -> Path:
    """Return path: data_dir/zone/type_year.parquet."""
    return Path(data_dir) / zone / f"{data_type}_{year}.parquet"


def save_to_cache(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_from_cache(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_parquet(path)


def cache_exists(path: Path) -> bool:
    """Check if cache file exists."""
    return path.exists()
