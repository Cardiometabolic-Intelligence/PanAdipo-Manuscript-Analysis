"""Reading inputs, the analysis configuration, and writing result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def cli(doc: str, extra=None):
    """Standard arguments of the analysis scripts: --data (input tables) and --out (result tables)."""
    import argparse
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data", help="folder with the input tables (docs/DATA.md)")
    p.add_argument("--out", default="results", help="folder for the result tables")
    if extra:
        extra(p)
    return p.parse_args()


def config(name: str = "analysis") -> dict:
    """Load configs/<name>.yaml."""
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def read(data_dir: str | Path, name: str, **kw) -> pd.DataFrame:
    """Read <name>.csv or <name>.parquet from the data folder (input tables are described in docs/DATA.md)."""
    for ext, fn in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = Path(data_dir) / f"{name}{ext}"
        if p.exists():
            return fn(p, **kw)
    raise FileNotFoundError(f"input table '{name}' (.csv or .parquet) not found in the data folder")


def exists(data_dir: str | Path, name: str) -> bool:
    return any((Path(data_dir) / f"{name}{ext}").exists() for ext in (".csv", ".parquet"))


def write(df: pd.DataFrame, out_dir: str | Path, name: str) -> None:
    """Write a result table as <name>.csv and print it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 80):
        print(f"\n== {name} ==")
        print(df.round(3).to_string(index=False))
