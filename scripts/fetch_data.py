"""Fetch all public data and cache it to data/de_wind_hourly.csv.gz (~2 MB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dwpf.data import build_dataset  # noqa: E402

if __name__ == "__main__":
    df = build_dataset()
    print(df.shape, df.index.min(), df.index.max())
    print(df.isna().sum())
