"""
Step 1: Filter FNSPID headlines to the 5 thesis stocks and save as parquet.

Input:  data/raw/nasdaq_exteral_data.csv   (large; gitignored)
Output: data/processed/fnspid_5stocks.parquet
"""

from pathlib import Path
import pandas as pd

# --- Paths (anchored to repo root regardless of where Python is invoked) ---
SCRIPT_DIR      = Path(__file__).parent          # .../scripts/
REPO_ROOT       = SCRIPT_DIR.parent              # .../thesis-llm-arima/
DATA_RAW        = REPO_ROOT / "data" / "raw"
DATA_PROCESSED  = REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------

TICKERS = {"AAPL", "TSLA", "XOM", "MRK", "MS"}
chunks = []

for chunk in pd.read_csv(
    DATA_RAW / "nasdaq_exteral_data.csv",
    chunksize=100_000,
    usecols=["Date", "Stock_symbol", "Article_title"],
    dtype=str,
    low_memory=False,
):
    filtered = chunk[chunk["Stock_symbol"].isin(TICKERS)]
    if not filtered.empty:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
df = df.dropna(subset=["Date", "Article_title"])

output_path = DATA_PROCESSED / "fnspid_5stocks.parquet"
df.to_parquet(output_path, index=False)
print(f"Kept {len(df):,} rows across {df['Stock_symbol'].nunique()} tickers")
print(f"Saved to: {output_path}\n")

# === Diagnostics on the new ticker set ===

print("=== Date range per stock ===")
for t in sorted(TICKERS):
    sub = df[df["Stock_symbol"] == t]
    print(f"  {t}: {len(sub):>5} articles, "
          f"{sub['Date'].min().date()} to {sub['Date'].max().date()}")

print("\n=== Daily headline counts per stock (summary stats) ===")
daily_counts = (
    df.groupby([df["Date"].dt.date, "Stock_symbol"])
      .size()
      .unstack(fill_value=0)
)
print(daily_counts.describe())

print("\n=== Monthly totals per stock (2023) ===")
monthly = (
    df.groupby([df["Date"].dt.to_period("M"), "Stock_symbol"])
      .size()
      .unstack(fill_value=0)
)
print(monthly.loc["2023-01":"2023-12"])
