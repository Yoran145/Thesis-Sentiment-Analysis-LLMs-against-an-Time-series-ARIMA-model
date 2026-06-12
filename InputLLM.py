"""
Prepares the LLM input dataset:
1. Loads filtered FNSPID headlines.
2. Anonymizes stock-identifying entities.
3. Aggregates headlines per (stock, trading day), capped at 10.
4. Aligns with next-trading-day returns for ground truth.
5. Saves a single JSONL file ready for the LLM pipeline.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import yfinance as yf

# --- Setup ---
script_dir = Path(__file__).parent
headlines_path = script_dir / "fnspid_5stocks.parquet"
output_path = script_dir / "llm_input.jsonl"

TICKERS = ["AAPL", "TSLA", "XOM", "MRK", "MS"]
TRAIN_START = "2023-01-01"
TEST_START  = "2023-05-01"
TEST_END    = "2023-08-01"
HEADLINES_PER_DAY_CAP = 10

# --- 1. Anonymization dictionary ---
# Each stock's entities map to a CONSISTENT generic placeholder.
# Order matters: longer/more-specific terms FIRST so they replace before
# substrings get caught (e.g. "Tim Cook" before "Apple").
ANONYMIZATION = {
    "AAPL": [
        ("Tim Cook", "CEO"),
        ("Apple Inc.", "COMPANY"), ("Apple Inc", "COMPANY"), ("Apple", "COMPANY"),
        ("AAPL", "TICKER"),
        ("iPhone", "PRODUCT"), ("iPad", "PRODUCT"), ("MacBook", "PRODUCT"),
        ("Mac ", "PRODUCT "), ("AirPods", "PRODUCT"), ("Vision Pro", "PRODUCT"),
        ("App Store", "PRODUCT"), ("iOS", "PRODUCT"), ("macOS", "PRODUCT"),
        ("Cupertino", "HEADQUARTERS"),
    ],
    "TSLA": [
        ("Elon Musk", "CEO"), ("Musk", "CEO"),
        ("Tesla, Inc.", "COMPANY"), ("Tesla Inc", "COMPANY"), ("Tesla", "COMPANY"),
        ("TSLA", "TICKER"),
        ("Cybertruck", "PRODUCT"), ("Model S", "PRODUCT"), ("Model 3", "PRODUCT"),
        ("Model X", "PRODUCT"), ("Model Y", "PRODUCT"), ("Roadster", "PRODUCT"),
        ("Powerwall", "PRODUCT"), ("Autopilot", "PRODUCT"),
        ("Full Self-Driving", "PRODUCT"), ("FSD", "PRODUCT"),
        ("Gigafactory", "FACILITY"), ("Fremont", "FACILITY"),
    ],
    "XOM": [
        ("Darren Woods", "CEO"),
        ("Exxon Mobil", "COMPANY"), ("ExxonMobil", "COMPANY"), ("Exxon", "COMPANY"),
        ("XOM", "TICKER"),
        ("Mobil", "PRODUCT"), ("Esso", "PRODUCT"),
        ("Permian", "REGION"), ("Guyana", "REGION"),
    ],
    "MRK": [
        ("Robert Davis", "CEO"), ("Rob Davis", "CEO"),
        ("Merck & Co.", "COMPANY"), ("Merck & Co", "COMPANY"),
        ("Merck", "COMPANY"),
        ("MRK", "TICKER"),
        ("Keytruda", "PRODUCT"), ("Gardasil", "PRODUCT"),
        ("Lagevrio", "PRODUCT"), ("Januvia", "PRODUCT"),
        ("Rahway", "HEADQUARTERS"),
    ],
    "MS": [
        ("James Gorman", "CEO"), ("Ted Pick", "CEO"), ("Gorman", "CEO"),
        ("Morgan Stanley", "COMPANY"),
        (" MS ", " TICKER "),  # padded to avoid matching inside words
        ("E*TRADE", "PRODUCT"), ("E-Trade", "PRODUCT"),
        ("Eaton Vance", "SUBSIDIARY"),
    ],
}

def anonymize(text: str, ticker: str) -> str:
    """Apply per-stock replacements, then case-insensitive cleanup."""
    if not isinstance(text, str):
        return ""
    out = text
    for original, replacement in ANONYMIZATION[ticker]:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        out = pattern.sub(replacement, out)
    return out

# --- 2. Load and anonymize headlines ---
print("Loading headlines...")
df = pd.read_parquet(headlines_path)
df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("US/Eastern")

# News published after 4 PM ET should predict the NEXT trading day
# (markets close at 4 PM ET). Shift after-hours news to the next day.
def trading_day(ts):
    if ts.hour >= 16:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()

df["trading_day"] = df["Date"].apply(trading_day)
df["anon_title"] = df.apply(lambda r: anonymize(r["Article_title"], r["Stock_symbol"]), axis=1)

# Restrict to study window
df = df[(df["trading_day"] >= pd.to_datetime(TRAIN_START).date()) &
        (df["trading_day"] <= pd.to_datetime(TEST_END).date())]

# --- 3. Aggregate per (stock, trading day), capped ---
print("Aggregating headlines per stock-day...")
grouped = (
    df.sort_values("Date")
      .groupby(["Stock_symbol", "trading_day"])["anon_title"]
      .apply(lambda s: list(s.head(HEADLINES_PER_DAY_CAP)))
      .reset_index()
)
grouped.columns = ["Stock_symbol", "trading_day", "headlines"]

# --- 4. Compute next-day directional ground truth ---
print("Computing ground-truth labels from price data...")
prices = yf.download(TICKERS, start=TRAIN_START, end=TEST_END,
                     auto_adjust=True, progress=False)["Close"]
log_returns = np.log(prices / prices.shift(1)).dropna()

labels = []
for _, row in grouped.iterrows():
    t = row["Stock_symbol"]
    d = pd.Timestamp(row["trading_day"])
    # Find the next trading day at or after d (handles weekends/holidays)
    future = log_returns.loc[log_returns.index >= d, t]
    if len(future) == 0:
        labels.append({"actual_log_return": None, "actual_direction": None,
                       "label_date": None})
        continue
    label_date = future.index[0]
    ret = float(future.iloc[0])
    labels.append({
        "actual_log_return": ret,
        "actual_direction":  1 if ret >= 0 else -1,
        "label_date": label_date.date().isoformat(),
    })

labels_df = pd.DataFrame(labels)
final = pd.concat([grouped.reset_index(drop=True), labels_df], axis=1)
final = final.dropna(subset=["actual_direction"])

print(f"\n=== Final dataset ===")
print(f"Total (stock, day) records: {len(final)}")
print(final.groupby("Stock_symbol").size().rename("records"))

# Split flag
final["split"] = np.where(
    pd.to_datetime(final["trading_day"]) < pd.to_datetime(TEST_START),
    "train", "test"
)
print("\nSp lit counts:")
print(final.groupby(["Stock_symbol", "split"]).size().unstack())

# --- 5. Save as JSONL (one record per line, easy to stream) ---
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in final.iterrows():
        record = {
            "stock_symbol": row["Stock_symbol"],
            "trading_day": str(row["trading_day"]),
            "label_date": row["label_date"],
            "headlines": row["headlines"],
            "n_headlines": len(row["headlines"]),
            "actual_log_return": row["actual_log_return"],
            "actual_direction": int(row["actual_direction"]),
            "split": row["split"],
        }
        f.write(json.dumps(record) + "\n")

print(f"\nSaved to: {output_path}")
print(f"Records: {len(final)}")