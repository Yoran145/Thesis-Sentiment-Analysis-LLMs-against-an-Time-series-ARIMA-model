"""
Sanity check across all 4 LLM prediction files + ARIMA baseline.
Reports: parse failures, class distribution, raw accuracy by stock,
and flags constant-classifier behavior.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score)

script_dir = Path(__file__).parent
MODELS = ["mistral", "qwen", "gemma2", "claude"]
TICKERS = ["AAPL", "TSLA", "XOM", "MRK", "MS"]

# --- Load all predictions ---
arima = pd.read_csv(script_dir / "arima_predictions.csv")
print(f"ARIMA: {len(arima)} rows")

llm_dfs = {}
for m in MODELS:
    f = script_dir / f"predictions_{m}.csv"
    if not f.exists():
        print(f"  WARNING: {f.name} missing")
        continue
    df = pd.read_csv(f)
    llm_dfs[m] = df
    print(f"{m:<10}: {len(df)} rows")

print("\n" + "="*70)
print("CHECK 1: Parse failure rate per model")
print("="*70)
for m, df in llm_dfs.items():
    n_total = len(df)
    n_failed = df["predicted_direction"].isna().sum()
    pct = n_failed / n_total * 100
    marker = "❌" if pct > 5 else ("⚠️ " if pct > 1 else "✓")
    print(f"  {marker} {m:<10}: {n_failed:>4}/{n_total} failed ({pct:.1f}%)")

print("\n" + "="*70)
print("CHECK 2: Class distribution per model (across all stocks)")
print("="*70)
print("  Healthy: predictions split across UP/DOWN, not 100% one class")
print()
for m, df in llm_dfs.items():
    ok = df.dropna(subset=["predicted_direction"])
    up   = (ok["predicted_direction"] ==  1).sum()
    down = (ok["predicted_direction"] == -1).sum()
    total = up + down
    pct_up = up / total * 100 if total else 0
    marker = "❌" if pct_up > 95 or pct_up < 5 else "✓"
    print(f"  {marker} {m:<10}: UP={up:>4} ({pct_up:.0f}%), DOWN={down:>4} ({100-pct_up:.0f}%)")

# ARIMA for reference
ok = arima.dropna(subset=["predicted_direction"])
up   = (ok["predicted_direction"] ==  1).sum()
down = (ok["predicted_direction"] == -1).sum()
total = up + down
pct_up = up / total * 100 if total else 0
print(f"    {'ARIMA':<10}: UP={up:>4} ({pct_up:.0f}%), DOWN={down:>4} ({100-pct_up:.0f}%) [baseline]")

print("\n" + "="*70)
print("CHECK 3: Class distribution per (model, stock)")
print("="*70)
for m, df in llm_dfs.items():
    print(f"\n  {m}:")
    ok = df.dropna(subset=["predicted_direction"])
    for t in TICKERS:
        sub = ok[ok["stock_symbol"] == t]
        if len(sub) == 0:
            print(f"    {t}: no predictions")
            continue
        up = (sub["predicted_direction"] == 1).sum()
        dn = (sub["predicted_direction"] == -1).sum()
        pct_up = up / (up+dn) * 100
        flag = " ← constant predictor!" if pct_up > 95 or pct_up < 5 else ""
        print(f"    {t}: UP={up:>3} ({pct_up:>3.0f}%), DOWN={dn:>3}{flag}")

print("\n" + "="*70)
print("CHECK 4: Raw metrics per model (test split only)")
print("="*70)
print(f"  {'model':<10} {'acc':>6} {'bal_acc':>8} {'f1_macro':>9}  (test-set only)")
print()

baselines = {"ARIMA": arima}
all_models = {**baselines, **llm_dfs}

for name, df in all_models.items():
    if "split" in df.columns:
        ok = df[df["split"] == "test"].dropna(subset=["predicted_direction"])
    else:
        ok = df.dropna(subset=["predicted_direction"])
    if len(ok) == 0:
        print(f"  {name:<10}: no test predictions")
        continue
    acc = accuracy_score(ok["actual_direction"], ok["predicted_direction"])
    bal = balanced_accuracy_score(ok["actual_direction"], ok["predicted_direction"])
    f1m = f1_score(ok["actual_direction"], ok["predicted_direction"], average="macro")
    print(f"  {name:<10} {acc:>6.3f} {bal:>8.3f} {f1m:>9.3f}  (n={len(ok)})")

print("\n" + "="*70)
print("CHECK 5: Confidence distribution per model")
print("="*70)
print("  Healthy: confidence varies, not stuck at 0.5 or 1.0")
print()
for m, df in llm_dfs.items():
    if "confidence" not in df.columns:
        continue
    c = df["confidence"].dropna()
    if len(c) == 0:
        continue
    print(f"  {m:<10}: mean={c.mean():.2f}, std={c.std():.2f}, "
          f"min={c.min():.2f}, max={c.max():.2f}")

print("\nSanity check complete.")