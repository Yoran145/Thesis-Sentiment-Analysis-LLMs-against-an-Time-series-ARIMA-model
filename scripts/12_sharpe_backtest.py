"""
Step 7: Sharpe ratio backtest for all 5 models (4 LLMs + ARIMA).

Strategy: for each (model, stock, trading day):
    if predicted_direction == +1:  go long  (P&L = +1 * actual_log_return)
    if predicted_direction == -1:  go short (P&L = -1 * actual_log_return)

Sharpe = (mean(daily_returns) - rf_daily) / std(daily_returns) * sqrt(252)
Reported per (model, stock) and pooled, with bootstrap 95% CIs.

Input:  data/processed/llm_input.jsonl
        data/predictions/arima_predictions.csv
        data/predictions/predictions_{model}.csv  (one per LLM)
Output: results/sharpe_per_stock.csv
        results/sharpe_pooled.csv
        results/sharpe_buy_and_hold.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# --- Paths ---
SCRIPT_DIR       = Path(__file__).parent
REPO_ROOT        = SCRIPT_DIR.parent
DATA_PROCESSED   = REPO_ROOT / "data" / "processed"
DATA_PREDICTIONS = REPO_ROOT / "data" / "predictions"
RESULTS          = REPO_ROOT / "results"

# ---------------------------------------------------------------------------

MODELS = ["mistral", "qwen", "gemma2", "claude"]
TICKERS = ["AAPL", "TSLA", "XOM", "MRK", "MS"]

ANNUAL_RF = 0.05
RF_DAILY = ANNUAL_RF / 252
TRADING_DAYS_PER_YEAR = 252

N_BOOT = 5000
RNG = np.random.default_rng(42)

# --- Load ground-truth log returns ---
print("Loading ground-truth returns from llm_input.jsonl...")
records = [json.loads(l) for l in open(DATA_PROCESSED / "llm_input.jsonl", encoding="utf-8")]
truth = pd.DataFrame([{
    "stock_symbol":      r["stock_symbol"],
    "trading_day":       r["trading_day"],
    "label_date":        r["label_date"],
    "actual_log_return": r["actual_log_return"],
    "actual_direction":  int(r["actual_direction"]),
    "split":             r["split"],
} for r in records])

# --- ARIMA predictions ---
print("Loading ARIMA predictions...")
arima = pd.read_csv(DATA_PREDICTIONS / "arima_predictions.csv")
arima = arima.rename(columns={"Date": "label_date", "Stock_symbol": "stock_symbol"})
arima["label_date"] = pd.to_datetime(arima["label_date"]).dt.date.astype(str)
arima = arima[["stock_symbol", "label_date",
               "predicted_direction", "actual_log_return", "actual_direction"]]
arima["model"] = "arima"

# --- LLM predictions, collapsed to one per (stock, trading_day) ---
def collapse(df, model_name, truth_df):
    df = df[df["split"] == "test"].dropna(subset=["predicted_direction"]).copy()
    df["trading_day"] = df["trading_day"].astype(str)
    df["predicted_direction"] = df["predicted_direction"].astype(int)
    agg = (df.groupby(["stock_symbol", "trading_day"])
             .agg(pred_sum=("predicted_direction", "sum"))
             .reset_index())
    agg["predicted_direction"] = np.where(agg["pred_sum"] >= 0, 1, -1)
    t = truth_df[truth_df["split"] == "test"][[
        "stock_symbol", "trading_day", "label_date",
        "actual_log_return", "actual_direction"
    ]].drop_duplicates(subset=["stock_symbol", "trading_day"])
    out = agg.merge(t, on=["stock_symbol", "trading_day"], how="inner")
    out["model"] = model_name
    return out[["model", "stock_symbol", "label_date",
                "predicted_direction", "actual_log_return", "actual_direction"]]

print("Loading and collapsing LLM predictions...")
frames = [arima]
for m in MODELS:
    df = pd.read_csv(DATA_PREDICTIONS / f"predictions_{m}.csv")
    collapsed = collapse(df, m, truth)
    print(f"  {m}: {len(collapsed)} predictions after collapse + truth merge")
    frames.append(collapsed)

all_preds = pd.concat(frames, ignore_index=True)
all_preds["strategy_return"] = (
    all_preds["predicted_direction"] * all_preds["actual_log_return"]
)

# --- Sharpe helpers ---
def sharpe(returns, rf_daily=RF_DAILY, periods=TRADING_DAYS_PER_YEAR):
    excess = returns - rf_daily
    s = excess.std(ddof=1)
    if s == 0 or len(returns) < 2:
        return float("nan")
    return excess.mean() / s * np.sqrt(periods)

def bootstrap_sharpe_ci(returns, n_boot=N_BOOT, rng=RNG):
    n = len(returns)
    if n < 5:
        return (float("nan"), float("nan"))
    arr = np.asarray(returns)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = sharpe(arr[idx])
        if not np.isnan(s):
            samples.append(s)
    if not samples:
        return (float("nan"), float("nan"))
    samples = np.array(samples)
    return (np.percentile(samples, 2.5), np.percentile(samples, 97.5))

# --- Per-(model, stock) Sharpe ---
print("\n" + "="*80)
print("Sharpe ratio per (model, stock)")
print("="*80)
print(f"{'Model':<10} {'Stock':<6} {'N':>5} {'Mean ret':>10} "
      f"{'Std':>8} {'Sharpe':>8} {'95% CI':<24}")
print("-"*80)

per_stock_rows = []
for model_name in ["arima"] + MODELS:
    for t in TICKERS:
        sub = all_preds[(all_preds["model"] == model_name) &
                        (all_preds["stock_symbol"] == t)]
        if len(sub) == 0:
            continue
        rets = sub["strategy_return"].values
        s_point = sharpe(rets)
        ci_lo, ci_hi = bootstrap_sharpe_ci(rets)
        print(f"{model_name:<10} {t:<6} {len(rets):>5} "
              f"{rets.mean():>10.5f} {rets.std(ddof=1):>8.5f} "
              f"{s_point:>8.3f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}]")
        per_stock_rows.append({
            "model": model_name, "stock": t, "n": len(rets),
            "mean_daily_return": rets.mean(),
            "std_daily_return":  rets.std(ddof=1),
            "sharpe": s_point, "sharpe_ci_lo": ci_lo, "sharpe_ci_hi": ci_hi,
        })

pd.DataFrame(per_stock_rows).to_csv(RESULTS / "sharpe_per_stock.csv", index=False)

# --- Pooled Sharpe per model ---
print("\n" + "="*80)
print("Pooled Sharpe ratio per model (all stocks combined)")
print("="*80)
print(f"{'Model':<10} {'N':>5} {'Mean ret':>10} {'Std':>8} "
      f"{'Sharpe':>8} {'95% CI':<24}")
print("-"*80)

pooled_rows = []
for model_name in ["arima"] + MODELS:
    sub = all_preds[all_preds["model"] == model_name]
    rets = sub["strategy_return"].values
    s_point = sharpe(rets)
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets)
    print(f"{model_name:<10} {len(rets):>5} "
          f"{rets.mean():>10.5f} {rets.std(ddof=1):>8.5f} "
          f"{s_point:>8.3f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}]")
    pooled_rows.append({
        "model": model_name, "n": len(rets),
        "mean_daily_return": rets.mean(),
        "std_daily_return":  rets.std(ddof=1),
        "sharpe": s_point, "sharpe_ci_lo": ci_lo, "sharpe_ci_hi": ci_hi,
    })

pd.DataFrame(pooled_rows).to_csv(RESULTS / "sharpe_pooled.csv", index=False)

# --- Buy-and-hold benchmark per stock ---
print("\n" + "="*80)
print("Buy-and-hold Sharpe per stock (benchmark, no model)")
print("="*80)
print(f"{'Stock':<6} {'N':>5} {'Mean ret':>10} {'Std':>8} "
      f"{'Sharpe':>8} {'95% CI':<24}")
print("-"*80)

bh_rows = []
for t in TICKERS:
    sub = arima[arima["stock_symbol"] == t]
    rets = sub["actual_log_return"].values
    s_point = sharpe(rets)
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets)
    print(f"{t:<6} {len(rets):>5} "
          f"{rets.mean():>10.5f} {rets.std(ddof=1):>8.5f} "
          f"{s_point:>8.3f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}]")
    bh_rows.append({
        "stock": t, "n": len(rets),
        "mean_daily_return": rets.mean(),
        "std_daily_return":  rets.std(ddof=1),
        "sharpe": s_point, "sharpe_ci_lo": ci_lo, "sharpe_ci_hi": ci_hi,
    })

# --- Robustness: pooled Sharpe excluding TSLA ---
print("\n" + "="*80)
print("Robustness: pooled Sharpe per model (excluding TSLA)")
print("="*80)
print(f"{'Model':<10} {'N':>5} {'Sharpe':>8} {'95% CI':<24}")
print("-"*80)
for model_name in ["arima"] + MODELS:
    sub = all_preds[(all_preds["model"] == model_name) &
                    (all_preds["stock_symbol"] != "TSLA")]
    rets = sub["strategy_return"].values
    s_point = sharpe(rets)
    ci_lo, ci_hi = bootstrap_sharpe_ci(rets)
    print(f"{model_name:<10} {len(rets):>5} {s_point:>8.3f} "
          f"[{ci_lo:>6.3f}, {ci_hi:>6.3f}]")

pd.DataFrame(bh_rows).to_csv(RESULTS / "sharpe_buy_and_hold.csv", index=False)

print(f"\nSaved: sharpe_per_stock.csv, sharpe_pooled.csv, sharpe_buy_and_hold.csv → {RESULTS}")
