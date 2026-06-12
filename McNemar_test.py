"""
Statistical comparison of LLMs vs ARIMA baseline (and LLMs vs each other).

Tests performed:
  1. McNemar's test on paired binary directional predictions
  2. Bootstrap 95% CIs on balanced accuracy and F1-macro

Comparisons are restricted to the intersection of test records where ALL
five models produced a valid prediction (handles parse failures cleanly).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

script_dir = Path(__file__).parent
MODELS = ["mistral", "qwen", "gemma2", "claude"]
N_BOOT = 5000
RNG = np.random.default_rng(42)

# --- Load all predictions ---
print("Loading prediction files...")
arima = pd.read_csv(script_dir / "arima_predictions.csv")
arima = arima.rename(columns={"Date": "label_date"})
arima["label_date"] = pd.to_datetime(arima["label_date"]).dt.date.astype(str)
arima = arima[["label_date", "Stock_symbol",
               "predicted_direction", "actual_direction"]].rename(
    columns={"Stock_symbol": "stock_symbol",
             "predicted_direction": "pred_arima"})

def collapse_to_one_per_day(df, model_name):
    """Majority vote across multiple headline-days mapping to same label_date.
    Ties default to +1 (UP)."""
    df = df[df["split"] == "test"].dropna(subset=["predicted_direction"]).copy()
    df["label_date"] = pd.to_datetime(df["label_date"]).dt.date.astype(str)
    df["predicted_direction"] = df["predicted_direction"].astype(int)
    
    # Group by (stock, label_date), aggregate: sum > 0 → UP, sum < 0 → DOWN, == 0 → UP
    agg = (df.groupby(["stock_symbol", "label_date"])
             .agg(pred_sum=("predicted_direction", "sum"),
                  n_headlines=("predicted_direction", "count"))
             .reset_index())
    agg[f"pred_{model_name}"] = np.where(agg["pred_sum"] >= 0, 1, -1)
    return agg[["stock_symbol", "label_date", f"pred_{model_name}"]]

merged = arima.copy()
for m in MODELS:
    df = pd.read_csv(script_dir / f"predictions_{m}.csv")
    collapsed = collapse_to_one_per_day(df, m)
    print(f"  {m}: {len(df)} raw rows -> {len(collapsed)} collapsed (one per stock-day)")
    merged = merged.merge(collapsed, on=["stock_symbol", "label_date"], how="inner")

# Cast predictions to int
merged["pred_arima"] = merged["pred_arima"].astype(int)
merged["actual_direction"] = merged["actual_direction"].astype(int)
for m in MODELS:
    merged[f"pred_{m}"] = merged[f"pred_{m}"].astype(int)

print(f"Intersection size (all 5 models produced predictions): {len(merged)} records")
print(f"Per-stock breakdown:")
print(merged.groupby("stock_symbol").size().to_string())
print()

# --- McNemar helper ---
def run_mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    """McNemar on paired predictions. Returns (b, c, statistic, p)."""
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    # 2x2 disagreement table:
    #               B correct  B wrong
    # A correct       n11        n10
    # A wrong         n01        n00
    n10 = int(((a_correct) & (~b_correct)).sum())   # A right, B wrong
    n01 = int(((~a_correct) & (b_correct)).sum())   # A wrong, B right
    table = [[0, n10], [n01, 0]]
    # Use exact binomial test when discordant count is small, chi-square otherwise
    use_exact = (n10 + n01) < 25
    result = mcnemar(table, exact=use_exact, correction=True)
    return {
        "comparison": f"{name_a} vs {name_b}",
        "a_only_correct": n10,
        "b_only_correct": n01,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "test": "exact binomial" if use_exact else "chi-square (continuity-corrected)",
    }

# --- All McNemar comparisons ---
print("=" * 80)
print("McNemar's test results")
print("=" * 80)
print("Convention: A vs B. 'A_only_correct' = records where A was right and B wrong.")
print()

y_true = merged["actual_direction"].values
all_models = {"arima": merged["pred_arima"].values}
for m in MODELS:
    all_models[m] = merged[f"pred_{m}"].values

mcnemar_rows = []
# Each LLM vs ARIMA (primary comparisons)
print("--- Primary: LLM vs ARIMA ---")
print(f"{'Comparison':<24} {'A_only':>8} {'B_only':>8} "
      f"{'Statistic':>10} {'p-value':>10}   Test")
print("-" * 88)
for m in MODELS:
    r = run_mcnemar(y_true, all_models[m], all_models["arima"], m, "arima")
    r["category"] = "LLM_vs_ARIMA"
    mcnemar_rows.append(r)
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
          else "*" if r["p_value"] < 0.05 else "ns"
    print(f"{r['comparison']:<24} {r['a_only_correct']:>8} {r['b_only_correct']:>8} "
          f"{r['statistic']:>10.3f} {r['p_value']:>10.4f} {sig:>4}  ({r['test']})")

# Pairwise LLM comparisons
print("\n--- Secondary: LLM vs LLM ---")
print(f"{'Comparison':<24} {'A_only':>8} {'B_only':>8} "
      f"{'Statistic':>10} {'p-value':>10}   Test")
print("-" * 88)
for i, m1 in enumerate(MODELS):
    for m2 in MODELS[i+1:]:
        r = run_mcnemar(y_true, all_models[m1], all_models[m2], m1, m2)
        r["category"] = "LLM_vs_LLM"
        mcnemar_rows.append(r)
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
              else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{r['comparison']:<24} {r['a_only_correct']:>8} {r['b_only_correct']:>8} "
              f"{r['statistic']:>10.3f} {r['p_value']:>10.4f} {sig:>4}  ({r['test']})")

print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# Save McNemar table
pd.DataFrame(mcnemar_rows).to_csv(script_dir / "mcnemar_results.csv", index=False)

# --- Bootstrap CIs ---
print("\n" + "=" * 80)
print(f"Bootstrap 95% CIs (N = {N_BOOT} resamples)")
print("=" * 80)

n = len(merged)
boot_rows = []
print(f"{'Model':<10} {'Bal.Acc':>9} {'95% CI':<22} {'F1-macro':>10} {'95% CI':<22}")
print("-" * 78)

for name, preds in all_models.items():
    bals, f1s = [], []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)  # sample with replacement
        bals.append(balanced_accuracy_score(y_true[idx], preds[idx]))
        f1s.append(f1_score(y_true[idx], preds[idx], average="macro", zero_division=0))
    bals = np.array(bals)
    f1s = np.array(f1s)
    bal_point = balanced_accuracy_score(y_true, preds)
    f1_point  = f1_score(y_true, preds, average="macro", zero_division=0)
    bal_ci = (np.percentile(bals, 2.5), np.percentile(bals, 97.5))
    f1_ci  = (np.percentile(f1s,  2.5), np.percentile(f1s,  97.5))
    print(f"{name:<10} {bal_point:>9.3f} [{bal_ci[0]:.3f}, {bal_ci[1]:.3f}]    "
          f"{f1_point:>10.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
    boot_rows.append({
        "model": name,
        "bal_acc": bal_point, "bal_acc_lo": bal_ci[0], "bal_acc_hi": bal_ci[1],
        "f1_macro": f1_point, "f1_macro_lo": f1_ci[0], "f1_macro_hi": f1_ci[1],
    })

pd.DataFrame(boot_rows).to_csv(script_dir / "bootstrap_results.csv", index=False)

print(f"\nSaved: mcnemar_results.csv, bootstrap_results.csv")
print(f"Intersection N: {len(merged)}")