import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score

warnings.filterwarnings("ignore")  # statsmodels is noisy

# --- Setup ---
TICKERS = ["AAPL", "TSLA", "XOM", "MRK", "MS"]
TRAIN_START = "2023-01-01"
TRAIN_END   = "2023-05-01"
TEST_START  = "2023-05-01"
TEST_END    = "2023-08-01"

# --- Download once, then split ---
prices = yf.download(TICKERS, start=TRAIN_START, end=TEST_END, auto_adjust=True)["Close"]
prices = prices.dropna()

# Log returns: log(P_t / P_{t-1}). Stationary, additive over time, standard in finance.
log_returns = np.log(prices / prices.shift(1)).dropna()

train_returns = log_returns.loc[TRAIN_START:TRAIN_END]
test_returns  = log_returns.loc[TEST_START:TEST_END]

# --- Stationarity check on log returns (sanity check; should be stationary) ---
print("=== ADF stationarity test on log returns ===")
for t in TICKERS:
    stat, p, *_ = adfuller(train_returns[t].dropna())
    verdict = "stationary" if p < 0.05 else "NON-stationary (investigate)"
    print(f"  {t}: ADF={stat:.3f}, p={p:.4f}  -> {verdict}")

# --- Pick best (p,d,q) per stock via AIC grid search on training set ---
# d is fixed at 0 because log returns are already stationary.
def best_order(series, p_max=3, q_max=3):
    best_aic, best = np.inf, (1, 0, 1)
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                aic = ARIMA(series, order=(p, 0, q)).fit().aic
                if aic < best_aic:
                    best_aic, best = aic, (p, 0, q)
            except Exception:
                continue
    return best, best_aic

print("\n=== Best ARIMA order per stock (AIC, fit on training) ===")
orders = {}
for t in TICKERS:
    order, aic = best_order(train_returns[t])
    orders[t] = order
    print(f"  {t}: order={order}, AIC={aic:.2f}")

# --- Walk-forward 1-step-ahead forecast on the test set ---
# For each day in the test period, refit ARIMA on all data up to (but not including)
# that day, forecast next day's log return, record sign as directional prediction.
print("\n=== Walk-forward forecasting (this takes a few minutes) ===")
all_rows = []
per_stock_metrics = {}

for t in TICKERS:
    history = list(train_returns[t].values)
    preds, actuals = [], list(test_returns[t].values)
    test_dates = test_returns.index

    for i, actual in enumerate(actuals):
        try:
            model = ARIMA(history, order=orders[t]).fit()
            yhat = float(model.forecast(steps=1).iloc[0])
        except Exception:
            yhat = 0.0
        preds.append(yhat)
        history.append(actual)  # expanding window: include true value going forward

    pred_dir   = np.where(np.array(preds)    >= 0, 1, -1)
    actual_dir = np.where(np.array(actuals)  >= 0, 1, -1)

    acc = accuracy_score(actual_dir, pred_dir)
    f1m = f1_score(actual_dir, pred_dir, average="macro")
    per_stock_metrics[t] = {"accuracy": acc, "f1_macro": f1m}
    print(f"  {t}: accuracy={acc:.3f}, F1(macro)={f1m:.3f}")

    for date, p_ret, a_ret, p_d, a_d in zip(test_dates, preds, actuals, pred_dir, actual_dir):
        all_rows.append({
            "Date": date,
            "Stock_symbol": t,
            "predicted_log_return": p_ret,
            "actual_log_return": a_ret,
            "predicted_direction": int(p_d),
            "actual_direction": int(a_d),
        })

# --- Aggregate results ---
results_df = pd.DataFrame(all_rows)
from pathlib import Path
out_path = Path(__file__).parent / "arima_predictions.csv"
results_df.to_csv(out_path, index=False)
print(f"\nSaved CSV to: {out_path.resolve()}")

summary = pd.DataFrame(per_stock_metrics).T
print("\n=== Per-stock results ===")
print(summary.round(3))

# Overall (pooled across all stocks and days)
overall_acc = accuracy_score(results_df["actual_direction"], results_df["predicted_direction"])
overall_f1  = f1_score(results_df["actual_direction"], results_df["predicted_direction"], average="macro")
print(f"\n=== Pooled across all stocks ===")
print(f"  Accuracy:    {overall_acc:.3f}")
print(f"  F1 (macro):  {overall_f1:.3f}")
print(f"\nSaved {len(results_df)} predictions to arima_predictions.csv")

results_df = pd.read_csv("arima_predictions.csv")

print("\n=== Prediction distribution per stock ===")
print(pd.crosstab(results_df["Stock_symbol"], results_df["predicted_direction"]))

print("\n=== Actual distribution per stock ===")
print(pd.crosstab(results_df["Stock_symbol"], results_df["actual_direction"]))

#This is the predicted accuracy of the ARIMA model
print("\n=== Final ARIMA metrics ===")
for t in TICKERS:
    sub = results_df[results_df["Stock_symbol"] == t]
    bal = balanced_accuracy_score(sub["actual_direction"], sub["predicted_direction"])
    print(f"  {t}: balanced_accuracy={bal:.3f}")

overall_bal = balanced_accuracy_score(results_df["actual_direction"],
                                       results_df["predicted_direction"])
print(f"\n  Pooled balanced accuracy: {overall_bal:.3f}")