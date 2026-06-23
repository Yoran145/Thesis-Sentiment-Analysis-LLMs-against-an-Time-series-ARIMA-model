"""
Step 12: FPB Stage 1 analysis.
Reports per-model accuracy, F1 (per class and macro), parse failure rate,
and confusion matrices on the 500-sentence stratified sample.

Input:  data/processed/fpb_sample.csv
        data/predictions/fpb_predictions_{model}.csv  (one per LLM)
Output: results/fpb_summary.csv
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# --- Paths ---
SCRIPT_DIR       = Path(__file__).parent
REPO_ROOT        = SCRIPT_DIR.parent
DATA_PROCESSED   = REPO_ROOT / "data" / "processed"
DATA_PREDICTIONS = REPO_ROOT / "data" / "predictions"
RESULTS          = REPO_ROOT / "results"

# ---------------------------------------------------------------------------

MODELS = ["mistral", "qwen", "gemma2", "claude"]
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# --- Load gold labels from the shared sample ---
sample = pd.read_csv(DATA_PROCESSED / "fpb_sample.csv")
print(f"Sample: {len(sample)} sentences")
print(f"Class distribution:\n{sample['gold_sentiment'].value_counts()}\n")

results = {}

for m in MODELS:
    df = pd.read_csv(DATA_PREDICTIONS / f"fpb_predictions_{m}.csv")
    ok = df.dropna(subset=["predicted_sentiment"])
    n_total = len(df)
    n_ok = len(ok)
    n_fail = n_total - n_ok

    y_true = ok["gold_sentiment"]
    y_pred = ok["predicted_sentiment"]

    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, labels=LABELS, average="macro",    zero_division=0)
    f1w  = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred,    labels=LABELS, average="macro", zero_division=0)

    results[m] = {
        "n_total": n_total,
        "n_failed": n_fail,
        "parse_fail_rate": n_fail / n_total,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "precision_macro": prec,
        "recall_macro": rec,
    }

# --- Headline summary ---
print("="*72)
print("HEADLINE RESULTS (3-class sentiment classification on FPB 75%-agree)")
print("="*72)
print(f"{'Model':<10} {'Acc':>7} {'F1-macro':>9} {'F1-wgt':>8} "
      f"{'Prec(M)':>9} {'Rec(M)':>8} {'Fail%':>7}")
print("-"*72)
for m in MODELS:
    r = results[m]
    print(f"{m:<10} {r['accuracy']:>7.3f} {r['f1_macro']:>9.3f} "
          f"{r['f1_weighted']:>8.3f} {r['precision_macro']:>9.3f} "
          f"{r['recall_macro']:>8.3f} {r['parse_fail_rate']*100:>6.1f}%")

# --- Per-class detail ---
print("\n" + "="*72)
print("PER-CLASS PRECISION / RECALL / F1 (each model)")
print("="*72)
for m in MODELS:
    df = pd.read_csv(DATA_PREDICTIONS / f"fpb_predictions_{m}.csv")
    ok = df.dropna(subset=["predicted_sentiment"])
    print(f"\n--- {m} (n={len(ok)}) ---")
    report = classification_report(
        ok["gold_sentiment"], ok["predicted_sentiment"],
        labels=LABELS, digits=3, zero_division=0
    )
    print(report)

# --- Confusion matrices ---
print("="*72)
print("CONFUSION MATRICES (rows = gold, columns = predicted)")
print("="*72)
for m in MODELS:
    df = pd.read_csv(DATA_PREDICTIONS / f"fpb_predictions_{m}.csv")
    ok = df.dropna(subset=["predicted_sentiment"])
    cm = confusion_matrix(ok["gold_sentiment"], ok["predicted_sentiment"], labels=LABELS)
    print(f"\n--- {m} ---")
    print(f"             pred->")
    print(f"{'gold':<10} {'NEG':>5} {'NEU':>5} {'POS':>5}")
    for i, lab in enumerate(LABELS):
        print(f"{lab:<10} {cm[i][0]:>5} {cm[i][1]:>5} {cm[i][2]:>5}")

# --- Save tidy summary CSV ---
summary_df = pd.DataFrame(results).T
summary_df.to_csv(RESULTS / "fpb_summary.csv")
print(f"\nSaved summary to: {RESULTS / 'fpb_summary.csv'}")
