"""
Parse the manually-downloaded FPB file into a clean CSV.
Run once. Produces fpb_dataset.csv used by fpb_sweep.py.
"""

from pathlib import Path
import pandas as pd

script_dir = Path(__file__).parent

# Path to the extracted file (adjust if you put it elsewhere)
FPB_FILE = script_dir / "FinancialPhraseBank-v1.0" / "Sentences_75Agree.txt"

LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_TO_STR = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

print(f"Reading: {FPB_FILE}")
if not FPB_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {FPB_FILE}. Make sure you extracted the zip "
        f"into the thesis folder."
    )

# FPB files are Latin-1 encoded (Scandinavian source data)
with open(FPB_FILE, "r", encoding="latin-1") as f:
    text = f.read()

rows = []
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    # Format: "sentence text@label" — split on LAST @ in case sentence has one
    sep = line.rfind("@")
    if sep == -1:
        continue
    sentence = line[:sep].strip()
    label_str = line[sep+1:].strip().lower()
    if label_str not in LABEL_TO_INT:
        continue
    label_int = LABEL_TO_INT[label_str]
    rows.append({
        "sentence": sentence,
        "label": label_int,
        "gold_sentiment": LABEL_TO_STR[label_int],
    })

df = pd.DataFrame(rows)
out_csv = script_dir / "fpb_dataset.csv"
df.to_csv(out_csv, index=False)

print(f"\nSaved {len(df)} sentences to: {out_csv}")
print(f"\nClass distribution:")
print(df["gold_sentiment"].value_counts())
print(f"\nSample sentences (random 5):")
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"  [{row['gold_sentiment']}] {row['sentence'][:120]}")