"""
Pilot run: Mistral 7B Instruct on TSLA train records via OpenRouter.
Goal: verify prompt, parsing, cost, and class balance before full sweep.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
import pandas as pd

# --- Setup ---
script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")

INPUT_FILE  = script_dir / "llm_input.jsonl"
OUTPUT_FILE = script_dir / "pilot_predictions.csv"

MODEL = "anthropic/claude-3.5-haiku"
PILOT_STOCK = "TSLA"
PILOT_SPLIT = "train"   # use train split for pilot — preserves test set integrity
PILOT_N = 50            # number of records to predict on

# --- Prompt template ---
SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. You read anonymized news "
    "headlines about a publicly traded company and predict the likely "
    "directional movement of its stock price on the next trading day. "
    "Respond ONLY with valid JSON. Do not include any other text."
)

USER_PROMPT_TEMPLATE = """The following headlines were published about an anonymized company. Based ONLY on the sentiment and content of these headlines, predict the likely directional movement of the company's stock on the next trading day.

Headlines:
{headlines_block}

Respond with a JSON object in this exact format, and nothing else:
{{"direction": "UP" or "DOWN", "confidence": 0.0 to 1.0, "reasoning": "one short sentence"}}"""

def build_prompt(headlines):
    block = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return USER_PROMPT_TEMPLATE.format(headlines_block=block)

# --- API call ---
def call_openrouter(system_prompt, user_prompt, model=MODEL, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 150,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=30
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{max_retries} after error: {e} "
                  f"(waiting {wait}s)")
            time.sleep(wait)
    return None, {}

# --- Parsing ---
def parse_response(raw_text):
    """Extract the JSON object even if the model wraps it in extra text."""
    if not raw_text:
        return None
    # Find the first { ... } block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(raw_text[start:end+1])
    except json.JSONDecodeError:
        return None
    direction = str(obj.get("direction", "")).strip().upper()
    if direction not in {"UP", "DOWN"}:
        return None
    try:
        confidence = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "predicted_label": direction,
        "predicted_direction": 1 if direction == "UP" else -1,
        "confidence": confidence,
        "reasoning": str(obj.get("reasoning", ""))[:200],
    }

# --- Load data ---
records = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8")]
pilot = [r for r in records 
         if r["stock_symbol"] == PILOT_STOCK and r["split"] == PILOT_SPLIT]
pilot = pilot[:PILOT_N]

print(f"Pilot: {len(pilot)} records ({PILOT_STOCK}, {PILOT_SPLIT})")
print(f"Model: {MODEL}")
print(f"Output: {OUTPUT_FILE}\n")

# --- Run ---
results = []
total_tokens_in = total_tokens_out = parse_fails = 0
t_start = time.time()

for i, rec in enumerate(pilot, 1):
    prompt = build_prompt(rec["headlines"])
    raw, usage = call_openrouter(SYSTEM_PROMPT, prompt)
    parsed = parse_response(raw)
    
    total_tokens_in  += usage.get("prompt_tokens", 0)
    total_tokens_out += usage.get("completion_tokens", 0)
    
    row = {
        "stock_symbol": rec["stock_symbol"],
        "trading_day": rec["trading_day"],
        "label_date": rec["label_date"],
        "actual_direction": rec["actual_direction"],
        "n_headlines": rec["n_headlines"],
        "raw_response": (raw or "")[:300],
    }
    if parsed:
        row.update(parsed)
    else:
        parse_fails += 1
        row.update({"predicted_label": None, "predicted_direction": None,
                    "confidence": None, "reasoning": None})
    results.append(row)
    
    status = parsed["predicted_label"] if parsed else "PARSE_FAIL"
    print(f"  [{i:2d}/{len(pilot)}] {rec['trading_day']} -> {status}")

elapsed = time.time() - t_start

# --- Save ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

# --- Diagnostics ---
print(f"\n=== Pilot summary ===")
print(f"  Records:       {len(df)}")
print(f"  Parse fails:   {parse_fails} ({parse_fails/len(df)*100:.1f}%)")
print(f"  Time elapsed:  {elapsed:.1f}s ({elapsed/len(df):.2f}s/call)")
print(f"  Tokens in:     {total_tokens_in:,}")
print(f"  Tokens out:    {total_tokens_out:,}")

ok = df.dropna(subset=["predicted_direction"])
if len(ok):
    print(f"\n  Prediction distribution:")
    print(ok["predicted_label"].value_counts().to_string())
    
    acc = (ok["predicted_direction"] == ok["actual_direction"]).mean()
    print(f"\n  Train-split accuracy: {acc:.3f} (sanity check only — "
          f"real eval uses test split)")

print(f"\nSaved to: {OUTPUT_FILE}")

import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "pilot_predictions.csv")

print("=== Headlines per prediction class ===")
print(df.groupby("predicted_label")["n_headlines"].describe())

print("\n=== Sample reasoning for DOWN predictions ===")
for _, row in df[df["predicted_label"] == "DOWN"].head(5).iterrows():
    print(f"\n  {row['trading_day']} ({row['n_headlines']} headlines)")
    print(f"  Reasoning: {row['reasoning']}")

print("\n=== Sample reasoning for UP predictions ===")
for _, row in df[df["predicted_label"] == "UP"].head(5).iterrows():
    print(f"\n  {row['trading_day']} ({row['n_headlines']} headlines)")
    print(f"  Reasoning: {row['reasoning']}")