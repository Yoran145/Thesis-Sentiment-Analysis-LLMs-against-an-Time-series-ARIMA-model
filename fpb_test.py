"""
Stage 1 validation: 4 LLMs × ~500 FPB sentences.
Classifies financial sentences as POSITIVE / NEGATIVE / NEUTRAL,
compares to expert gold labels.
"""

import os
import json
import time
import random
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

MODELS = [
    ("mistral",    "mistralai/mistral-nemo"),
    ("qwen",       "qwen/qwen-2.5-7b-instruct"),
    ("gemma2",     "google/gemma-2-27b-it"),
    ("claude",     "anthropic/claude-3.5-haiku"),
]

SAMPLE_SIZE = 500   # sentences to evaluate
RANDOM_SEED = 42

# --- Prompt ---
SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. You read a single financial "
    "news sentence and classify its sentiment regarding the company or asset "
    "discussed. Respond ONLY with valid JSON. Do not include any other text."
)

USER_PROMPT_TEMPLATE = """Classify the sentiment of the following financial sentence as POSITIVE, NEGATIVE, or NEUTRAL.

POSITIVE: the sentence describes favorable conditions, growth, profits, or other positive developments for the company.
NEGATIVE: the sentence describes unfavorable conditions, losses, declines, or other negative developments.
NEUTRAL: the sentence is descriptive, factual, or does not carry clear positive or negative sentiment about the company's prospects.

Sentence: "{sentence}"

Respond with a JSON object in this exact format, and nothing else:
{{"sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL", "reasoning": "one short sentence"}}"""

def build_prompt(sentence):
    return USER_PROMPT_TEMPLATE.format(sentence=sentence)

# --- API call (identical to sweep script) ---
def call_openrouter(model, system_prompt, user_prompt, max_retries=4):
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
        "max_tokens": 100,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=60
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"      retry {attempt+1}/{max_retries}: {e} (waiting {wait}s)")
            time.sleep(wait)
    return None, {}

def parse_response(raw):
    if not raw: return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1: return None
    try:
        obj = json.loads(raw[start:end+1])
    except json.JSONDecodeError:
        return None
    sentiment = str(obj.get("sentiment", "")).strip().upper()
    if sentiment not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        return None
    return {
        "predicted_sentiment": sentiment,
        "reasoning": str(obj.get("reasoning", ""))[:200],
    }

# --- Load FPB ---
print("Loading Financial PhraseBank from local CSV...")
fpb = pd.read_csv(script_dir / "fpb_dataset.csv")

print(f"Full FPB (75% agree): {len(fpb)} sentences")
print(f"  Class distribution:\n{fpb['gold_sentiment'].value_counts()}\n")

# Stratified sample to keep class balance
random.seed(RANDOM_SEED)
sample = (
    fpb.groupby("gold_sentiment", group_keys=False)
       .apply(lambda g: g.sample(min(len(g), SAMPLE_SIZE // 3 + 50),
                                  random_state=RANDOM_SEED))
       .sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
       .reset_index(drop=True)
)
sample["record_id"] = sample.index
print(f"Sample size: {len(sample)}")
print(f"Sample class distribution:\n{sample['gold_sentiment'].value_counts()}\n")

# Save the sample so all models see identical sentences
sample_path = script_dir / "fpb_sample.csv"
sample[["record_id", "sentence", "gold_sentiment"]].to_csv(sample_path, index=False)
print(f"Saved sample to: {sample_path}\n")

# --- Process each model ---
for model_short, model_id in MODELS:
    output_file = script_dir / f"fpb_predictions_{model_short}.csv"

    done_ids = set()
    if output_file.exists():
        existing = pd.read_csv(output_file)
        done_ids = set(existing["record_id"].tolist())
        print(f"\n=== {model_short} ({model_id}) ===")
        print(f"  Resuming: {len(done_ids)} records already done")
    else:
        print(f"\n=== {model_short} ({model_id}) ===")
        print(f"  Starting fresh")

    todo = sample[~sample["record_id"].isin(done_ids)]
    print(f"  Remaining: {len(todo)}")

    if len(todo) == 0:
        print("  Nothing to do — skipping")
        continue

    t_start = time.time()
    total_in = total_out = fails = 0
    batch = []

    for i, (_, rec) in enumerate(todo.iterrows(), 1):
        prompt = build_prompt(rec["sentence"])
        raw, usage = call_openrouter(model_id, SYSTEM_PROMPT, prompt)
        parsed = parse_response(raw)

        total_in  += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)

        row = {
            "model": model_short,
            "record_id": rec["record_id"],
            "sentence": rec["sentence"][:300],
            "gold_sentiment": rec["gold_sentiment"],
            "raw_response": (raw or "")[:300],
        }
        if parsed:
            row.update(parsed)
            row["correct"] = (parsed["predicted_sentiment"] == rec["gold_sentiment"])
        else:
            fails += 1
            row.update({"predicted_sentiment": None, "reasoning": None,
                        "correct": None})
        batch.append(row)

        if len(batch) >= 25 or i == len(todo):
            pd.DataFrame(batch).to_csv(
                output_file, mode="a",
                header=not output_file.exists(), index=False
            )
            batch = []
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(todo) - i) / rate
            print(f"  [{i:4d}/{len(todo)}] saved. "
                  f"{rate:.2f} req/s, ETA {eta/60:.1f} min, fails={fails}")

    elapsed = time.time() - t_start
    print(f"  Done. {elapsed/60:.1f} min, "
          f"tokens in={total_in:,} out={total_out:,}, fails={fails}")

print("\n=== FPB Stage 1 complete ===")
for model_short, _ in MODELS:
    f = script_dir / f"fpb_predictions_{model_short}.csv"
    if f.exists():
        n = len(pd.read_csv(f))
        print(f"  {model_short}: {n} predictions in {f.name}")