"""
Full LLM sweep: 4 models × 954 records.
Sequential per model, with append-mode CSV checkpoints so crashes don't lose work.
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

INPUT_FILE = script_dir / "llm_input.jsonl"

MODELS = [
    ("mistral",    "mistralai/mistral-nemo"),
    ("qwen",       "qwen/qwen-2.5-7b-instruct"),
    ("gemma2",     "google/gemma-2-27b-it"),
    ("claude",     "anthropic/claude-3.5-haiku"),
]

# --- Prompt (locked, identical to pilot) ---
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
        "max_tokens": 150,
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
    direction = str(obj.get("direction", "")).strip().upper()
    if direction not in {"UP", "DOWN"}: return None
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "predicted_label": direction,
        "predicted_direction": 1 if direction == "UP" else -1,
        "confidence": conf,
        "reasoning": str(obj.get("reasoning", ""))[:200],
    }

# --- Load all records ---
all_records = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8")]
print(f"Total records to process per model: {len(all_records)}")

# --- Process each model sequentially ---
for model_short, model_id in MODELS:
    output_file = script_dir / f"predictions_{model_short}.csv"
    
    # Resume logic: read existing CSV and skip already-done records
    done_keys = set()
    if output_file.exists():
        existing = pd.read_csv(output_file)
        done_keys = set(zip(existing["stock_symbol"], existing["trading_day"]))
        print(f"\n=== {model_short} ({model_id}) ===")
        print(f"  Resuming: {len(done_keys)} records already done")
    else:
        print(f"\n=== {model_short} ({model_id}) ===")
        print(f"  Starting fresh")
    
    todo = [r for r in all_records 
            if (r["stock_symbol"], r["trading_day"]) not in done_keys]
    print(f"  Remaining: {len(todo)}")
    
    if not todo:
        print(f"  Nothing to do — skipping")
        continue
    
    t_start = time.time()
    total_in = total_out = fails = 0
    batch = []
    
    for i, rec in enumerate(todo, 1):
        prompt = build_prompt(rec["headlines"])
        raw, usage = call_openrouter(model_id, SYSTEM_PROMPT, prompt)
        parsed = parse_response(raw)
        
        total_in  += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)
        
        row = {
            "model": model_short,
            "stock_symbol": rec["stock_symbol"],
            "trading_day": rec["trading_day"],
            "label_date": rec["label_date"],
            "actual_direction": rec["actual_direction"],
            "n_headlines": rec["n_headlines"],
            "split": rec["split"],
            "raw_response": (raw or "")[:300],
        }
        if parsed:
            row.update(parsed)
        else:
            fails += 1
            row.update({"predicted_label": None, "predicted_direction": None,
                        "confidence": None, "reasoning": None})
        batch.append(row)
        
        # Checkpoint: flush every 25 records
        if len(batch) >= 25 or i == len(todo):
            pd.DataFrame(batch).to_csv(
                output_file, mode="a", header=not output_file.exists(),
                index=False
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

print("\n=== Full sweep complete ===")
for model_short, _ in MODELS:
    f = script_dir / f"predictions_{model_short}.csv"
    if f.exists():
        n = len(pd.read_csv(f))
        print(f"  {model_short}: {n} predictions in {f.name}")