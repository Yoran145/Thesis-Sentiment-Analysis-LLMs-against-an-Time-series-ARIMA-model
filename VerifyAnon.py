import json
import random
from pathlib import Path

script_dir = Path(__file__).parent
records = [json.loads(l) for l in open(script_dir / "llm_input.jsonl", encoding="utf-8")]
by_stock = {t: [r for r in records if r["stock_symbol"] == t] 
            for t in ["AAPL","TSLA","XOM","MRK","MS"]}

random.seed(42)
for t, recs in by_stock.items():
    print(f"\n=== {t} — 3 random samples ===")
    for r in random.sample(recs, k=min(3, len(recs))):
        print(f"\n  {r['trading_day']} ({len(r['headlines'])} headlines, "
              f"direction={r['actual_direction']}):")
        for h in r["headlines"][:5]:
            print(f"    - {h}")

# Leakage check: scan ALL records for direct company name mentions
print("\n\n=== LEAKAGE CHECK ===")
leak_terms = {
    "AAPL": ["Apple", "iPhone", "Tim Cook", "AAPL", "Cupertino", "Mac"],
    "TSLA": ["Tesla", "Musk", "Cybertruck", "TSLA", "Model 3", "Model Y"],
    "XOM":  ["Exxon", "ExxonMobil", "XOM", "Darren Woods"],
    "MRK":  ["Merck", "Keytruda", "MRK", "Robert Davis"],
    "MS":   ["Morgan Stanley", "Gorman", "Ted Pick"],
}
for t, recs in by_stock.items():
    print(f"\n{t}:")
    for term in leak_terms[t]:
        count = sum(1 for r in recs for h in r["headlines"] 
                    if term.lower() in h.lower())
        marker = "❌ LEAK" if count > 0 else "✓ clean"
        print(f"  {marker} '{term}': {count} occurrences")


records = [json.loads(l) for l in open(Path(__file__).parent / "llm_input.jsonl", encoding="utf-8")]

print("=== All headlines containing 'Mac' (case-insensitive) ===")
for r in records:
    if r["stock_symbol"] != "AAPL":
        continue
    for h in r["headlines"]:
        if "mac" in h.lower():
            print(f"  [{r['trading_day']}] {h}")