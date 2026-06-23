# Can LLMs Beat ARIMA? Sentiment-Driven Stock Direction Prediction

**Bachelor's thesis — Artificial Intelligence, Utrecht University**  
Yoran de Jongh · 2024

---

## Overview

This project tests whether large language models (LLMs) can predict next-day stock price direction more accurately than an ARIMA baseline by classifying the sentiment of anonymised financial news headlines. Four open-weight LLMs (Mistral, Qwen, Gemma 2, Claude) are evaluated against a walk-forward ARIMA model on five S&P 500 stocks across a 7-month window (Jan–Aug 2023).

---

## Repository structure

```
thesis-llm-arima/
├── data/
│   ├── raw/              # Source data (gitignored — download separately)
│   │   ├── nasdaq_exteral_data.csv       # FNSPID headline dataset
│   │   └── FinancialPhraseBank-v1.0/    # FPB sentiment benchmark
│   ├── processed/        # Cleaned, intermediate datasets
│   │   ├── fnspid_5stocks.parquet        # Filtered headlines (5 tickers)
│   │   ├── llm_input.jsonl               # Anonymised LLM input records
│   │   ├── fpb_dataset.csv               # Parsed Financial PhraseBank
│   │   └── fpb_sample.csv                # Stratified 500-sentence FPB sample
│   └── predictions/      # Model output files
│       ├── arima_predictions.csv
│       ├── predictions_{mistral,qwen,gemma2,claude}.csv
│       ├── fpb_predictions_{mistral,qwen,gemma2,claude}.csv
│       └── pilot_predictions.csv
├── scripts/              # Run in order
│   ├── 01_data_text.py       # Filter FNSPID to 5 tickers → parquet
│   ├── 02_input_llm.py       # Anonymise + align headlines → JSONL
│   ├── 03_arima.py           # Walk-forward ARIMA baseline
│   ├── 04_pilot_run.py       # Pilot LLM run (TSLA train, 50 records)
│   ├── 05_full_llm_sweep.py  # Full 4-model × 954-record LLM sweep
│   ├── 06_mcnemar_test.py    # McNemar's test + bootstrap CIs
│   ├── 07_sharpe_ratio.py    # Sharpe ratio backtest
│   ├── 08_sanity_check.py    # Prediction sanity checks
│   ├── 09_verify_anon.py     # Anonymisation leakage check
│   ├── 10_fpb_build.py       # Parse Financial PhraseBank → CSV
│   ├── 11_fpb_test.py        # LLM sentiment classification on FPB
│   └── 12_analyze_fpb.py     # FPB accuracy + confusion matrices
├── results/              # Statistical output tables
│   ├── mcnemar_results.csv
│   ├── bootstrap_results.csv
│   ├── sharpe_per_stock.csv
│   ├── sharpe_pooled.csv
│   ├── sharpe_buy_and_hold.csv
│   └── fpb_summary.csv
├── .env                  # API key (gitignored — create locally)
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/thesis-llm-arima.git
cd thesis-llm-arima

# 2. Install dependencies
pip install pandas numpy yfinance statsmodels scikit-learn \
            requests python-dotenv pyarrow

# 3. Add your API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# 4. Download raw data (not tracked in git)
#    - FNSPID: place nasdaq_exteral_data.csv in data/raw/
#    - FPB:    extract FinancialPhraseBank-v1.0/ into data/raw/
```

---

## Running the pipeline

Scripts are numbered and self-contained. Run them in order:

```bash
python scripts/01_data_text.py       # ~1 min
python scripts/02_input_llm.py       # ~2 min (downloads prices)
python scripts/03_arima.py           # ~10–15 min (walk-forward ARIMA)
python scripts/04_pilot_run.py       # ~5 min (optional sanity check)
python scripts/05_full_llm_sweep.py  # ~3–5 hours (API calls, resumable)
python scripts/06_mcnemar_test.py    # <1 min
python scripts/07_sharpe_ratio.py    # ~2 min (bootstrap)
python scripts/08_sanity_check.py    # <1 min
python scripts/10_fpb_build.py       # <1 min
python scripts/11_fpb_test.py        # ~1–2 hours (API calls, resumable)
python scripts/12_analyze_fpb.py     # <1 min
```

Scripts 05 and 11 are crash-safe: they checkpoint every 25 records and resume automatically if interrupted.

---

## Models evaluated

| Short name | OpenRouter model ID               |
|------------|----------------------------------|
| mistral    | mistralai/mistral-nemo           |
| qwen       | qwen/qwen-2.5-7b-instruct        |
| gemma2     | google/gemma-2-27b-it            |
| claude     | anthropic/claude-3.5-haiku       |
| arima      | ARIMA(p,0,q) — per-stock AIC fit |

---

## Data sources

- **FNSPID** — Financial News and Stock Price Integration Dataset (Nasdaq)
- **Financial PhraseBank** — Malo et al. (2014), 75%-agreement subset
- **Price data** — Yahoo Finance via `yfinance`

---

## License

Code: MIT. Data files follow their respective source licences.
