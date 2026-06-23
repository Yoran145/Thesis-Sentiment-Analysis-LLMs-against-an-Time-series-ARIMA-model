# LLMs vs ARIMA for Next-Day Stock Movement Prediction

This repository contains the code, data, and analysis outputs for the bachelor's thesis:

**"Comparing LLM-based directional stock prediction against an ARIMA baseline"**
Yoran de Jongh — Utrecht University, Artificial Intelligence — 2026

The thesis evaluates four large language models (Mistral Nemo, Qwen 2.5 7B Instruct, Gemma-2-27b-it, and Claude 3.5 Haiku) against an AIC-tuned ARIMA baseline on next-day directional prediction for five S&P 500 stocks (AAPL, TSLA, XOM, MRK, MS) over a six-month window (January–August 2023).

---

## Overview

A two-stage experimental design is used:

- **Stage 1** validates each LLM's competence as a financial sentiment classifier on the Financial PhraseBank (Malo et al., 2014).
- **Stage 2** measures directional next-day prediction accuracy on 954 anonymized headline records from the FNSPID dataset, with statistical significance assessed via McNemar's test and uncertainty quantified via bootstrap 95% confidence intervals. A simulated trading-strategy backtest reports Sharpe ratios for each model.
---

## Repository structure

```
.
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for API key configuration
│
├── data/                           # All input and derived datasets
│   ├── raw/                        # Original raw data (not committed; see below)
│   │   └── nasdaq_external_data.csv
│   ├── processed/                  # Filtered and prepared data
│   │   ├── fnspid_5stocks.parquet
│   │   ├── llm_input.jsonl
│   │   ├── fpb_dataset.csv
│   │   └── fpb_sample.csv
│   └── predictions/                # Model outputs
│       ├── arima_predictions.csv
│       ├── predictions_mistral.csv
│       ├── predictions_qwen.csv
│       ├── predictions_gemma2.csv
│       ├── predictions_claude.csv
│       ├── fpb_predictions_mistral.csv
│       ├── fpb_predictions_qwen.csv
│       ├── fpb_predictions_gemma2.csv
│       └── fpb_predictions_claude.csv
│
├── scripts/                        # All Python scripts
│   ├── 01_data_text.py             # FNSPID filtering for 5 target tickers
│   ├── 02_input_llm.py             # Anonymization, alignment, aggregation
│   ├── 03_verify_anon.py           # Anonymization leakage check
│   ├── 04_arima_regressor.py       # ARIMA baseline fitting and forecasting
│   ├── 05_build_fpb.py             # Parse FPB zip into CSV
│   ├── 06_fpb_sweep.py             # Stage 1 LLM sentiment classification
│   ├── 07_analyze_fpb.py           # Stage 1 metrics and confusion matrices
│   ├── 08_pilot_run.py             # Pilot run for prompt validation
│   ├── 09_full_llm_sweep.py        # Stage 2 directional prediction sweep
│   ├── 10_sanity_check.py          # Parse failures and class distributions
│   ├── 11_statistical_tests.py     # McNemar tests + bootstrap CIs
│   └── 12_sharpe_backtest.py       # Trading-strategy Sharpe ratio backtest
│
├── results/                        # Analysis outputs (CSV format)
│   ├── fpb_summary.csv
│   ├── mcnemar_results.csv
│   ├── bootstrap_results.csv
│   ├── sharpe_per_stock.csv
│   ├── sharpe_pooled.csv
│   └── sharpe_buy_and_hold.csv
│
├── thesis/                         # Thesis document
│   └── Thesis_de_Jongh.pdf
│
└── prompts/                        # Prompt templates used in Stages 1 and 2
    ├── stage1_fpb_prompt.txt
    └── stage2_directional_prompt.txt
```

---

## Data sources

| Source                    | Description                                                       | License                                |
|---------------------------|-------------------------------------------------------------------|----------------------------------------|
| FNSPID                    | Financial news headlines, Nasdaq subset (Dong et al., 2024)       | Available via HuggingFace `Zihan1004/FNSPID` |
| Financial PhraseBank      | 4,840 expert-annotated financial sentences (Malo et al., 2014)    | Academic research use                  |
| Yahoo Finance             | Historical stock prices via `yfinance` library                    | Yahoo Finance Terms of Service         |

> **Note:** The raw FNSPID file (`nasdaq_external_data.csv`, ~22 GB) is not committed to this repository due to size. Download it from [HuggingFace `Zihan1004/FNSPID`](https://huggingface.co/datasets/Zihan1004/FNSPID) and place it in `data/raw/` before running `scripts/01_data_text.py`.

---

## Reproducing the experiment

### 1. Environment setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. API key configuration

LLM access is via the [OpenRouter](https://openrouter.ai/) API. Create a `.env` file at the repository root:

```
OPENROUTER_API_KEY=your_key_here
```

A template is provided in `.env.example`.

### 3. Run the pipeline in order

```bash
# Data preparation
python scripts/01_data_text.py
python scripts/02_input_llm.py
python scripts/03_verify_anon.py

# ARIMA baseline
python scripts/04_arima_regressor.py

# Stage 1: FPB validation
python scripts/05_build_fpb.py
python scripts/06_fpb_sweep.py
python scripts/07_analyze_fpb.py

# Stage 2: directional prediction
python scripts/08_pilot_run.py
python scripts/09_full_llm_sweep.py
python scripts/10_sanity_check.py

# Statistical analysis
python scripts/11_statistical_tests.py
python scripts/12_sharpe_backtest.py
```

Each script is independently runnable and writes its outputs to the appropriate subdirectory. Earlier outputs are inputs to later scripts.

---

## Key results

- **Stage 1 (FPB):** All four LLMs exceed the FinBERT benchmark of approximately 0.86 accuracy. Headline metrics: Mistral 0.900, Gemma 2 0.886, Claude 0.884, Qwen 0.876.
- **Stage 2 (directional):** All four LLMs outperform ARIMA's balanced accuracy of 0.500 on point estimates. Claude (p = 0.009) and Gemma 2 (p = 0.023) reach statistical significance under McNemar's test on the 257-record five-way paired intersection.
- **Sharpe backtest:** All four LLMs produce positive Sharpe ratios with 95% CIs strictly above zero; the LLM advantage remains positive after excluding the strongest-trending stock (TSLA).

Full numerical results are in `results/` and discussed in `thesis/Thesis_de_Jongh.pdf`.

---

## Dependencies

The pipeline requires Python 3.10 or later. Core dependencies:

- `pandas`, `numpy`, `scipy`, `statsmodels` — data manipulation and statistical tests
- `pmdarima` — ARIMA model fitting
- `yfinance` — historical stock price retrieval
- `requests`, `python-dotenv` — OpenRouter API access
- `scikit-learn` — classification metrics
- `matplotlib`, `seaborn` — plots (optional)

See `requirements.txt` for exact versions.

---

## Citation

If you use this code or data, please cite the thesis:

```bibtex
@thesis{dejongh2026llmarima,
  author = {de Jongh, Yoran},
  title  = {Comparing LLM-based directional stock prediction against an ARIMA baseline},
  school = {Utrecht University},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```

---

## License

Code is released under the MIT License (see `LICENSE`). Underlying datasets retain their original licenses (see Data Sources section).

---

## Contact

For questions or feedback, contact: y.l.dejongh@students.uu.nl
