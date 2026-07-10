# Microfinance Loan Delinquency Early Warning System

> **Research prototype** — Predicts borrower delinquency *before* it happens using pre-loan income behavior.
> Simulated loan labels on real farmer income data. Not a production lending system.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (loan simulation → training → SHAP → fairness → stress test)
python run_pipeline.py

# 3. Launch the interactive dashboard
python -m streamlit run app/app.py
```

Then open **http://localhost:8501** in your browser.

---

## Results

| Metric | XGBoost | Logistic Regression |
|---|---|---|
| ROC-AUC | **0.727** | 0.700 |
| Recall (Delinquent) | 0.58 | 0.64 |
| Precision (Delinquent) | 0.45 | 0.40 |
| F1 (Delinquent) | 0.50 | 0.48 |
| Accuracy | 0.71 | 0.70 |

**Delinquency rate:** ~32% (1,264 qualifying borrowers, 20-week simulated loan term)

### Cost-Sensitive Threshold
| Setting | Threshold | Total Cost | Recall |
|---|---|---|---|
| Default | 0.50 | 1,345,804 | 0.578 |
| **Optimal (cost-minimized)** | **0.06** | **204,699** | **0.984** |

Cost saving of **1,141,104 units** by lowering threshold — driven by high loss-given-default (60% of loan).

### SHAP — Top Risk Drivers
| Reason | Borrowers |
|---|---|
| Declining income trend before loan | 87 |
| Stable income (positive signal) | 57 |
| Income trending upward (positive signal) | 49 |
| Low average income relative to installment | 27 |
| Highly irregular income pattern | 16 |

### Fairness (Q1 vs Q4 Income Quartile)
| Model | Demo. Parity Diff | Equalized Odds Gap | Recall Q1 / Q4 |
|---|---|---|---|
| Logistic Regression | 0.2633 | 0.1822 | 0.609 / 0.444 |
| **XGBoost** | **0.1066** | **0.1824** | **1.000 / 0.889** |

> XGBoost is **more fair** on demographic parity AND achieves higher recall for the most vulnerable borrowers (Q1).

### Stress Test — Tier Migration (Baseline → Severe -30% Income Shock)
| | Low | Medium | High |
|---|---|---|---|
| **Low (Baseline)** | 556 | 118 | 17 |
| **Medium (Baseline)** | 86 | 156 | 54 |
| **High (Baseline)** | 18 | 74 | 185 |

---

## Architecture

```
data/raw/farmers_salary_transactions.csv
        |
        v
src/loan_simulation.py   --> labeled_borrowers.csv, trajectories.csv
        |
        v
src/train_model.py       --> lr_baseline.pkl, xgb_model.json, test_predictions.csv
        |
    +---+---+---+---+
    |   |   |   |
    v   v   v   v
shap_explain.py   --> shap_explanations.csv
interventions.py  --> borrower_interventions.csv
cost_threshold.py --> optimal_threshold.json
fairness_audit.py --> fairness_*_metrics.csv
stress_test.py    --> stress_test_results.csv
        |
        v
app/app.py  (Streamlit — 6 pages)
```

## Features

| # | Feature | Status |
|---|---|---|
| 1 | Risk trajectory (5/10/15/20 week checkpoints) | Done |
| 2 | SHAP reason codes + intervention mapping | Done |
| 3 | Cost-sensitive threshold optimization | Done |
| 4 | Fairness audit (LR vs XGBoost, Q1/Q4 subgroups) | Done |
| 5 | Stress test simulator (-10/-20/-30% income shock) | Done |
| 6 | What-if simulator (live risk score from sliders) | Done |

## App Pages
1. **Portfolio Overview** — KPIs, risk tier donut, score distribution, model performance table
2. **Borrower Explorer** — Filterable/sortable table of all scored borrowers
3. **Borrower Detail** — SHAP waterfall, risk trajectory, intervention card
4. **What-If Simulator** — Sliders → live gauge → instant risk tier
5. **Stress Test Dashboard** — Stacked bars, delinquency line chart, migration matrix
6. **Fairness Report** — Side-by-side fairness metrics, honest tradeoff documentation

## Loan Simulation Design

| Parameter | Value |
|---|---|
| Lookback window | 8 weeks (features computed here ONLY) |
| Loan term | 20 weeks |
| Installment | 35% of avg lookback income |
| Missed payment | Income < 70% of installment |
| Delinquent label | >= 4 missed payments |

> **Data leakage guard:** All features derived exclusively from weeks 1–8.
> The 20-week loan period (weeks 9–28) is used only for label generation.

## Known Limitations

- Labels are **simulated**, not observed real-world defaults
- 8-week lookback window is short; real underwriting uses longer histories
- Fairness analysis limited to income quartile (no demographic attributes in dataset)
- 7-feature model; real MFI systems use credit bureau, group lending, guarantor data
- Not validated against actual MFI repayment outcomes — research prototype only

## Project Structure

```
Microfinance Loan Delinquency/
├── data/
│   ├── raw/                     # Input CSV
│   └── processed/               # All generated CSVs
├── src/
│   ├── loan_simulation.py
│   ├── train_model.py
│   ├── shap_explain.py
│   ├── interventions.py
│   ├── cost_threshold.py
│   ├── fairness_audit.py
│   └── stress_test.py
├── models/                      # Saved model artifacts + plots
├── app/app.py                   # Streamlit dashboard
├── run_pipeline.py              # Single entry point
└── requirements.txt
```
