"""
stress_test.py
==============
Simulates income shocks across all borrowers by re-running the loan
simulation with scaled incomes, then re-scores with the trained XGBoost.

Scenarios:
  A: -10% income shock (mild downturn)
  B: -20% income shock (moderate crisis — e.g., drought, market shock)
  C: -30% income shock (severe crisis)

Output:
  - Risk tier migration matrix per scenario
  - Portfolio-level risk shift summary
  - Saves stress_test_results.csv with per-borrower risk tiers under each scenario
"""

import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from pathlib import Path
from scipy import stats

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RAW_DATA  = BASE_DIR / "data" / "raw" / "farmers_salary_transactions.csv"

FEATURE_COLS = [
    "avg_lookback_income",
    "income_volatility_cv",
    "income_trend_slope",
    "active_weeks_pre_loan",
    "zero_income_weeks_pre_loan",
    "loan_amount",
    "installment",
]

LOOKBACK_WEEKS  = 8
LOAN_TERM_WEEKS = 20
AFFORD_RATIO    = 0.35
MIN_ACTIVE_WEEKS = 10

SCENARIOS = {
    "Baseline"          : 1.00,
    "Mild (-10%)"       : 0.90,
    "Moderate (-20%)"   : 0.80,
    "Severe (-30%)"     : 0.70,
}

TIERS = ["Low", "Medium", "High"]


def parse_currency(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s in ("-", " - ", ""):
        return 0.0
    s = re.sub(r"[,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_and_filter():
    df = pd.read_csv(RAW_DATA)
    df.rename(columns={df.columns[0]: "farmer_id"}, inplace=True)
    week_cols = [c for c in df.columns if c.lower().startswith("week")]
    for col in week_cols:
        df[col] = df[col].apply(parse_currency)
    active = (df[week_cols] > 0).sum(axis=1)
    df = df[active >= MIN_ACTIVE_WEEKS].copy().reset_index(drop=True)
    return df, week_cols


def compute_features_scaled(df, week_cols, scale_factor=1.0):
    """Compute features on scaled income data."""
    lb_cols = week_cols[:LOOKBACK_WEEKS]
    lb = df[lb_cols].values.astype(float) * scale_factor

    avg_income = lb.mean(axis=1)
    std_income = lb.std(axis=1, ddof=1)
    cv         = np.where(avg_income > 0, std_income / avg_income, 0.0)

    x = np.arange(LOOKBACK_WEEKS)
    slopes = np.array([
        stats.linregress(x, lb[i])[0] for i in range(len(lb))
    ])

    active_lb = (lb > 0).sum(axis=1)
    zero_lb   = (lb == 0).sum(axis=1)

    installment = AFFORD_RATIO * avg_income
    loan_amount = installment * LOAN_TERM_WEEKS

    feat = pd.DataFrame({
        "farmer_id"                  : df["farmer_id"].values,
        "avg_lookback_income"        : avg_income,
        "income_volatility_cv"       : cv,
        "income_trend_slope"         : slopes,
        "active_weeks_pre_loan"      : active_lb,
        "zero_income_weeks_pre_loan" : zero_lb,
        "loan_amount"                : loan_amount,
        "installment"                : installment,
    })
    return feat


def risk_tier(p):
    if p < 0.33:  return "Low"
    if p < 0.66:  return "Medium"
    return "High"


def run():
    print("=" * 60)
    print("  Stress Test Simulator")
    print("=" * 60)

    # Load model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))

    df, week_cols = load_and_filter()

    # Get optimal threshold
    thresh_path = MODEL_DIR / "optimal_threshold.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            threshold = json.load(f)["optimal_threshold"]
    else:
        threshold = 0.50
    print(f"[stress] Using threshold: {threshold}")

    # ── Score each scenario ───────────────────────────────────────────────
    all_results = {}
    summary_records = []

    for scenario_name, scale in SCENARIOS.items():
        feat = compute_features_scaled(df, week_cols, scale_factor=scale)
        X    = feat[FEATURE_COLS]
        proba = xgb_model.predict_proba(X)[:, 1]
        tiers = [risk_tier(p) for p in proba]

        all_results[scenario_name] = {
            "farmer_id" : feat["farmer_id"].values,
            "proba"     : proba,
            "tier"      : tiers,
        }

        tier_counts = pd.Series(tiers).value_counts()
        total = len(tiers)
        summary_records.append({
            "scenario"          : scenario_name,
            "scale_factor"      : scale,
            "pct_low"           : tier_counts.get("Low", 0) / total,
            "pct_medium"        : tier_counts.get("Medium", 0) / total,
            "pct_high"          : tier_counts.get("High", 0) / total,
            "pct_delinquent_pred": (np.array(proba) >= threshold).mean(),
            "n_low"             : tier_counts.get("Low", 0),
            "n_medium"          : tier_counts.get("Medium", 0),
            "n_high"            : tier_counts.get("High", 0),
        })
        print(f"[stress] {scenario_name:20s} -> Low:{tier_counts.get('Low',0):4d}  "
              f"Med:{tier_counts.get('Medium',0):4d}  High:{tier_counts.get('High',0):4d}")

    summary_df = pd.DataFrame(summary_records)

    # ── Build per-borrower results table ──────────────────────────────────
    results_df = pd.DataFrame({"farmer_id": all_results["Baseline"]["farmer_id"]})
    for scenario_name in SCENARIOS:
        r = all_results[scenario_name]
        slug = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").lower()
        results_df[f"proba_{slug}"]   = r["proba"]
        results_df[f"tier_{slug}"]    = r["tier"]

    results_df.to_csv(PROC_DIR / "stress_test_results.csv", index=False)
    summary_df.to_csv(PROC_DIR / "stress_test_summary.csv", index=False)

    # ── Migration matrix: Baseline -> Severe ──────────────────────────────
    base_tiers   = pd.Categorical(all_results["Baseline"]["tier"], categories=TIERS)
    severe_tiers = pd.Categorical(all_results["Severe (-30%)"]["tier"], categories=TIERS)
    migration_matrix = pd.crosstab(
        pd.Series(base_tiers, name="Baseline"),
        pd.Series(severe_tiers, name="Severe (-30%)"),
    )

    print("\n[stress] Tier Migration Matrix (Baseline -> Severe -30%):")
    print(migration_matrix.to_string())

    migration_matrix.to_csv(PROC_DIR / "stress_migration_matrix.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Stress Test: Income Shock Impact on Risk Distribution",
                 color="#cdd6f4", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")

    # Stacked bar chart: tier distribution per scenario
    sc_labels = list(SCENARIOS.keys())
    lows    = summary_df["pct_low"].values * 100
    meds    = summary_df["pct_medium"].values * 100
    highs   = summary_df["pct_high"].values * 100
    x = np.arange(len(sc_labels))

    axes[0].bar(x, lows,  label="Low Risk",    color="#a6e3a1", alpha=0.9)
    axes[0].bar(x, meds,  bottom=lows,          label="Medium Risk", color="#f9e2af", alpha=0.9)
    axes[0].bar(x, highs, bottom=lows + meds,   label="High Risk",   color="#f38ba8", alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(sc_labels, rotation=10)
    axes[0].set_ylabel("% of Borrowers"); axes[0].set_ylim(0, 100)
    axes[0].set_title("Risk Tier Distribution by Scenario")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4")

    # Line chart: % predicted delinquent per scenario
    axes[1].plot(sc_labels, summary_df["pct_delinquent_pred"].values * 100,
                 marker="o", color="#f38ba8", lw=2.5, markersize=8, markerfacecolor="#1a1d2e")
    for i, v in enumerate(summary_df["pct_delinquent_pred"].values * 100):
        axes[1].annotate(f"{v:.1f}%", (sc_labels[i], v), textcoords="offset points",
                         xytext=(0, 10), ha="center", color="#f38ba8", fontsize=10)
    axes[1].set_ylabel("% Predicted Delinquent")
    axes[1].set_title("Predicted Delinquency Rate Under Shock")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    out_path = MODEL_DIR / "stress_test_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    print(f"\n[save]  Per-borrower results  -> {PROC_DIR}/stress_test_results.csv")
    print(f"[save]  Summary              -> {PROC_DIR}/stress_test_summary.csv")
    print(f"[save]  Migration matrix     -> {PROC_DIR}/stress_migration_matrix.csv")
    print(f"[save]  Plot                 -> {out_path}")

    return results_df, summary_df, migration_matrix


if __name__ == "__main__":
    run()
