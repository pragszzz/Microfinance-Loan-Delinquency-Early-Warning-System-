"""
shap_explain.py
===============
Computes SHAP values for all test-set borrowers using the trained XGBoost
model. Maps top SHAP features to human-readable reason codes and saves
per-borrower explanations.
"""

import json
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    "avg_lookback_income",
    "income_volatility_cv",
    "income_trend_slope",
    "active_weeks_pre_loan",
    "zero_income_weeks_pre_loan",
    "loan_amount",
    "installment",
]

# ── Reason code mapping ────────────────────────────────────────────────────
# Each feature maps to a human-readable explanation based on SHAP direction
REASON_MAP = {
    "income_volatility_cv": {
        "high": "Highly irregular income pattern",
        "low":  "Stable income pattern (positive signal)",
    },
    "zero_income_weeks_pre_loan": {
        "high": "Multiple income gaps before loan",
        "low":  "Consistent income activity (positive signal)",
    },
    "avg_lookback_income": {
        "high": "Strong average income relative to installment",
        "low":  "Low average income relative to installment",
    },
    "income_trend_slope": {
        "high": "Income trending upward (positive signal)",
        "low":  "Declining income trend before loan",
    },
    "active_weeks_pre_loan": {
        "high": "High income activity in pre-loan period (positive signal)",
        "low":  "Low income activity in pre-loan period",
    },
    "loan_amount": {
        "high": "Large loan relative to income capacity",
        "low":  "Modest loan size (positive signal)",
    },
    "installment": {
        "high": "High weekly installment burden",
        "low":  "Affordable weekly installment (positive signal)",
    },
}


def shap_direction(feature_name: str, shap_value: float, feature_value: float) -> str:
    """Return 'high' or 'low' label based on SHAP value sign."""
    return "high" if shap_value > 0 else "low"


def build_reason_code(top_features: list[tuple]) -> str:
    """Given ranked (feature, shap_val, feat_val) tuples, return top reason string."""
    top_feat, top_shap, top_val = top_features[0]
    direction = shap_direction(top_feat, top_shap, top_val)
    return REASON_MAP.get(top_feat, {}).get(direction, f"{top_feat} impact")


def build_secondary_reason(top_features: list[tuple]) -> str:
    if len(top_features) < 2:
        return "N/A"
    feat, shap_val, val = top_features[1]
    direction = shap_direction(feat, shap_val, val)
    return REASON_MAP.get(feat, {}).get(direction, f"{feat} impact")


def run():
    print("=" * 60)
    print("  SHAP Explanation Pipeline")
    print("=" * 60)

    # Load model and test predictions
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))

    test_df = pd.read_csv(PROC_DIR / "test_predictions.csv")
    X_test  = test_df[FEATURE_COLS]

    print(f"[shap]  Computing SHAP values for {len(X_test)} test borrowers...")

    # TreeExplainer is fast and exact for XGBoost
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)   # shape: (n_borrowers, n_features)

    # Per-borrower: rank features by |SHAP| descending
    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS, index=X_test.index)

    records = []
    for i, idx in enumerate(X_test.index):
        row_shap = shap_values[i]
        row_feat = X_test.iloc[i].values
        ranked   = sorted(
            zip(FEATURE_COLS, row_shap, row_feat),
            key=lambda x: abs(x[1]), reverse=True
        )
        primary   = build_reason_code(ranked)
        secondary = build_secondary_reason(ranked)
        top_feat  = ranked[0][0]
        records.append({
            "farmer_id"        : test_df.iloc[i]["farmer_id"],
            "primary_reason"   : primary,
            "secondary_reason" : secondary,
            "top_shap_feature" : top_feat,
            "top_shap_value"   : round(ranked[0][1], 5),
            "second_shap_feat" : ranked[1][0] if len(ranked) > 1 else "",
            "second_shap_value": round(ranked[1][1], 5) if len(ranked) > 1 else 0.0,
        })

    explanation_df = pd.DataFrame(records)
    explanation_df.to_csv(PROC_DIR / "shap_explanations.csv", index=False)

    # Save raw SHAP values
    shap_df.to_csv(PROC_DIR / "shap_values_raw.csv", index=False)

    # ── Global SHAP summary plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = sorted(zip(FEATURE_COLS, mean_abs_shap), key=lambda x: x[1], reverse=True)
    feats, vals = zip(*feat_importance)

    colors = ["#f38ba8" if v > 0.05 else "#89b4fa" for v in vals]
    bars = ax.barh(list(feats)[::-1], list(vals)[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Mean |SHAP Value|", color="#cdd6f4")
    ax.set_title("Feature Importance (SHAP)", color="#cdd6f4", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")

    plt.tight_layout()
    out_path = MODEL_DIR / "shap_feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    print(f"[shap]  Explanations -> {PROC_DIR}/shap_explanations.csv")
    print(f"[shap]  Raw SHAP     -> {PROC_DIR}/shap_values_raw.csv")
    print(f"[shap]  Plot         -> {out_path}")
    print(f"\nTop reason distribution:\n{explanation_df['primary_reason'].value_counts().to_string()}")

    return explanation_df, shap_values


if __name__ == "__main__":
    run()
