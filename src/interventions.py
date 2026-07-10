"""
interventions.py
================
Maps SHAP-derived reason codes to specific, actionable interventions.
Merges intervention recommendations into the test predictions dataset.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

# ── Intervention mapping ───────────────────────────────────────────────────
INTERVENTION_MAP = {
    # Reason code string -> intervention recommendation
    "Highly irregular income pattern": {
        "action"      : "Offer flexible repayment schedule",
        "detail"      : "Restructure installments to align with income arrival dates "
                        "(e.g., allow bi-weekly or harvest-cycle payments instead of weekly).",
        "urgency"     : "High",
        "icon"        : "🔄",
    },
    "Multiple income gaps before loan": {
        "action"      : "Add grace period buffer + emergency fund",
        "detail"      : "Build a 2-week grace buffer into the loan terms and encourage "
                        "participation in a group savings scheme to cover gap weeks.",
        "urgency"     : "High",
        "icon"        : "🛡️",
    },
    "Declining income trend before loan": {
        "action"      : "Loan restructure or term extension review",
        "detail"      : "Investigate root cause of declining income (seasonal, market). "
                        "Consider extending the loan term to reduce weekly burden.",
        "urgency"     : "High",
        "icon"        : "📉",
    },
    "Low average income relative to installment": {
        "action"      : "Loan right-sizing review",
        "detail"      : "Recalculate installment targeting ≤30% of income instead of 35%. "
                        "Consider a smaller initial loan with step-up terms.",
        "urgency"     : "Medium",
        "icon"        : "⚖️",
    },
    "Low income activity in pre-loan period": {
        "action"      : "Pair with income-generating support program",
        "detail"      : "Connect borrower with livelihood program or group lending circle "
                        "to increase income consistency before loan disbursement.",
        "urgency"     : "Medium",
        "icon"        : "🌱",
    },
    "Large loan relative to income capacity": {
        "action"      : "Reduce loan principal",
        "detail"      : "Cap loan at 25× average weekly income. Offer a smaller tranche "
                        "with performance-linked top-up after 10 weeks.",
        "urgency"     : "Medium",
        "icon"        : "💰",
    },
    "High weekly installment burden": {
        "action"      : "Reduce installment via term extension",
        "detail"      : "Extend term from 20 to 26 weeks to reduce weekly installment "
                        "to ≤28% of average income.",
        "urgency"     : "Medium",
        "icon"        : "📅",
    },
    # Positive signals -> monitoring only
    "Stable income pattern (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "Borrower shows stable income — proceed with standard terms. "
                        "Monthly check-in recommended.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "Strong average income relative to installment": {
        "action"      : "Standard monitoring",
        "detail"      : "High income capacity — low risk. Standard monthly monitoring.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "Income trending upward (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "Positive income trend — good repayment prospects. Standard terms.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "Consistent income activity (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "Consistent weekly activity — low gap risk. Standard terms.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "High income activity in pre-loan period (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "High engagement — good risk profile. Standard terms apply.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "Affordable weekly installment (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "Installment comfortably within income capacity. Low risk.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
    "Modest loan size (positive signal)": {
        "action"      : "Standard monitoring",
        "detail"      : "Conservative loan sizing — good repayment outlook.",
        "urgency"     : "Low",
        "icon"        : "✅",
    },
}

DEFAULT_INTERVENTION = {
    "action" : "Manual underwriter review",
    "detail" : "Could not determine specific risk driver — escalate to loan officer for manual review.",
    "urgency": "Medium",
    "icon"   : "🔍",
}


def run():
    print("=" * 60)
    print("  Intervention Mapping Pipeline")
    print("=" * 60)

    expl_df  = pd.read_csv(PROC_DIR / "shap_explanations.csv")
    test_df  = pd.read_csv(PROC_DIR / "test_predictions.csv")

    # Merge explanations onto test predictions
    merged = test_df.merge(expl_df, on="farmer_id", how="left")

    # Map reason code -> intervention
    def get_field(reason: str, field: str) -> str:
        entry = INTERVENTION_MAP.get(reason, DEFAULT_INTERVENTION)
        return entry[field]

    merged["intervention_action"]  = merged["primary_reason"].apply(lambda r: get_field(r, "action"))
    merged["intervention_detail"]  = merged["primary_reason"].apply(lambda r: get_field(r, "detail"))
    merged["intervention_urgency"] = merged["primary_reason"].apply(lambda r: get_field(r, "urgency"))
    merged["intervention_icon"]    = merged["primary_reason"].apply(lambda r: get_field(r, "icon"))

    out_path = PROC_DIR / "borrower_interventions.csv"
    merged.to_csv(out_path, index=False)

    print(f"[interventions] Saved -> {out_path}")
    print(f"\nIntervention urgency distribution:")
    print(merged["intervention_urgency"].value_counts().to_string())
    print(f"\nTop interventions:")
    print(merged["intervention_action"].value_counts().head(5).to_string())

    return merged


if __name__ == "__main__":
    run()
