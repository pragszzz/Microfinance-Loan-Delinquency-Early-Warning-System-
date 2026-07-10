"""
fairness_audit.py
=================
Compares Logistic Regression vs XGBoost across income-quartile subgroups.

Fairness metrics computed:
  - Demographic Parity Difference  (positive rate gap across groups)
  - Equalized Odds                 (TPR gap + FPR gap)
  - Accuracy per subgroup
  - Recall (Delinquent) per subgroup

Subgroup definition: income quartile (Q1 = lowest 25% avg income, Q4 = highest 25%).
Rationale: In MFI lending, income level is the most meaningful fairness axis derivable
from this dataset. We compare Q1 (most vulnerable) vs Q4 (highest capacity).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

OPTIMAL_THRESHOLD = 0.5   # Will be overridden if optimal_threshold.json exists


def load_threshold():
    path = MODEL_DIR / "optimal_threshold.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["optimal_threshold"]
    return OPTIMAL_THRESHOLD


def group_metrics(y_true, y_proba, threshold, model_name, group_name):
    """Compute confusion matrix metrics for a subgroup."""
    y_pred = (y_proba >= threshold).astype(int)
    n = len(y_true)
    if n == 0:
        return {}

    pos_rate = y_pred.mean()
    acc      = (y_true == y_pred).mean()

    # Handle edge cases where group has no positives/negatives
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "model"         : model_name,
        "group"         : group_name,
        "n"             : n,
        "positive_rate" : round(pos_rate, 4),
        "accuracy"      : round(acc, 4),
        "recall_deliq"  : round(tpr, 4),
        "fpr"           : round(fpr, 4),
        "precision_del" : round(prec, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_fairness_metrics(df_q1, df_q4, model_col, threshold, model_name):
    """Compute fairness gaps between Q1 and Q4 groups."""
    m_q1 = group_metrics(df_q1["delinquent"].values, df_q1[model_col].values,
                          threshold, model_name, "Q1 (Lowest Income)")
    m_q4 = group_metrics(df_q4["delinquent"].values, df_q4[model_col].values,
                          threshold, model_name, "Q4 (Highest Income)")

    dp_diff       = abs(m_q1["positive_rate"] - m_q4["positive_rate"])
    tpr_gap       = abs(m_q1["recall_deliq"]  - m_q4["recall_deliq"])
    fpr_gap       = abs(m_q1["fpr"]           - m_q4["fpr"])
    eq_odds       = (tpr_gap + fpr_gap) / 2

    return m_q1, m_q4, {
        "model"                      : model_name,
        "demographic_parity_diff"    : round(dp_diff, 4),
        "tpr_gap"                    : round(tpr_gap, 4),
        "fpr_gap"                    : round(fpr_gap, 4),
        "equalized_odds_gap"         : round(eq_odds, 4),
        "acc_q1"                     : m_q1["accuracy"],
        "acc_q4"                     : m_q4["accuracy"],
        "recall_q1"                  : m_q1["recall_deliq"],
        "recall_q4"                  : m_q4["recall_deliq"],
    }


def run():
    print("=" * 60)
    print("  Fairness Audit")
    print("=" * 60)

    test_df   = pd.read_csv(PROC_DIR / "test_predictions.csv")
    threshold = load_threshold()

    print(f"[fairness] Using threshold: {threshold}")
    print(f"[fairness] Test set size  : {len(test_df)}")

    # ── Derive income quartile subgroup ───────────────────────────────────
    test_df["income_quartile"] = pd.qcut(
        test_df["avg_lookback_income"], q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    df_q1 = test_df[test_df["income_quartile"] == "Q1"].copy()
    df_q4 = test_df[test_df["income_quartile"] == "Q4"].copy()

    print(f"[fairness] Q1 (lowest income): {len(df_q1)} borrowers, "
          f"delinquency rate {df_q1['delinquent'].mean():.1%}")
    print(f"[fairness] Q4 (highest income): {len(df_q4)} borrowers, "
          f"delinquency rate {df_q4['delinquent'].mean():.1%}")

    # ── Compute metrics for LR and XGBoost ───────────────────────────────
    lr_q1,  lr_q4,  lr_gaps  = compute_fairness_metrics(df_q1, df_q4, "lr_proba",  0.50,      "Logistic Regression")
    xgb_q1, xgb_q4, xgb_gaps = compute_fairness_metrics(df_q1, df_q4, "xgb_proba", threshold, "XGBoost")

    print("\n-- Logistic Regression Fairness --")
    print(f"  Demographic Parity Diff : {lr_gaps['demographic_parity_diff']:.4f}")
    print(f"  Equalized Odds Gap      : {lr_gaps['equalized_odds_gap']:.4f}")
    print(f"  Recall Q1 / Q4          : {lr_gaps['recall_q1']:.3f} / {lr_gaps['recall_q4']:.3f}")

    print("\n-- XGBoost Fairness --")
    print(f"  Demographic Parity Diff : {xgb_gaps['demographic_parity_diff']:.4f}")
    print(f"  Equalized Odds Gap      : {xgb_gaps['equalized_odds_gap']:.4f}")
    print(f"  Recall Q1 / Q4          : {xgb_gaps['recall_q1']:.3f} / {xgb_gaps['recall_q4']:.3f}")

    # ── Save results ──────────────────────────────────────────────────────
    group_rows = [lr_q1, lr_q4, xgb_q1, xgb_q4]
    group_df   = pd.DataFrame(group_rows)
    group_df.to_csv(PROC_DIR / "fairness_subgroup_metrics.csv", index=False)

    gap_rows = [lr_gaps, xgb_gaps]
    gaps_df  = pd.DataFrame(gap_rows)
    gaps_df.to_csv(PROC_DIR / "fairness_gap_metrics.csv", index=False)

    # Save quartile info back for app
    test_df.to_csv(PROC_DIR / "test_predictions.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Fairness Audit: LR vs XGBoost — Income Quartile Subgroups",
                 color="#cdd6f4", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")

    groups = ["Q1 (Lowest)", "Q4 (Highest)"]
    x      = np.arange(len(groups))
    width  = 0.35

    # Recall
    lr_rec  = [lr_q1["recall_deliq"],  lr_q4["recall_deliq"]]
    xgb_rec = [xgb_q1["recall_deliq"], xgb_q4["recall_deliq"]]
    axes[0].bar(x - width/2, lr_rec,  width, label="LR",     color="#89b4fa", alpha=0.85)
    axes[0].bar(x + width/2, xgb_rec, width, label="XGBoost", color="#a6e3a1", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(groups)
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Recall (Delinquent) by Group")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4")
    for i, (lr_v, xgb_v) in enumerate(zip(lr_rec, xgb_rec)):
        axes[0].text(i - width/2, lr_v + 0.02,  f"{lr_v:.2f}",  ha="center", color="#89b4fa", fontsize=9)
        axes[0].text(i + width/2, xgb_v + 0.02, f"{xgb_v:.2f}", ha="center", color="#a6e3a1", fontsize=9)

    # Positive Rate (Demographic Parity)
    lr_pr  = [lr_q1["positive_rate"],  lr_q4["positive_rate"]]
    xgb_pr = [xgb_q1["positive_rate"], xgb_q4["positive_rate"]]
    axes[1].bar(x - width/2, lr_pr,  width, label="LR",     color="#89b4fa", alpha=0.85)
    axes[1].bar(x + width/2, xgb_pr, width, label="XGBoost", color="#a6e3a1", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(groups)
    axes[1].set_ylim(0, 1.05); axes[1].set_title("Positive Rate (Demographic Parity)")
    axes[1].legend(facecolor="#313244", labelcolor="#cdd6f4")

    # Fairness gap summary
    metrics  = ["Demo. Parity Diff", "Equalized Odds Gap", "TPR Gap", "FPR Gap"]
    lr_vals  = [lr_gaps["demographic_parity_diff"],  lr_gaps["equalized_odds_gap"],
                lr_gaps["tpr_gap"],                  lr_gaps["fpr_gap"]]
    xgb_vals = [xgb_gaps["demographic_parity_diff"], xgb_gaps["equalized_odds_gap"],
                xgb_gaps["tpr_gap"],                 xgb_gaps["fpr_gap"]]
    x2 = np.arange(len(metrics))
    axes[2].bar(x2 - width/2, lr_vals,  width, label="LR",     color="#89b4fa", alpha=0.85)
    axes[2].bar(x2 + width/2, xgb_vals, width, label="XGBoost", color="#a6e3a1", alpha=0.85)
    axes[2].set_xticks(x2); axes[2].set_xticklabels(metrics, rotation=15, ha="right", fontsize=8)
    axes[2].set_title("Fairness Gap Metrics\n(lower = more fair)")
    axes[2].legend(facecolor="#313244", labelcolor="#cdd6f4")

    plt.tight_layout()
    out_path = MODEL_DIR / "fairness_audit.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    print(f"\n[save]  Subgroup metrics -> {PROC_DIR}/fairness_subgroup_metrics.csv")
    print(f"[save]  Gap metrics      -> {PROC_DIR}/fairness_gap_metrics.csv")
    print(f"[save]  Plot             -> {out_path}")

    return group_df, gaps_df


if __name__ == "__main__":
    run()
