"""
cost_threshold.py
=================
Finds the optimal classification threshold by minimizing expected total cost,
using an asymmetric cost matrix that reflects real MFI economics:

  False Negative (missed true default) : 60% of loan amount (loss given default)
  False Positive (false alarm)         : fixed friction cost (₹500 intervention overhead)

Sweeps thresholds 0.05–0.95 and identifies the minimum-cost point.
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

# ── Cost parameters ────────────────────────────────────────────────────────
LGD_RATE        = 0.60      # Loss Given Default: 60% of loan_amount
FP_COST_FIXED   = 500.0     # Fixed friction cost per false alarm (currency units)
THRESHOLD_SWEEP = np.arange(0.05, 0.96, 0.01)


def compute_cost(y_true, y_proba, loan_amounts, threshold):
    """Compute total expected cost at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # FN cost: per-borrower loan amount × LGD
    fn_mask     = (y_true == 1) & (y_pred == 0)
    fn_cost     = loan_amounts[fn_mask].sum() * LGD_RATE

    # FP cost: fixed per false alarm
    fp_cost     = fp * FP_COST_FIXED

    total_cost  = fn_cost + fp_cost
    return total_cost, fn_cost, fp_cost, tn, fp, fn, tp


def run():
    print("=" * 60)
    print("  Cost-Sensitive Threshold Optimization")
    print("=" * 60)

    test_df      = pd.read_csv(PROC_DIR / "test_predictions.csv")
    y_true       = test_df["delinquent"].values
    y_proba      = test_df["xgb_proba"].values
    loan_amounts = test_df["loan_amount"].values

    results = []
    for thresh in THRESHOLD_SWEEP:
        total, fn_c, fp_c, tn, fp, fn, tp = compute_cost(
            y_true, y_proba, loan_amounts, thresh
        )
        recall    = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        results.append({
            "threshold"  : round(thresh, 2),
            "total_cost" : total,
            "fn_cost"    : fn_c,
            "fp_cost"    : fp_c,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall"     : recall,
            "precision"  : precision,
        })

    results_df = pd.DataFrame(results)
    opt_row    = results_df.loc[results_df["total_cost"].idxmin()]
    opt_thresh = float(opt_row["threshold"])

    print(f"\n  Default threshold (0.50):")
    default_row = results_df[results_df["threshold"] == 0.50].iloc[0]
    print(f"    Total cost   : {default_row['total_cost']:,.0f}")
    print(f"    Recall       : {default_row['recall']:.3f}")
    print(f"    Precision    : {default_row['precision']:.3f}")

    print(f"\n  Optimal threshold ({opt_thresh:.2f}):")
    print(f"    Total cost   : {opt_row['total_cost']:,.0f}")
    print(f"    Recall       : {opt_row['recall']:.3f}")
    print(f"    Precision    : {opt_row['precision']:.3f}")
    print(f"    TP={int(opt_row['tp'])}  FP={int(opt_row['fp'])}  "
          f"FN={int(opt_row['fn'])}  TN={int(opt_row['tn'])}")
    print(f"\n  Cost saving vs default: "
          f"{default_row['total_cost'] - opt_row['total_cost']:,.0f} units")

    # ── Save threshold ────────────────────────────────────────────────────
    opt_info = {
        "optimal_threshold"     : opt_thresh,
        "total_cost_at_optimal" : float(opt_row["total_cost"]),
        "total_cost_at_default" : float(default_row["total_cost"]),
        "cost_saving"           : float(default_row["total_cost"] - opt_row["total_cost"]),
        "recall_at_optimal"     : float(opt_row["recall"]),
        "precision_at_optimal"  : float(opt_row["precision"]),
        "lgd_rate"              : LGD_RATE,
        "fp_cost_fixed"         : FP_COST_FIXED,
    }
    with open(MODEL_DIR / "optimal_threshold.json", "w") as f:
        json.dump(opt_info, f, indent=2)

    results_df.to_csv(PROC_DIR / "cost_threshold_sweep.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")

    # Total cost curve
    axes[0].plot(results_df["threshold"], results_df["total_cost"],
                 color="#89b4fa", lw=2, label="Total Cost")
    axes[0].plot(results_df["threshold"], results_df["fn_cost"],
                 color="#f38ba8", lw=1.5, linestyle="--", label="FN Cost (missed defaults)")
    axes[0].plot(results_df["threshold"], results_df["fp_cost"],
                 color="#fab387", lw=1.5, linestyle="--", label="FP Cost (false alarms)")
    axes[0].axvline(x=opt_thresh, color="#a6e3a1", lw=2, linestyle=":",
                    label=f"Optimal threshold ({opt_thresh:.2f})")
    axes[0].axvline(x=0.50, color="#585b70", lw=1.5, linestyle=":",
                    label="Default threshold (0.50)")
    axes[0].set_xlabel("Classification Threshold")
    axes[0].set_ylabel("Expected Total Cost")
    axes[0].set_title("Cost vs Classification Threshold")
    axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=8)

    # Recall/Precision vs threshold
    axes[1].plot(results_df["threshold"], results_df["recall"],
                 color="#a6e3a1", lw=2, label="Recall (Delinquent)")
    axes[1].plot(results_df["threshold"], results_df["precision"],
                 color="#f9e2af", lw=2, label="Precision (Delinquent)")
    axes[1].axvline(x=opt_thresh, color="#89b4fa", lw=2, linestyle=":",
                    label=f"Optimal threshold ({opt_thresh:.2f})")
    axes[1].set_xlabel("Classification Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Recall / Precision vs Threshold")
    axes[1].legend(facecolor="#313244", labelcolor="#cdd6f4", fontsize=9)

    plt.tight_layout()
    out_path = MODEL_DIR / "cost_threshold_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    print(f"\n[save]  Optimal threshold  -> {MODEL_DIR}/optimal_threshold.json")
    print(f"[save]  Sweep results      -> {PROC_DIR}/cost_threshold_sweep.csv")
    print(f"[save]  Plot               -> {out_path}")

    return opt_thresh, results_df


if __name__ == "__main__":
    run()
