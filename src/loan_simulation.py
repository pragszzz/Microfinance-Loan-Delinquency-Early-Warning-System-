"""
loan_simulation.py
==================
Loads raw farmer income data, simulates a micro-loan cycle, computes
pre-loan features, generates delinquency labels, and saves trajectory
checkpoints.

Design:
  - Lookback window : weeks 1–8  (feature computation only)
  - Loan term       : weeks 9–28 (20 weeks)
  - Installment     : 35% of avg lookback income
  - Missed payment  : week income < 70% of installment
  - Delinquent      : ≥ 4 missed payments (≥ 20% of 20-week term)
  - Trajectory CPs  : cumulative missed payments at weeks 5, 10, 15, 20
                      of the loan term (loan weeks 9–13, 9–18, 9–23, 9–28)

Data leakage guard: ALL predictive features derived ONLY from weeks 1–8.
"""

import re
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA  = BASE_DIR / "data" / "raw" / "farmers_salary_transactions.csv"
PROC_DIR  = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
LOOKBACK_WEEKS   = 8          # weeks used for feature engineering
LOAN_TERM_WEEKS  = 20         # loan repayment period
AFFORD_RATIO     = 0.35       # installment = 35% × avg lookback income
COVER_RATIO      = 0.70       # income must cover ≥70% of installment
DELINQ_THRESHOLD = 4          # ≥ 4 missed payments -> delinquent
MIN_ACTIVE_WEEKS = 10         # minimum active (non-zero) weeks to qualify

TRAJ_CHECKPOINTS = [5, 10, 15, 20]   # cumulative missed payments checkpoints


def parse_currency(val) -> float:
    """Convert ' - ' dashes and comma-formatted numbers to float."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s in ("-", " - ", ""):
        return 0.0
    # Remove commas and any currency symbols
    s = re.sub(r"[,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load CSV and convert all Week columns to numeric."""
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "farmer_id"}, inplace=True)
    week_cols = [c for c in df.columns if c.lower().startswith("week")]
    for col in week_cols:
        df[col] = df[col].apply(parse_currency)
    print(f"[load]  Raw shape: {df.shape} ({len(week_cols)} week columns)")
    return df, week_cols


def filter_borrowers(df: pd.DataFrame, week_cols: list) -> pd.DataFrame:
    """Keep only farmers with ≥ MIN_ACTIVE_WEEKS non-zero income weeks."""
    active = (df[week_cols] > 0).sum(axis=1)
    mask = active >= MIN_ACTIVE_WEEKS
    filtered = df[mask].copy().reset_index(drop=True)
    print(f"[filter] {len(df)} -> {len(filtered)} farmers "
          f"(dropped {len(df)-len(filtered)} with <{MIN_ACTIVE_WEEKS} active weeks)")
    return filtered


def compute_features(df: pd.DataFrame, week_cols: list) -> pd.DataFrame:
    """Compute pre-loan features from the lookback window only (weeks 1–8)."""
    lb_cols = week_cols[:LOOKBACK_WEEKS]   # Week1 … Week8

    lb = df[lb_cols].values.astype(float)

    avg_income   = lb.mean(axis=1)
    std_income   = lb.std(axis=1, ddof=1)
    cv           = np.where(avg_income > 0, std_income / avg_income, 0.0)

    # Linear trend slope over lookback
    x = np.arange(LOOKBACK_WEEKS)
    slopes = np.array([
        stats.linregress(x, lb[i])[0] for i in range(len(lb))
    ])

    active_lb    = (lb > 0).sum(axis=1)
    zero_lb      = (lb == 0).sum(axis=1)

    installment  = AFFORD_RATIO * avg_income
    loan_amount  = installment * LOAN_TERM_WEEKS

    feat = pd.DataFrame({
        "farmer_id"              : df["farmer_id"].values,
        "avg_lookback_income"    : avg_income,
        "income_volatility_cv"   : cv,
        "income_trend_slope"     : slopes,
        "active_weeks_pre_loan"  : active_lb,
        "zero_income_weeks_pre_loan": zero_lb,
        "loan_amount"            : loan_amount,
        "installment"            : installment,
    })
    return feat


def simulate_loan(df: pd.DataFrame, week_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the 20-week loan simulation on weeks 9–28 of income data.
    Returns:
        labeled_df   : feature df + delinquent label + raw missed-payment count
        traj_df      : trajectory DataFrame (farmer_id + checkpoint columns)
    """
    loan_cols = week_cols[LOOKBACK_WEEKS: LOOKBACK_WEEKS + LOAN_TERM_WEEKS]  # Week9 … Week28

    feat = compute_features(df, week_cols)
    installments = feat["installment"].values

    # Loan income matrix
    loan_income = df[loan_cols].values.astype(float)   # (n_farmers, 20)

    # Missed payment matrix: 1 if income < COVER_RATIO * installment
    threshold   = (COVER_RATIO * installments)[:, np.newaxis]
    missed      = (loan_income < threshold).astype(int)  # (n_farmers, 20)

    # Delinquency label
    total_missed = missed.sum(axis=1)
    delinquent   = (total_missed >= DELINQ_THRESHOLD).astype(int)

    feat["total_missed_payments"] = total_missed
    feat["delinquent"]            = delinquent

    print(f"[simulate] Delinquency rate: {delinquent.mean():.1%}  "
          f"({delinquent.sum()} / {len(delinquent)} borrowers)")

    # Trajectory checkpoints — cumulative missed payments at weeks 5,10,15,20
    traj_records = {"farmer_id": feat["farmer_id"].values}
    for cp in TRAJ_CHECKPOINTS:
        traj_records[f"missed_by_week_{cp}"] = missed[:, :cp].sum(axis=1)

    traj_df = pd.DataFrame(traj_records)

    return feat, traj_df


def run(path: Path = RAW_DATA):
    print("=" * 60)
    print("  Loan Simulation Pipeline")
    print("=" * 60)

    df, week_cols = load_raw(path)
    df            = filter_borrowers(df, week_cols)
    labeled, traj = simulate_loan(df, week_cols)

    out_labeled = PROC_DIR / "labeled_borrowers.csv"
    out_traj    = PROC_DIR / "trajectories.csv"

    labeled.to_csv(out_labeled, index=False)
    traj.to_csv(out_traj, index=False)

    print(f"[save]   Labeled borrowers -> {out_labeled}")
    print(f"[save]   Trajectories      -> {out_traj}")
    print(f"\nFeature columns: {list(labeled.columns)}")
    print(f"Class distribution:\n{labeled['delinquent'].value_counts().to_string()}")
    return labeled, traj


if __name__ == "__main__":
    run()
