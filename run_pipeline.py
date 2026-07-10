"""
run_pipeline.py
===============
Single entry point that runs the entire EWS pipeline in sequence:
  1. Loan Simulation
  2. Model Training
  3. SHAP Explanations
  4. Intervention Mapping
  5. Cost-Sensitive Threshold
  6. Fairness Audit
  7. Stress Test

Usage:
    python run_pipeline.py
    python run_pipeline.py --data-file path/to/custom.csv
"""

import sys
import time
import argparse
from pathlib import Path

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))


def banner(title: str):
    print(f"\n{'#'*60}")
    print(f"#  {title}")
    print(f"{'#'*60}\n")


def run_step(name: str, fn, *args, **kwargs):
    banner(f"STEP: {name}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"\nOK {name} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Microfinance EWS Pipeline")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to raw CSV (default: data/raw/farmers_salary_transactions.csv)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  MICROFINANCE EARLY WARNING SYSTEM — FULL PIPELINE")
    print("="*60)

    start = time.time()

    # ── Step 1: Loan Simulation ───────────────────────────────────────────
    import loan_simulation as ls
    if args.data_file:
        run_step("Loan Simulation", ls.run, path=Path(args.data_file))
    else:
        run_step("Loan Simulation", ls.run)

    # ── Step 2: Model Training ────────────────────────────────────────────
    import train_model as tm
    run_step("Model Training", tm.run)

    # ── Step 3: SHAP Explanations ─────────────────────────────────────────
    import shap_explain as se
    run_step("SHAP Explanations", se.run)

    # ── Step 4: Intervention Mapping ──────────────────────────────────────
    import interventions as iv
    run_step("Intervention Mapping", iv.run)

    # ── Step 5: Cost-Sensitive Threshold ──────────────────────────────────
    import cost_threshold as ct
    run_step("Cost-Sensitive Threshold", ct.run)

    # ── Step 6: Fairness Audit ────────────────────────────────────────────
    import fairness_audit as fa
    run_step("Fairness Audit", fa.run)

    # ── Step 7: Stress Test ───────────────────────────────────────────────
    import stress_test as st
    run_step("Stress Test", st.run)

    total = time.time() - start
    print("\n" + "="*60)
    print(f"  [OK] ALL STEPS COMPLETE  ({total:.1f}s total)")
    print("="*60)
    print("\n  Generated outputs:")
    print("    data/processed/labeled_borrowers.csv")
    print("    data/processed/trajectories.csv")
    print("    data/processed/test_predictions.csv")
    print("    data/processed/shap_explanations.csv")
    print("    data/processed/borrower_interventions.csv")
    print("    data/processed/cost_threshold_sweep.csv")
    print("    data/processed/fairness_subgroup_metrics.csv")
    print("    data/processed/stress_test_results.csv")
    print("    models/lr_baseline.pkl")
    print("    models/xgb_model.json")
    print("    models/optimal_threshold.json")
    print("\n  Launch the app with:")
    print("    streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
