"""
train_model.py
==============
Trains Logistic Regression (baseline) and XGBoost (main model) on the
labeled borrower dataset. Saves model artifacts and evaluation metrics.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "avg_lookback_income",
    "income_volatility_cv",
    "income_trend_slope",
    "active_weeks_pre_loan",
    "zero_income_weeks_pre_loan",
    "loan_amount",
    "installment",
]
TARGET_COL = "delinquent"
RANDOM_SEED = 42


def load_data():
    df = pd.read_csv(PROC_DIR / "labeled_borrowers.csv")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    meta = df[["farmer_id", "total_missed_payments"]]
    print(f"[data]  Shape: {X.shape}  |  Delinquency rate: {y.mean():.1%}")
    return X, y, meta, df


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )


def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_sc, y_train)
    print("[LR]    Baseline Logistic Regression trained.")
    return lr, scaler


def train_xgboost(X_train, y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    print(f"[XGB]   XGBoost trained. scale_pos_weight={spw:.2f}")
    return xgb_model


def evaluate_model(name, model, X_test, y_test, scaler=None, threshold=0.5):
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    proba = model.predict_proba(X_eval)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc  = roc_auc_score(y_test, proba)
    ap   = average_precision_score(y_test, proba)
    cm   = confusion_matrix(y_test, preds)
    cr   = classification_report(y_test, preds, target_names=["Non-Delinquent", "Delinquent"], output_dict=True)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print(classification_report(y_test, preds, target_names=["Non-Delinquent", "Delinquent"]))

    metrics = {
        "name": name,
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "recall_delinquent": round(cr["Delinquent"]["recall"], 4),
        "precision_delinquent": round(cr["Delinquent"]["precision"], 4),
        "f1_delinquent": round(cr["Delinquent"]["f1-score"], 4),
        "accuracy": round(cr["accuracy"], 4),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }
    return metrics, proba


def save_plots(lr_proba, xgb_proba, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")

    # ROC curves
    for proba, label, color in [
        (lr_proba, "Logistic Regression", "#89b4fa"),
        (xgb_proba, "XGBoost", "#a6e3a1"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#585b70", lw=1)
    axes[0].set_title("ROC Curve"); axes[0].legend(facecolor="#313244", labelcolor="#cdd6f4")

    # Precision-Recall curves
    for proba, label, color in [
        (lr_proba, "Logistic Regression", "#89b4fa"),
        (xgb_proba, "XGBoost", "#a6e3a1"),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, label=f"{label} (AP={ap:.3f})", color=color, lw=2)
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(facecolor="#313244", labelcolor="#cdd6f4")

    # XGBoost feature importance
    xgb_model_loaded = xgb.XGBClassifier()
    # Use probabilities proxy since we have the model
    axes[2].set_title("XGBoost Feature Importance (Gain)")

    plt.tight_layout()
    out_path = MODEL_DIR / "model_evaluation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"[plots] Saved -> {out_path}")


def run():
    print("=" * 60)
    print("  Model Training Pipeline")
    print("=" * 60)

    X, y, meta, df = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save split indices for reproducibility
    test_idx  = X_test.index.tolist()
    train_idx = X_train.index.tolist()

    # ── Logistic Regression ──────────────────────────────────────────────
    lr, scaler = train_logistic(X_train, y_train)
    lr_metrics, lr_proba = evaluate_model("Logistic Regression", lr, X_test, y_test, scaler=scaler)

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, xgb_proba = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    # ── Save models ───────────────────────────────────────────────────────
    joblib.dump(lr,      MODEL_DIR / "lr_baseline.pkl")
    joblib.dump(scaler,  MODEL_DIR / "lr_scaler.pkl")
    xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))

    # Save metrics
    all_metrics = {"logistic_regression": lr_metrics, "xgboost": xgb_metrics}
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save test predictions for downstream modules
    test_df = df.iloc[test_idx].copy()
    test_df["lr_proba"]  = lr_proba
    test_df["xgb_proba"] = xgb_proba
    test_df["xgb_pred_default"] = (xgb_proba >= 0.5).astype(int)

    # Add risk tier
    def risk_tier(p):
        if p < 0.33:  return "Low"
        if p < 0.66:  return "Medium"
        return "High"
    test_df["risk_tier"] = test_df["xgb_proba"].apply(risk_tier)

    test_df.to_csv(PROC_DIR / "test_predictions.csv", index=False)

    # Save train data predictions for fairness/stress modules
    train_df = df.iloc[train_idx].copy()
    train_scaler = scaler
    lr_train_proba  = lr.predict_proba(train_scaler.transform(X_train))[:, 1]
    xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    train_df["lr_proba"]  = lr_train_proba
    train_df["xgb_proba"] = xgb_train_proba
    train_df.to_csv(PROC_DIR / "train_predictions.csv", index=False)

    # Save split info
    split_info = {"train_indices": train_idx, "test_indices": test_idx, "random_seed": RANDOM_SEED}
    with open(MODEL_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n[save]  LR model       -> {MODEL_DIR}/lr_baseline.pkl")
    print(f"[save]  XGBoost model  -> {MODEL_DIR}/xgb_model.json")
    print(f"[save]  Metrics        -> {MODEL_DIR}/model_metrics.json")
    print(f"[save]  Test preds     -> {PROC_DIR}/test_predictions.csv")

    return xgb_model, lr, scaler, test_df


if __name__ == "__main__":
    run()
