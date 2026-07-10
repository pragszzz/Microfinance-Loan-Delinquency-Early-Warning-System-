"""
app.py — Microfinance Early Warning System Dashboard
=====================================================
6 pages:
  1. Portfolio Overview
  2. Borrower Explorer
  3. Borrower Detail (SHAP + trajectory + intervention card)
  4. What-If Simulator
  5. Stress Test Dashboard
  6. Fairness Report
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import xgboost as xgb
import joblib
import shap

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
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

FEATURE_LABELS = {
    "avg_lookback_income"        : "Avg Pre-Loan Income",
    "income_volatility_cv"       : "Income Volatility (CV)",
    "income_trend_slope"         : "Income Trend Slope",
    "active_weeks_pre_loan"      : "Active Weeks (of 8)",
    "zero_income_weeks_pre_loan" : "Zero-Income Weeks",
    "loan_amount"                : "Loan Amount",
    "installment"                : "Weekly Installment",
}

TIER_COLORS = {"Low": "#a6e3a1", "Medium": "#f9e2af", "High": "#f38ba8"}
TIER_BG     = {"Low": "#1a2e1a", "Medium": "#2e2b1a", "High": "#2e1a1a"}
URGENCY_COLORS = {"High": "#f38ba8", "Medium": "#f9e2af", "Low": "#a6e3a1"}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MFI Early Warning System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  
  .stApp { background: #0f1117; color: #cdd6f4; }
  
  .metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #181825 100%);
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .label { font-size: 12px; color: #6c7086; text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { font-size: 32px; font-weight: 700; margin-top: 6px; }
  .metric-card .sub   { font-size: 12px; color: #6c7086; margin-top: 4px; }
  
  .tier-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  
  .intervention-card {
    background: #1a1d2e;
    border-left: 4px solid #89b4fa;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
  }
  
  .reason-card {
    background: #181825;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
  }
  
  section[data-testid="stSidebar"] {
    background: #181825;
    border-right: 1px solid #313244;
  }
  
  .stSelectbox label, .stSlider label { color: #bac2de !important; }
  
  h1 { color: #cdd6f4 !important; font-weight: 700; }
  h2 { color: #bac2de !important; font-weight: 600; }
  h3 { color: #a6adc8 !important; }
  
  .stDataFrame { background: #1a1d2e; }
  
  div[data-testid="stMetricValue"] { color: #89b4fa !important; font-weight: 700; }
  
  .page-header {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
    border-bottom: 1px solid #313244;
    padding: 24px 0 16px 0;
    margin-bottom: 28px;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_test_predictions():
    return pd.read_csv(PROC_DIR / "test_predictions.csv")

@st.cache_data
def load_interventions():
    path = PROC_DIR / "borrower_interventions.csv"
    if path.exists():
        return pd.read_csv(path)
    return load_test_predictions()

@st.cache_data
def load_trajectories():
    return pd.read_csv(PROC_DIR / "trajectories.csv")

@st.cache_data
def load_shap_raw():
    return pd.read_csv(PROC_DIR / "shap_values_raw.csv")

@st.cache_data
def load_stress_results():
    return pd.read_csv(PROC_DIR / "stress_test_results.csv")

@st.cache_data
def load_stress_summary():
    return pd.read_csv(PROC_DIR / "stress_test_summary.csv")

@st.cache_data
def load_fairness_metrics():
    sg = pd.read_csv(PROC_DIR / "fairness_subgroup_metrics.csv")
    gp = pd.read_csv(PROC_DIR / "fairness_gap_metrics.csv")
    return sg, gp

@st.cache_data
def load_threshold():
    path = MODEL_DIR / "optimal_threshold.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"optimal_threshold": 0.5, "total_cost_at_optimal": 0, "total_cost_at_default": 0, "cost_saving": 0}

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "xgb_model.json"))
    return model

@st.cache_data
def get_model_metrics():
    path = MODEL_DIR / "model_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def risk_tier(p):
    if p < 0.33:  return "Low"
    if p < 0.66:  return "Medium"
    return "High"

def tier_badge(tier):
    colors = {"Low": "#a6e3a1", "Medium": "#f9e2af", "High": "#f38ba8"}
    bg     = {"Low": "#1a2e1a", "Medium": "#2e2b1a", "High": "#2e1a1a"}
    c = colors.get(tier, "#cdd6f4")
    b = bg.get(tier, "#1a1d2e")
    return f'<span class="tier-badge" style="color:{c};background:{b};border:1px solid {c};">{tier}</span>'


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 MFI Early Warning")
    st.markdown("<hr style='border-color:#313244;margin:8px 0 16px 0'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "📊 Portfolio Overview",
            "🔍 Borrower Explorer",
            "👤 Borrower Detail",
            "🎛️ What-If Simulator",
            "⚡ Stress Test",
            "⚖️ Fairness Report",
        ],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#313244;margin:16px 0 12px 0'>", unsafe_allow_html=True)
    st.markdown("<small style='color:#6c7086'>Microfinance EWS v1.0<br>Simulated loan labels · Research prototype</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":
    st.markdown("# 📊 Portfolio Overview")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>Risk distribution across all scored borrowers</p>",
                unsafe_allow_html=True)

    df = load_test_predictions()
    metrics = get_model_metrics()

    total = len(df)
    n_high   = (df["risk_tier"] == "High").sum()
    n_medium = (df["risk_tier"] == "Medium").sum()
    n_low    = (df["risk_tier"] == "Low").sum()
    delinq_rate = df["delinquent"].mean() * 100
    avg_score   = df["xgb_proba"].mean() * 100

    # KPI metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, sub, color in [
        (c1, "Total Borrowers",   f"{total:,}",    "In test set",           "#89b4fa"),
        (c2, "High Risk",         f"{n_high:,}",   f"{n_high/total:.1%} of portfolio", "#f38ba8"),
        (c3, "Medium Risk",       f"{n_medium:,}", f"{n_medium/total:.1%} of portfolio","#f9e2af"),
        (c4, "Low Risk",          f"{n_low:,}",    f"{n_low/total:.1%} of portfolio",  "#a6e3a1"),
        (c5, "Delinquency Rate",  f"{delinq_rate:.1f}%", "Actual (simulated)",        "#cba6f7"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value" style="color:{color}">{val}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    # Risk tier donut
    with col_left:
        fig = go.Figure(go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[n_low, n_medium, n_high],
            hole=0.55,
            marker_colors=["#a6e3a1", "#f9e2af", "#f38ba8"],
            textfont=dict(color="#cdd6f4", size=13),
            textinfo="label+percent",
        ))
        fig.update_layout(
            title=dict(text="Risk Tier Distribution", font=dict(color="#cdd6f4", size=16)),
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font=dict(color="#cdd6f4"),
            legend=dict(font=dict(color="#cdd6f4"), bgcolor="#1a1d2e"),
            showlegend=True, height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk score distribution
    with col_right:
        fig2 = go.Figure()
        for tier, color in TIER_COLORS.items():
            subset = df[df["risk_tier"] == tier]["xgb_proba"]
            fig2.add_trace(go.Histogram(
                x=subset, name=f"{tier} Risk", nbinsx=20,
                marker_color=color, opacity=0.75,
            ))
        fig2.update_layout(
            title=dict(text="Risk Score Distribution", font=dict(color="#cdd6f4", size=16)),
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            xaxis=dict(title="XGBoost Risk Score", color="#6c7086", gridcolor="#313244"),
            yaxis=dict(title="Count", color="#6c7086", gridcolor="#313244"),
            font=dict(color="#cdd6f4"), barmode="overlay",
            legend=dict(font=dict(color="#cdd6f4"), bgcolor="#1a1d2e"),
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Model performance
    st.markdown("### 📈 Model Performance")
    xgb_m = metrics.get("xgboost", {})
    lr_m  = metrics.get("logistic_regression", {})

    perf_data = {
        "Metric"    : ["ROC-AUC", "Recall (Delinquent)", "Precision (Delinquent)", "F1 (Delinquent)", "Accuracy"],
        "XGBoost"   : [xgb_m.get("roc_auc","—"), xgb_m.get("recall_delinquent","—"),
                       xgb_m.get("precision_delinquent","—"), xgb_m.get("f1_delinquent","—"), xgb_m.get("accuracy","—")],
        "Logistic Reg": [lr_m.get("roc_auc","—"), lr_m.get("recall_delinquent","—"),
                         lr_m.get("precision_delinquent","—"), lr_m.get("f1_delinquent","—"), lr_m.get("accuracy","—")],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df.set_index("Metric"), use_container_width=True)

    st.markdown("""
    <div class="intervention-card">
    <b>⚠️ Limitations</b><br>
    Delinquency labels are <b>simulated</b>, not observed real-world defaults. Features are derived from a short 8-week lookback window.
    This is a research prototype — not for production lending decisions.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — BORROWER EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Borrower Explorer":
    st.markdown("# 🔍 Borrower Explorer")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>Browse and filter all scored borrowers</p>",
                unsafe_allow_html=True)

    df = load_interventions()

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        tier_filter = st.multiselect("Risk Tier", ["Low", "Medium", "High"], default=["Low","Medium","High"])
    with fc2:
        show_delinquent = st.selectbox("Label", ["All", "Delinquent Only", "Non-Delinquent Only"])
    with fc3:
        sort_by = st.selectbox("Sort By", ["xgb_proba ↓", "xgb_proba ↑", "farmer_id"])

    filtered = df[df["risk_tier"].isin(tier_filter)].copy()
    if show_delinquent == "Delinquent Only":
        filtered = filtered[filtered["delinquent"] == 1]
    elif show_delinquent == "Non-Delinquent Only":
        filtered = filtered[filtered["delinquent"] == 0]

    sort_map = {"xgb_proba ↓": ("xgb_proba", False), "xgb_proba ↑": ("xgb_proba", True), "farmer_id": ("farmer_id", True)}
    sk, sasc = sort_map[sort_by]
    filtered = filtered.sort_values(sk, ascending=sasc)

    st.markdown(f"**{len(filtered):,} borrowers** matching filters")

    # Display table
    display_cols = ["farmer_id", "avg_lookback_income", "income_volatility_cv",
                    "xgb_proba", "risk_tier", "delinquent",
                    "primary_reason" if "primary_reason" in filtered.columns else "xgb_proba",
                    "intervention_action" if "intervention_action" in filtered.columns else "xgb_proba"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    show_df = filtered[display_cols].copy()
    show_df["xgb_proba"] = show_df["xgb_proba"].apply(lambda x: f"{x:.3f}")
    show_df["avg_lookback_income"] = show_df["avg_lookback_income"].apply(lambda x: f"₹{x:,.0f}")
    show_df["income_volatility_cv"] = show_df["income_volatility_cv"].apply(lambda x: f"{x:.3f}")

    st.dataframe(
        show_df.rename(columns={
            "farmer_id": "ID", "avg_lookback_income": "Avg Income",
            "income_volatility_cv": "Volatility CV", "xgb_proba": "Risk Score",
            "risk_tier": "Tier", "delinquent": "Actual Label",
            "primary_reason": "Primary Risk Driver", "intervention_action": "Recommended Action",
        }),
        use_container_width=True, height=480,
    )

    st.info("👆 Go to **Borrower Detail** page and enter a Farmer ID to see full explanation and intervention.")


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — BORROWER DETAIL
# ══════════════════════════════════════════════════════════════════════════
elif page == "👤 Borrower Detail":
    st.markdown("# 👤 Borrower Detail")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>Per-borrower SHAP explanation, risk trajectory, and intervention</p>",
                unsafe_allow_html=True)

    df      = load_interventions()
    traj    = load_trajectories()
    shap_df = load_shap_raw()

    farmer_ids = sorted(df["farmer_id"].astype(str).tolist())
    selected = st.selectbox("Select Farmer ID", farmer_ids)

    if selected:
        row    = df[df["farmer_id"].astype(str) == selected].iloc[0]
        t_row  = traj[traj["farmer_id"].astype(str) == selected]
        idx    = df[df["farmer_id"].astype(str) == selected].index[0]

        # ── Header ───────────────────────────────────────────────────────
        tier   = row["risk_tier"]
        score  = row["xgb_proba"]
        actual = "✅ Non-Delinquent" if row["delinquent"] == 0 else "❌ Delinquent"
        tc     = TIER_COLORS.get(tier, "#cdd6f4")

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.markdown(f"""<div class="metric-card"><div class="label">Farmer ID</div>
            <div class="value" style="color:#89b4fa;font-size:24px">{selected}</div></div>""",
            unsafe_allow_html=True)
        col_h2.markdown(f"""<div class="metric-card"><div class="label">Risk Score</div>
            <div class="value" style="color:{tc}">{score:.3f}</div></div>""",
            unsafe_allow_html=True)
        col_h3.markdown(f"""<div class="metric-card"><div class="label">Risk Tier</div>
            <div class="value" style="color:{tc}">{tier}</div></div>""",
            unsafe_allow_html=True)
        col_h4.markdown(f"""<div class="metric-card"><div class="label">Actual Label</div>
            <div class="value" style="font-size:18px">{actual}</div></div>""",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 1])

        with col_left:
            # ── SHAP waterfall ────────────────────────────────────────────
            st.markdown("#### 🧠 Risk Driver Breakdown (SHAP)")
            shap_row = shap_df.iloc[idx][FEATURE_COLS]
            feat_vals = {f: row[f] for f in FEATURE_COLS if f in row.index}

            shap_sorted = shap_row.sort_values(key=abs, ascending=True)
            colors_shap = ["#f38ba8" if v > 0 else "#a6e3a1" for v in shap_sorted.values]
            labels = [FEATURE_LABELS.get(f, f) for f in shap_sorted.index]

            fig_shap = go.Figure(go.Bar(
                x=shap_sorted.values,
                y=labels,
                orientation="h",
                marker_color=colors_shap,
                text=[f"{v:+.4f}" for v in shap_sorted.values],
                textposition="outside",
                textfont=dict(color="#cdd6f4", size=11),
            ))
            fig_shap.update_layout(
                paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                xaxis=dict(title="SHAP Value (impact on risk score)", color="#6c7086", gridcolor="#313244", zeroline=True, zerolinecolor="#585b70"),
                yaxis=dict(color="#cdd6f4"),
                font=dict(color="#cdd6f4"), height=320, margin=dict(l=0, r=60, t=20, b=40),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # ── Reason codes ──────────────────────────────────────────────
            st.markdown("#### 📌 Reason Codes")
            if "primary_reason" in row:
                st.markdown(f"""<div class="reason-card">
                  <b style="color:#89b4fa">Primary:</b> {row.get('primary_reason','—')}
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="reason-card">
                  <b style="color:#cba6f7">Secondary:</b> {row.get('secondary_reason','—')}
                </div>""", unsafe_allow_html=True)

        with col_right:
            # ── Risk trajectory ───────────────────────────────────────────
            if not t_row.empty:
                st.markdown("#### 📈 Risk Trajectory (Cumulative Missed Payments)")
                cp_cols = [c for c in t_row.columns if c.startswith("missed_by_week_")]
                weeks   = [int(c.split("_")[-1]) for c in cp_cols]
                misses  = t_row[cp_cols].values[0].tolist()

                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter(
                    x=weeks, y=misses,
                    mode="lines+markers+text",
                    text=[str(int(m)) for m in misses],
                    textposition="top center",
                    line=dict(color=tc, width=3),
                    marker=dict(size=10, color=tc, line=dict(color="#1a1d2e", width=2)),
                    fill="tozeroy",
                    fillcolor=f"rgba(243,139,168,0.1)" if tier=="High" else "rgba(166,227,161,0.1)",
                    name="Missed Payments",
                ))
                fig_traj.add_hline(y=4, line_dash="dot", line_color="#f38ba8",
                                   annotation_text="Delinquency threshold (4)", annotation_font_color="#f38ba8")
                fig_traj.update_layout(
                    paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                    xaxis=dict(title="Loan Week Checkpoint", color="#6c7086", gridcolor="#313244",
                               tickvals=weeks, ticktext=[f"Wk {w}" for w in weeks]),
                    yaxis=dict(title="Cumulative Missed Payments", color="#6c7086", gridcolor="#313244"),
                    font=dict(color="#cdd6f4"), height=300, margin=dict(l=0, r=20, t=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_traj, use_container_width=True)

            # ── Intervention card ─────────────────────────────────────────
            st.markdown("#### 💡 Recommended Intervention")
            if "intervention_action" in row:
                urgency = row.get("intervention_urgency", "Medium")
                uc = URGENCY_COLORS.get(urgency, "#cdd6f4")
                icon = row.get("intervention_icon", "🔍")
                st.markdown(f"""
                <div class="intervention-card" style="border-left-color:{uc};">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <span style="font-size:24px">{icon}</span>
                    <span style="font-weight:700;color:{uc};font-size:15px">{row['intervention_action']}</span>
                    <span style="margin-left:auto;background:{uc}22;color:{uc};padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600">{urgency} Urgency</span>
                  </div>
                  <p style="color:#bac2de;font-size:13px;line-height:1.6;margin:0">{row['intervention_detail']}</p>
                </div>""", unsafe_allow_html=True)

        # ── Feature values table ──────────────────────────────────────────
        st.markdown("#### 📋 Feature Values")
        feat_display = {FEATURE_LABELS.get(f, f): [f"₹{row[f]:,.0f}" if "income" in f or f in ("loan_amount","installment") else f"{row[f]:.3f}"]
                        for f in FEATURE_COLS if f in row.index}
        st.dataframe(pd.DataFrame(feat_display).T.rename(columns={0: "Value"}), use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎛️ What-If Simulator":
    st.markdown("# 🎛️ What-If Simulator")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>Adjust borrower features and see live risk score updates</p>",
                unsafe_allow_html=True)

    model = load_model()
    df    = load_test_predictions()
    thresh_info = load_threshold()
    threshold   = thresh_info.get("optimal_threshold", 0.5)

    st.markdown("### Adjust Borrower Profile")
    st.markdown("<p style='color:#6c7086;font-size:13px'>Move sliders to modify features — risk score updates instantly</p>",
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([1, 1, 1])

    with col_s1:
        avg_income = st.slider(
            "💰 Avg Pre-Loan Weekly Income (₹)",
            min_value=0, max_value=int(df["avg_lookback_income"].max() * 1.2),
            value=int(df["avg_lookback_income"].median()),
            step=100, key="wi_avg_income",
        )
        volatility_cv = st.slider(
            "📊 Income Volatility (CV)",
            min_value=0.0, max_value=float(df["income_volatility_cv"].max() * 1.1),
            value=float(df["income_volatility_cv"].median()),
            step=0.01, key="wi_cv",
        )
        trend_slope = st.slider(
            "📈 Income Trend Slope",
            min_value=float(df["income_trend_slope"].min() * 1.2),
            max_value=float(df["income_trend_slope"].max() * 1.2),
            value=0.0,
            step=10.0, key="wi_slope",
        )

    with col_s2:
        active_weeks = st.slider(
            "✅ Active Weeks (of 8)",
            min_value=0, max_value=8,
            value=6, step=1, key="wi_active",
        )
        zero_weeks = st.slider(
            "⚠️ Zero-Income Weeks",
            min_value=0, max_value=8,
            value=2, step=1, key="wi_zero",
        )

    with col_s3:
        # Auto-derived from avg income and affordability ratio
        installment = avg_income * 0.35
        loan_amount = installment * 20

        st.markdown(f"""
        <div class="metric-card" style="margin-top:12px">
          <div class="label">Weekly Installment (auto)</div>
          <div class="value" style="color:#89b4fa;font-size:22px">₹{installment:,.0f}</div>
          <div class="sub">35% of avg income</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="margin-top:12px">
          <div class="label">Loan Amount (auto)</div>
          <div class="value" style="color:#cba6f7;font-size:22px">₹{loan_amount:,.0f}</div>
          <div class="sub">20-week term</div>
        </div>""", unsafe_allow_html=True)

    # ── Live prediction ───────────────────────────────────────────────────
    X_whatif = np.array([[avg_income, volatility_cv, trend_slope,
                           active_weeks, zero_weeks, loan_amount, installment]])
    X_df = pd.DataFrame(X_whatif, columns=FEATURE_COLS)

    proba = model.predict_proba(X_df)[0][1]
    tier  = risk_tier(proba)
    tc    = TIER_COLORS.get(tier, "#cdd6f4")
    bc    = TIER_BG.get(tier, "#1a1d2e")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Live Risk Assessment")

    res1, res2, res3 = st.columns(3)
    res1.markdown(f"""<div class="metric-card" style="border-color:{tc};">
      <div class="label">Risk Score</div>
      <div class="value" style="color:{tc}">{proba:.3f}</div>
      <div class="sub">XGBoost probability</div>
    </div>""", unsafe_allow_html=True)
    res2.markdown(f"""<div class="metric-card" style="border-color:{tc};">
      <div class="label">Risk Tier</div>
      <div class="value" style="color:{tc}">{tier}</div>
      <div class="sub">Threshold: {threshold:.2f}</div>
    </div>""", unsafe_allow_html=True)
    predicted_label = "⚠️ Delinquency Risk" if proba >= threshold else "✅ Low Delinquency Risk"
    res3.markdown(f"""<div class="metric-card" style="border-color:{tc};">
      <div class="label">Prediction</div>
      <div class="value" style="font-size:20px">{predicted_label}</div>
    </div>""", unsafe_allow_html=True)

    # ── Gauge chart ───────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        title={"text": "Delinquency Risk Score (%)", "font": {"color": "#cdd6f4", "size": 16}},
        gauge={
            "axis"       : {"range": [0, 100], "tickcolor": "#6c7086"},
            "bar"        : {"color": tc},
            "steps"      : [
                {"range": [0, 33],   "color": "#1a2e1a"},
                {"range": [33, 66],  "color": "#2e2b1a"},
                {"range": [66, 100], "color": "#2e1a1a"},
            ],
            "threshold"  : {"line": {"color": "#cdd6f4", "width": 3},
                            "thickness": 0.75, "value": threshold * 100},
        },
        number={"font": {"color": tc, "size": 40}, "suffix": "%"},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#1a1d2e", font=dict(color="#cdd6f4"), height=300,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Comparison to portfolio ───────────────────────────────────────────
    pct_rank = (df["xgb_proba"] < proba).mean() * 100
    st.markdown(f"<p style='color:#6c7086;text-align:center;font-size:13px'>"
                f"This borrower is riskier than <b style='color:#89b4fa'>{pct_rank:.0f}%</b> of the portfolio</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 5 — STRESS TEST
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Stress Test":
    st.markdown("# ⚡ Stress Test Dashboard")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>Income shock simulation: how does the portfolio shift under crisis?</p>",
                unsafe_allow_html=True)

    try:
        summary_df = load_stress_summary()
        results_df = load_stress_results()
    except Exception as e:
        st.error(f"Stress test data not found. Run `python run_pipeline.py` first.\n\n{e}")
        st.stop()

    scenarios = summary_df["scenario"].tolist()

    # ── Stacked bar: tier distribution per scenario ───────────────────────
    fig_stack = go.Figure()
    for tier, color in [("pct_low","#a6e3a1"), ("pct_medium","#f9e2af"), ("pct_high","#f38ba8")]:
        label = tier.replace("pct_", "").capitalize() + " Risk"
        fig_stack.add_trace(go.Bar(
            name=label,
            x=scenarios,
            y=(summary_df[tier] * 100).round(1),
            marker_color=color,
            text=(summary_df[tier] * 100).round(1).astype(str) + "%",
            textposition="inside",
        ))
    fig_stack.update_layout(
        barmode="stack",
        title=dict(text="Risk Tier Distribution Under Income Shocks", font=dict(color="#cdd6f4", size=16)),
        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
        xaxis=dict(color="#6c7086", gridcolor="#313244"),
        yaxis=dict(title="% of Borrowers", color="#6c7086", gridcolor="#313244", range=[0,100]),
        font=dict(color="#cdd6f4"),
        legend=dict(font=dict(color="#cdd6f4"), bgcolor="#1a1d2e"),
        height=380,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Delinquency rate line
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=scenarios,
            y=(summary_df["pct_delinquent_pred"] * 100).round(1),
            mode="lines+markers+text",
            text=(summary_df["pct_delinquent_pred"] * 100).round(1).astype(str) + "%",
            textposition="top center",
            line=dict(color="#f38ba8", width=3),
            marker=dict(size=10, color="#f38ba8"),
            fill="tozeroy",
            fillcolor="rgba(243,139,168,0.1)",
        ))
        fig_line.update_layout(
            title=dict(text="Predicted Delinquency Rate Under Shock", font=dict(color="#cdd6f4", size=14)),
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            xaxis=dict(color="#6c7086", gridcolor="#313244"),
            yaxis=dict(title="% Predicted Delinquent", color="#6c7086", gridcolor="#313244"),
            font=dict(color="#cdd6f4"), height=320, showlegend=False,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_r:
        # Migration matrix
        st.markdown("#### Tier Migration: Baseline → Severe (-30%)")
        try:
            mig_df = pd.read_csv(PROC_DIR / "stress_migration_matrix.csv", index_col=0)
            st.dataframe(mig_df, use_container_width=True)
            st.markdown("<small style='color:#6c7086'>Rows = Baseline tier, Columns = Severe scenario tier.<br>Diagonal = stayed same tier.</small>",
                        unsafe_allow_html=True)
        except Exception:
            st.info("Migration matrix not found.")

    # Summary table
    st.markdown("#### Scenario Summary")
    show_summary = summary_df[["scenario","n_low","n_medium","n_high","pct_delinquent_pred"]].copy()
    show_summary["pct_delinquent_pred"] = (show_summary["pct_delinquent_pred"] * 100).round(1).astype(str) + "%"
    show_summary.columns = ["Scenario", "# Low", "# Medium", "# High", "% Pred. Delinquent"]
    st.dataframe(show_summary.set_index("Scenario"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 6 — FAIRNESS REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Report":
    st.markdown("# ⚖️ Fairness Report")
    st.markdown("<p style='color:#6c7086;margin-top:-12px'>LR vs XGBoost — fairness across income subgroups (Q1 lowest vs Q4 highest income)</p>",
                unsafe_allow_html=True)

    try:
        sg_df, gaps_df = load_fairness_metrics()
    except Exception as e:
        st.error(f"Fairness data not found. Run `python run_pipeline.py` first.\n\n{e}")
        st.stop()

    thresh_info = load_threshold()

    # Interpretation card
    st.markdown("""
    <div class="intervention-card" style="border-left-color:#cba6f7;">
    <b style='color:#cba6f7'>📐 Fairness Definition</b><br>
    <b>Demographic Parity</b>: Do both income groups receive similar prediction rates? (Lower gap = more fair)<br>
    <b>Equalized Odds</b>: Do both groups experience similar True Positive Rate and False Positive Rate?<br>
    <b>Subgroup</b>: Q1 = bottom 25% income, Q4 = top 25% income — the most vulnerable vs most capable borrowers.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gap metrics comparison
    col_lr, col_xgb = st.columns(2)
    for col, model_name, color in [(col_lr, "Logistic Regression", "#89b4fa"), (col_xgb, "XGBoost", "#a6e3a1")]:
        row = gaps_df[gaps_df["model"] == model_name]
        if row.empty:
            continue
        row = row.iloc[0]
        with col:
            st.markdown(f"### {model_name}")
            metrics_show = {
                "Demographic Parity Diff" : (row["demographic_parity_diff"], "Lower is more fair. 0 = perfect parity."),
                "Equalized Odds Gap"      : (row["equalized_odds_gap"], "Average of TPR gap + FPR gap."),
                "TPR Gap (Recall)"        : (row["tpr_gap"], "Difference in recall between Q1 and Q4."),
                "FPR Gap"                 : (row["fpr_gap"], "Difference in false alarm rate between groups."),
            }
            for metric, (val, desc) in metrics_show.items():
                st.markdown(f"""
                <div class="reason-card">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:#bac2de;font-size:13px">{metric}</span>
                    <span style="color:{color};font-size:20px;font-weight:700">{val:.4f}</span>
                  </div>
                  <div style="color:#6c7086;font-size:11px;margin-top:4px">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # Subgroup recall comparison bar chart
    st.markdown("### Recall (Delinquent Class) by Subgroup")
    fig_fair = go.Figure()
    for model_name, color in [("Logistic Regression", "#89b4fa"), ("XGBoost", "#a6e3a1")]:
        sg_rows = sg_df[sg_df["model"] == model_name]
        fig_fair.add_trace(go.Bar(
            name=model_name,
            x=sg_rows["group"].tolist(),
            y=sg_rows["recall_deliq"].tolist(),
            marker_color=color,
            text=[f"{v:.2f}" for v in sg_rows["recall_deliq"].tolist()],
            textposition="outside",
        ))
    fig_fair.update_layout(
        barmode="group",
        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
        xaxis=dict(color="#6c7086", gridcolor="#313244"),
        yaxis=dict(title="Recall (Delinquent)", color="#6c7086", gridcolor="#313244", range=[0, 1.15]),
        font=dict(color="#cdd6f4"),
        legend=dict(font=dict(color="#cdd6f4"), bgcolor="#1a1d2e"),
        height=350,
    )
    st.plotly_chart(fig_fair, use_container_width=True)

    # Detailed subgroup table
    st.markdown("### Full Subgroup Metrics Table")
    st.dataframe(sg_df.set_index(["model","group"]), use_container_width=True)

    # Honest findings
    st.markdown("""
    <div class="intervention-card" style="border-left-color:#f9e2af;">
    <b style='color:#f9e2af'>⚠️ Honest Findings</b><br>
    Fairness analysis is performed on income-quartile subgroups — the only meaningful proxy axis available in this dataset.
    No demographic attributes (gender, age, region) are present in the data.
    If one model is more accurate but less fair, this tradeoff is explicitly documented here rather than hidden.
    A production MFI system would require demographic fairness audits and regulatory compliance review.
    </div>""", unsafe_allow_html=True)
