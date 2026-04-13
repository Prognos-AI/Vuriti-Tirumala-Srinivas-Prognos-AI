import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PrognosAI RUL Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg: #181a20;
        --panel: #23272f;
        --ink: #f3f4f6;
        --accent: #3b82f6;
        --accent-2: #6366f1;
        --ok: #22c55e;
        --warn: #f59e42;
        --crit: #ef4444;
    }
    .stApp {
        background: linear-gradient(120deg, #181a20 0%, #23272f 100%);
        color: var(--ink);
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em;
        color: var(--ink) !important;
    }
    div[data-testid='stMetric'] {
        background: var(--panel);
        border: 1px solid #23272f;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        margin-bottom: 8px;
        color: #f3f4f6 !important;
        font-weight: 600;
        text-shadow: none;
    }
    div[data-testid='stMetric'] span {
        color: #f3f4f6 !important;
        font-weight: 700;
        text-shadow: none;
    }
    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-left: 8px;
    }
    .tag-ok { background: #1e2e22; color: var(--ok); }
    .tag-warn { background: #2d2412; color: var(--warn); }
    .tag-crit { background: #2d1818; color: var(--crit); }
    .stTabs [data-baseweb=\"tab-list\"] {
        gap: 2rem;
    }
    /* Sidebar improvements */
    section[data-testid='stSidebar'] {
        background: #181a20 !important;
        color: #f3f4f6 !important;
    }
    section[data-testid='stSidebar'] label, section[data-testid='stSidebar'] div, section[data-testid='stSidebar'] span {
        color: #f3f4f6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Prefer the detailed file when available; fall back to the compact export.
    try:
        pred_df = pd.read_csv("artifacts/test_predictions_detailed.csv")
    except FileNotFoundError:
        pred_df = pd.read_csv("artifacts/test_predictions.csv")
    metric_df = pd.read_csv("artifacts/per_dataset_metrics.csv")
    risk_df = pd.read_csv("artifacts/engine_risk.csv")
    return pred_df, metric_df, risk_df


def _to_alert_level(series: pd.Series) -> pd.Series:
    return np.where(series <= 30, "CRITICAL", np.where(series <= 60, "WARNING", "HEALTHY"))


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Backfill model-specific columns from best-prediction exports.
    if "pred_lstm" not in out.columns and "pred_best" in out.columns:
        out["pred_lstm"] = out["pred_best"]
    if "pred_gru" not in out.columns and "pred_best" in out.columns:
        out["pred_gru"] = out["pred_best"]
    if "residual_lstm" not in out.columns and "residual_best" in out.columns:
        out["residual_lstm"] = out["residual_best"]
    if "residual_gru" not in out.columns and "residual_best" in out.columns:
        out["residual_gru"] = out["residual_best"]

    # Derive alert labels when not present in the exported file.
    if "alert_level" not in out.columns:
        if "pred_best" in out.columns:
            out["alert_level"] = _to_alert_level(out["pred_best"])
        elif "pred_gru" in out.columns:
            out["alert_level"] = _to_alert_level(out["pred_gru"])
        elif "pred_lstm" in out.columns:
            out["alert_level"] = _to_alert_level(out["pred_lstm"])
        else:
            out["alert_level"] = "HEALTHY"

    return out


def compute_metrics(df: pd.DataFrame, model_choice: str) -> dict[str, float]:
    pred_col = "pred_gru" if model_choice == "GRU" else "pred_lstm"
    actual = df["actual_rul"].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)

    mse = np.mean((actual - pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - pred)))

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def build_engine_trend(engine_df: pd.DataFrame, engine_id: str, model_choice: str) -> go.Figure:
    pred_col = "pred_gru" if model_choice == "GRU" else "pred_lstm"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=engine_df["actual_rul"],
            mode="lines",
            name="Actual RUL",
            line={"color": "#1f77b4", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[pred_col],
            mode="lines",
            name=f"Predicted RUL ({model_choice})",
            line={"color": "#ef6c00", "width": 2, "dash": "dash"},
        )
    )

    fig.add_hrect(y0=0, y1=30, fillcolor="#ffcdd2", opacity=0.35, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#ffe0b2", opacity=0.35, line_width=0)

    fig.update_layout(
        title=f"Engine Timeline: {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="RUL (cycles)",
        legend_title="Series",
        template="plotly_white",
        height=430,
    )
    return fig



st.markdown("""
<div style='display: flex; align-items: center; gap: 18px;'>
    <span style='font-size:2.5rem;'>🤖</span>
    <h1 style='margin-bottom:0;'>PrognosAI Predictive Maintenance</h1>
</div>
""", unsafe_allow_html=True)

try:
    predictions, per_dataset, engine_risk = load_artifacts()
    predictions = normalize_predictions(predictions)
except FileNotFoundError:
    st.error(
        "Missing artifacts. Generate: artifacts/test_predictions_detailed.csv (or test_predictions.csv), "
        "artifacts/per_dataset_metrics.csv, and artifacts/engine_risk.csv"
    )
    st.stop()


with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("<hr style='margin:0 0 10px 0;'>", unsafe_allow_html=True)
    model_choice = st.radio("Model", ["GRU", "LSTM"], horizontal=True)
    all_datasets = sorted(predictions["dataset"].unique().tolist())
    selected_datasets = st.multiselect(
        "Datasets",
        options=all_datasets,
        default=all_datasets,
    )
    all_alerts = ["CRITICAL", "WARNING", "HEALTHY"]
    selected_alerts = st.multiselect(
        "Alert Levels",
        options=all_alerts,
        default=all_alerts,
    )
    cycle_min = int(predictions["cycle"].min())
    cycle_max = int(predictions["cycle"].max())
    selected_cycle = st.slider("Cycle Range", min_value=cycle_min, max_value=cycle_max, value=(cycle_min, cycle_max))
    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.markdown("<b>Alert Legend</b>", unsafe_allow_html=True)
    st.markdown(
        "<span class='tag tag-crit'>CRITICAL ≤ 30</span>"
        "<span class='tag tag-warn'>WARNING ≤ 60</span>"
        "<span class='tag tag-ok'>HEALTHY > 60</span>",
        unsafe_allow_html=True,
    )

filtered = predictions[
    (predictions["dataset"].isin(selected_datasets))
    & (predictions["alert_level"].isin(selected_alerts))
    & (predictions["cycle"].between(selected_cycle[0], selected_cycle[1]))
].copy()

if filtered.empty:
    st.warning("No rows match the current filter settings.")
    st.stop()

metrics = compute_metrics(filtered, model_choice)


with st.container():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(filtered):,}")
    c2.metric("Engines", f"{filtered['unit_id'].nunique():,}")
    c3.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c4.metric("R2", f"{metrics['R2']:.3f}")

pred_col = "pred_gru" if model_choice == "GRU" else "pred_lstm"
res_col = "residual_gru" if model_choice == "GRU" else "residual_lstm"


tab1, tab2 = st.tabs(["📈 RUL Scatter & Residuals", "🔔 Alerts & Metrics"])

with tab1:
    left, right = st.columns([1.4, 1])
    with left:
        scatter = px.scatter(
            filtered,
            x="actual_rul",
            y=pred_col,
            color="dataset",
            opacity=0.5,
            title=f"Actual vs Predicted RUL ({model_choice})",
            labels={pred_col: f"Predicted RUL ({model_choice})", "actual_rul": "Actual RUL"},
        )
        lim_min = float(min(filtered["actual_rul"].min(), filtered[pred_col].min()))
        lim_max = float(max(filtered["actual_rul"].max(), filtered[pred_col].max()))
        scatter.add_trace(
            go.Scatter(
                x=[lim_min, lim_max],
                y=[lim_min, lim_max],
                mode="lines",
                name="Ideal",
                line={"dash": "dash", "color": "#6366f1"},
            )
        )
        scatter.update_layout(
            template="plotly_dark",
            height=450,
            plot_bgcolor="#23272f",
            paper_bgcolor="#181a20",
            font_color="#f3f4f6",
            xaxis=dict(gridcolor="#353945"),
            yaxis=dict(gridcolor="#353945"),
        )
        st.plotly_chart(scatter, width="stretch")

    with right:
        hist = px.histogram(
            filtered,
            x=res_col,
            nbins=45,
            color_discrete_sequence=["#3b82f6"],
            title=f"Residual Distribution ({model_choice})",
            labels={res_col: "Prediction Error (Pred - Actual)"},
        )
        hist.add_vline(x=0, line_dash="dash", line_color="#23272f")
        hist.update_layout(
            template="plotly_dark",
            height=450,
            plot_bgcolor="#23272f",
            paper_bgcolor="#181a20",
            font_color="#f3f4f6",
            xaxis=dict(gridcolor="#353945"),
            yaxis=dict(gridcolor="#353945"),
        )
        st.plotly_chart(hist, width="stretch")

with tab2:
    row2_col1, row2_col2 = st.columns([1, 1])
    with row2_col1:
        alert_counts = (
            filtered["alert_level"]
            .value_counts()
            .reindex(["CRITICAL", "WARNING", "HEALTHY"], fill_value=0)
            .reset_index()
        )
        alert_counts.columns = ["alert_level", "count"]

        alert_fig = px.pie(
            alert_counts,
            names="alert_level",
            values="count",
            color="alert_level",
            color_discrete_map={
                "CRITICAL": "#ef4444",
                "WARNING": "#f59e42",
                "HEALTHY": "#22c55e",
            },
            title="Alert Distribution",
        )
        alert_fig.update_traces(textposition="inside", textinfo="percent+label")
        alert_fig.update_layout(
            template="plotly_dark",
            height=380,
            showlegend=True,
            plot_bgcolor="#23272f",
            paper_bgcolor="#181a20",
            font_color="#f3f4f6",
        )
        st.plotly_chart(alert_fig, width="stretch")

    with row2_col2:
        st.subheader("Per-Dataset Metrics")
        st.dataframe(per_dataset, width="stretch", hide_index=True)


st.markdown("<hr style='margin:2.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)
st.subheader("Engine Risk Explorer", divider="rainbow")

risk_filtered = engine_risk[engine_risk["dataset"].isin(selected_datasets)].copy()
ranked_engine_options = risk_filtered.sort_values("min_predicted_rul")["unit_id"].tolist()

if not ranked_engine_options:
    st.warning("No engines available for selected dataset filter.")
    st.stop()

selected_engine = st.selectbox(
    "Select Engine",
    options=ranked_engine_options,
    index=0,
)

engine_rows = filtered[filtered["unit_id"] == selected_engine].sort_values("cycle")
if engine_rows.empty:
    st.info("Selected engine has no rows after current cycle/alert filters. Showing full engine history.")
    engine_rows = predictions[predictions["unit_id"] == selected_engine].sort_values("cycle")



trend_fig = build_engine_trend(engine_rows, selected_engine, model_choice)
trend_fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#23272f",
    paper_bgcolor="#181a20",
    font_color="#f3f4f6",
    xaxis=dict(gridcolor="#353945"),
    yaxis=dict(gridcolor="#353945"),
)
st.plotly_chart(trend_fig, width="stretch", key='trend_fig')

st.markdown("<hr style='margin:2.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)
st.subheader("Data Export", divider="rainbow")
export_cols = [
    "dataset",
    "unit_id",
    "cycle",
    "actual_rul",
    pred_col,
    res_col,
    "alert_level",
]
export_df = filtered[export_cols].copy()

st.caption("Interactive filtered dataset preview")
preview_df = export_df.copy()
preview_df["alert_level"] = preview_df["alert_level"].map(
    {
        "CRITICAL": "🔴 CRITICAL",
        "WARNING": "🟠 WARNING",
        "HEALTHY": "🟢 HEALTHY",
    }
)

st.dataframe(preview_df, width="stretch", hide_index=True)

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered View CSV",
    data=csv_bytes,
    file_name="filtered_dashboard_view.csv",
    mime="text/csv",
)
