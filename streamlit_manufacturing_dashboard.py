"""
Streamlit Interactive Analytics & Real-Time Efficiency Classification Dashboard
Project: AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data
Dataset: Thales_Group_Manufacturing.csv (100,000 Telemetry Records | 50 Industrial Machines)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="6G Smart Factory Efficiency Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f2942;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #0f2942;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. DATA LOADING, PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading & engineering 6G smart factory telemetry...")
def load_and_engineer_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Combine Date and Timestamp into a chronological Datetime index
    time_col = "Timestamp" if "Timestamp" in df.columns else "Time"
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df[time_col].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["Date_Only"] = df["Datetime"].dt.date
    df["Hour"] = df["Datetime"].dt.hour

    # Feature Engineering (Section 4 Specification)
    # 1. Sensor Stability Indicator (0-100 scale combining normalized thermal & vibration load)
    temp_norm = (df["Temperature_C"] - 30.0) / 60.0
    vib_norm = (df["Vibration_Hz"] - 0.1) / 4.9
    df["Sensor_Stability_Index"] = 100.0 * (1.0 - (0.5 * temp_norm + 0.5 * vib_norm))
    df["Thermal_Vibration_Stress"] = df["Temperature_C"] * df["Vibration_Hz"]

    # 2. Energy Efficiency Ratio (Units produced per hour per kW consumed)
    df["Energy_Efficiency_Ratio"] = (
        df["Production_Speed_units_per_hr"] / df["Power_Consumption_kW"].clip(lower=0.1)
    )

    # 3. Error-to-Output & Total Fault-to-Output Ratios
    df["Error_to_Output_Ratio"] = (
        df["Error_Rate_%"] * 100.0
    ) / df["Production_Speed_units_per_hr"].clip(lower=1.0)
    df["Total_Fault_to_Output_Ratio"] = (
        (df["Error_Rate_%"] + df["Quality_Control_Defect_Rate_%"]) * 100.0
    ) / df["Production_Speed_units_per_hr"].clip(lower=1.0)

    # 4. Unified 6G Network Reliability Score (0-100 scale combining latency & packet loss)
    lat_norm = (df["Network_Latency_ms"] - 1.0) / 49.0
    ploss_norm = df["Packet_Loss_%"] / 5.0
    df["Network_Reliability_Score"] = 100.0 * (1.0 - (0.5 * lat_norm + 0.5 * ploss_norm))

    return df


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING & CACHING
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Sensor_Stability_Index",
    "Energy_Efficiency_Ratio",
    "Error_to_Output_Ratio",
    "Total_Fault_to_Output_Ratio",
    "Network_Reliability_Score",
    "Mode_Active",
    "Mode_Idle",
    "Mode_Maintenance",
]

CLASS_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_CLASS_MAP = {0: "Low", 1: "Medium", 2: "High"}
COLOR_MAP = {"High": "#059669", "Medium": "#d97706", "Low": "#dc2626"}


@st.cache_resource(show_spinner="Training Multi-Class Classification Models (LR, RF, GBDT)...")
def train_efficiency_models(df: pd.DataFrame):
    op_dummies = pd.get_dummies(df["Operation_Mode"], prefix="Mode", dtype=float)
    for col in ["Mode_Active", "Mode_Idle", "Mode_Maintenance"]:
        if col not in op_dummies.columns:
            op_dummies[col] = 0.0

    df_mod = pd.concat([df, op_dummies], axis=1)
    X = df_mod[FEATURE_COLS].values
    y = df_mod["Efficiency_Status"].map(CLASS_MAP).values

    # Stratified sample for rapid interactive training while preserving exact distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline Model: Multi-class Logistic Regression (Class-Balanced)
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    # 2. Advanced Model A: Random Forest Classifier (Class-Balanced)
    rf_model = RandomForestClassifier(
        n_estimators=120, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 3. Advanced Model B: Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    gb_model.fit(X_train, y_train)

    models = {
        "Gradient Boosting (XGBoost/GBDT)": (gb_model, False),
        "Random Forest (Class-Balanced)": (rf_model, False),
        "Logistic Regression (Baseline)": (lr_model, True),
    }

    metrics_summary = {}
    for name, (m, use_scaled) in models.items():
        X_eval = X_test_scaled if use_scaled else X_test
        preds = m.predict(X_eval)
        probs = m.predict_proba(X_eval)
        metrics_summary[name] = {
            "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
            "Balanced Accuracy (%)": round(balanced_accuracy_score(y_test, preds) * 100, 2),
            "Macro F1 (%)": round(f1_score(y_test, preds, average="macro") * 100, 2),
            "Weighted F1 (%)": round(f1_score(y_test, preds, average="weighted") * 100, 2),
            "Mean Confidence (%)": round(float(np.mean(np.max(probs, axis=1))) * 100, 2),
            "Confusion Matrix": confusion_matrix(y_test, preds),
        }

    feature_means = pd.Series(X_train.mean(axis=0), index=FEATURE_COLS)
    feature_stds = pd.Series(X_train.std(axis=0) + 1e-8, index=FEATURE_COLS)

    return models, scaler, metrics_summary, feature_means, feature_stds


# ---------------------------------------------------------------------------
# 4. SIDEBAR: INTERACTIVE USER CONTROLS (SECTION 6 SPECIFICATION)
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        '<div class="main-header">⚙️ AI-Based Manufacturing Efficiency Classification Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Real-Time Predictive Intelligence Using Industrial IoT Sensor, Production Quality, and 6G Network Telemetry</div>',
        unsafe_allow_html=True,
    )

    default_path = "Thales_Group_Manufacturing.csv"
    fallback_path = "/working_dir/c_943d6eb295431537/user_uploaded_files/ttl=63d/user_uploaded/Thales_Group_Manufacturing.csv"
    data_path = default_path if os.path.exists(default_path) else fallback_path

    with st.sidebar:
        st.header("🎛️ Operational Controls & Filters")
        uploaded_file = st.file_uploader("Upload Custom Telemetry CSV", type=["csv"])
        active_path = uploaded_file if uploaded_file is not None else data_path

    df = load_and_engineer_data(active_path)
    models, scaler, metrics_summary, feat_means, feat_stds = train_efficiency_models(df)

    with st.sidebar:
        # 1. Model Selector
        selected_model_name = st.selectbox(
            "Active Inference Model",
            options=list(models.keys()),
            index=0,
        )

        # 2. Operation Mode Selection Dropdown
        all_modes = sorted(df["Operation_Mode"].unique().tolist())
        selected_modes = st.multiselect(
            "Operation Mode Filter",
            options=all_modes,
            default=all_modes,
        )

        # 3. Machine Selector
        all_machines = sorted(df["Machine_ID"].unique().tolist())
        machine_filter_mode = st.radio(
            "Machine Scope", ["All Fleet (50 Machines)", "Select Specific Machine(s)"], index=0
        )
        if machine_filter_mode == "Select Specific Machine(s)":
            selected_machines = st.multiselect(
                "Select Machine_ID(s)", options=all_machines, default=all_machines[:5]
            )
        else:
            selected_machines = all_machines

        # 4. Time-Window Range Filter
        min_date = df["Date_Only"].min()
        max_date = df["Date_Only"].max()
        date_range = st.date_input(
            "Telemetry Date Window",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # 5. 6G Network Quality Filters
        st.subheader("📡 6G Network Quality Filters")
        max_latency = st.slider(
            "Max 6G Network Latency (ms)",
            min_value=float(df["Network_Latency_ms"].min()),
            max_value=float(df["Network_Latency_ms"].max()),
            value=float(df["Network_Latency_ms"].max()),
            step=1.0,
        )
        max_packet_loss = st.slider(
            "Max Packet Loss (%)",
            min_value=float(df["Packet_Loss_%"].min()),
            max_value=float(df["Packet_Loss_%"].max()),
            value=float(df["Packet_Loss_%"].max()),
            step=0.25,
        )

        # 6. Metric Sensitivity Adjustment Sliders (Alert Thresholds)
        st.subheader("⚡ Metric Sensitivity Sliders")
        conf_threshold = st.slider(
            "Minimum Confidence Alert Threshold (%)",
            min_value=50.0,
            max_value=99.0,
            value=85.0,
            step=1.0,
        )
        error_sensitivity = st.slider(
            "Operational Error Sensitivity Multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Scales Error_Rate_% during live sensitivity stress-testing.",
        )
        speed_sensitivity = st.slider(
            "Production Speed Sensitivity Multiplier",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.05,
            help="Scales Production_Speed_units_per_hr during live sensitivity stress-testing.",
        )

    # Apply filters to dataframe
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range[0], date_range[1]
    else:
        start_d = end_d = min_date

    filtered_df = df[
        (df["Operation_Mode"].isin(selected_modes))
        & (df["Machine_ID"].isin(selected_machines))
        & (df["Date_Only"] >= start_d)
        & (df["Date_Only"] <= end_d)
        & (df["Network_Latency_ms"] <= max_latency)
        & (df["Packet_Loss_%"] <= max_packet_loss)
    ].copy()

    if filtered_df.empty:
        st.warning("No telemetry records match the active filter constraints. Please widen the sidebar filters.")
        return

    # ---------------------------------------------------------------------------
    # 5. TOP KPI RIBBON (SECTION 5 DEFINED KPIS)
    # ---------------------------------------------------------------------------
    active_model, use_scaled = models[selected_model_name]
    active_metrics = metrics_summary[selected_model_name]

    high_share = (filtered_df["Efficiency_Status"] == "High").mean() * 100
    med_share = (filtered_df["Efficiency_Status"] == "Medium").mean() * 100
    low_share = (filtered_df["Efficiency_Status"] == "Low").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Filtered Telemetry Records", f"{len(filtered_df):,}", f"{len(selected_machines)} Machines")
    with k2:
        st.metric("High Efficiency Share", f"{high_share:.2f}%", f"{int((filtered_df['Efficiency_Status']=='High').sum()):,} states")
    with k3:
        st.metric("Medium Efficiency Share", f"{med_share:.2f}%", f"{int((filtered_df['Efficiency_Status']=='Medium').sum()):,} states")
    with k4:
        st.metric("Low Efficiency Share", f"{low_share:.2f}%", f"{int((filtered_df['Efficiency_Status']=='Low').sum()):,} states")
    with k5:
        st.metric(
            "Model Prediction Confidence",
            f"{active_metrics['Mean Confidence (%)']:.2f}%",
            f"Acc: {active_metrics['Accuracy (%)']:.2f}%",
        )

    # ---------------------------------------------------------------------------
    # 6. FOUR CORE DASHBOARD MODULES (SECTION 6 SPECIFICATION)
    # ---------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 1. Efficiency Prediction Dashboard",
            "🏭 2. Machine-Level Insights",
            "🔍 3. Explainability Panel (XAI)",
            "📡 4. Operational & 6G Monitoring View",
        ]
    )

    # === MODULE 1: EFFICIENCY PREDICTION DASHBOARD ===
    with tab1:
        st.subheader("Real-Time Efficiency Classification & Confidence Simulator")
        col_sim, col_out = st.columns([1.15, 1.0])

        with col_sim:
            st.markdown("##### Live Telemetry Input Parameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                in_mode = st.selectbox("Operation Mode", ["Active", "Idle", "Maintenance"], index=0)
                in_speed = st.number_input("Production Speed (units/hr)", 50.0, 500.0, 425.0, 5.0)
                in_error = st.number_input("Operational Error Rate (%)", 0.0, 15.0, 1.45, 0.1)
            with c2:
                in_defect = st.number_input("QC Defect Rate (%)", 0.0, 10.0, 3.20, 0.2)
                in_temp = st.number_input("Temperature (°C)", 30.0, 90.0, 58.5, 1.0)
                in_vib = st.number_input("Vibration (Hz)", 0.1, 5.0, 2.10, 0.1)
            with c3:
                in_power = st.number_input("Power Consumption (kW)", 1.5, 10.0, 5.20, 0.2)
                in_lat = st.number_input("6G Latency (ms)", 1.0, 50.0, 18.0, 1.0)
                in_ploss = st.number_input("Packet Loss (%)", 0.0, 5.0, 1.10, 0.1)
            in_pm = st.slider("Predictive Maintenance Readiness Score", 0.0, 1.0, 0.72, 0.02)

        # Apply sensitivity multipliers
        eff_speed = np.clip(in_speed * speed_sensitivity, 50.0, 500.0)
        eff_error = np.clip(in_error * error_sensitivity, 0.0, 15.0)

        # Engineer live features
        s_stab = 100.0 * (1.0 - (0.5 * ((in_temp - 30.0) / 60.0) + 0.5 * ((in_vib - 0.1) / 4.9)))
        e_eff = eff_speed / max(in_power, 0.1)
        err_out = (eff_error * 100.0) / max(eff_speed, 1.0)
        flt_out = ((eff_error + in_defect) * 100.0) / max(eff_speed, 1.0)
        net_rel = 100.0 * (1.0 - (0.5 * ((in_lat - 1.0) / 49.0) + 0.5 * (in_ploss / 5.0)))

        x_vec = pd.DataFrame(
            [
                {
                    "Temperature_C": in_temp,
                    "Vibration_Hz": in_vib,
                    "Power_Consumption_kW": in_power,
                    "Network_Latency_ms": in_lat,
                    "Packet_Loss_%": in_ploss,
                    "Quality_Control_Defect_Rate_%": in_defect,
                    "Production_Speed_units_per_hr": eff_speed,
                    "Predictive_Maintenance_Score": in_pm,
                    "Error_Rate_%": eff_error,
                    "Sensor_Stability_Index": s_stab,
                    "Energy_Efficiency_Ratio": e_eff,
                    "Error_to_Output_Ratio": err_out,
                    "Total_Fault_to_Output_Ratio": flt_out,
                    "Network_Reliability_Score": net_rel,
                    "Mode_Active": 1.0 if in_mode == "Active" else 0.0,
                    "Mode_Idle": 1.0 if in_mode == "Idle" else 0.0,
                    "Mode_Maintenance": 1.0 if in_mode == "Maintenance" else 0.0,
                }
            ]
        )[FEATURE_COLS]

        X_live = scaler.transform(x_vec.values) if use_scaled else x_vec.values
        live_probs = active_model.predict_proba(X_live)[0]
        pred_idx = int(np.argmax(live_probs))
        pred_label = INV_CLASS_MAP[pred_idx]
        pred_conf = float(live_probs[pred_idx]) * 100.0

        with col_out:
            st.markdown("##### Real-Time Inference & Confidence Output")
            badge_color = COLOR_MAP[pred_label]
            alert_status = (
                "CERTAINTY PASS"
                if pred_conf >= conf_threshold
                else f"BELOW ALERT THRESHOLD ({conf_threshold:.0f}%)"
            )
            st.markdown(
                f"""
                <div style="background:{badge_color}15; border:2px solid {badge_color}; border-radius:8px; padding:16px; margin-bottom:12px;">
                    <div style="font-size:0.85rem; font-weight:700; color:{badge_color};">PREDICTED EFFICIENCY CLASS ({selected_model_name})</div>
                    <div style="font-size:2.1rem; font-weight:900; color:{badge_color};">{pred_label.upper()} EFFICIENCY</div>
                    <div style="font-size:1.0rem; color:#1e293b;">Prediction Confidence: <b>{pred_conf:.2f}%</b> | Status: <b>{alert_status}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame(
                {
                    "Efficiency Class": ["Low", "Medium", "High"],
                    "Probability (%)": [live_probs[0] * 100, live_probs[1] * 100, live_probs[2] * 100],
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Probability (%)",
                y="Efficiency Class",
                orientation="h",
                color="Efficiency Class",
                color_discrete_map=COLOR_MAP,
                text_auto=".2f",
                range_x=[0, 105],
                height=220,
                title="Class Probability Distribution (%)",
            )
            fig_prob.update_layout(margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Benchmark Comparison & Confusion Matrix")
        b_col1, b_col2 = st.columns([1.2, 1.0])
        with b_col1:
            bench_rows = []
            for m_name, m_dict in metrics_summary.items():
                bench_rows.append(
                    {
                        "Model": m_name,
                        "Accuracy (%)": m_dict["Accuracy (%)"],
                        "Balanced Accuracy (%)": m_dict["Balanced Accuracy (%)"],
                        "Macro F1 (%)": m_dict["Macro F1 (%)"],
                        "Weighted F1 (%)": m_dict["Weighted F1 (%)"],
                        "Mean Confidence (%)": m_dict["Mean Confidence (%)"],
                    }
                )
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

        with b_col2:
            cm = active_metrics["Confusion Matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Low", "Medium", "High"],
                y=["Low", "Medium", "High"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix — {selected_model_name}",
                height=300,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # === MODULE 2: MACHINE-LEVEL INSIGHTS ===
    with tab2:
        st.subheader("Per-Machine Efficiency Profiling & Historical Classification Trends")
        m_grp = (
            filtered_df.groupby("Machine_ID")
            .agg(
                Total_Records=("Efficiency_Status", "count"),
                High_Pct=("Efficiency_Status", lambda s: (s == "High").mean() * 100),
                Medium_Pct=("Efficiency_Status", lambda s: (s == "Medium").mean() * 100),
                Low_Pct=("Efficiency_Status", lambda s: (s == "Low").mean() * 100),
                Mean_Speed=("Production_Speed_units_per_hr", "mean"),
                Mean_Error=("Error_Rate_%", "mean"),
                Mean_Energy_Eff=("Energy_Efficiency_Ratio", "mean"),
                Mean_Net_Rel=("Network_Reliability_Score", "mean"),
            )
            .reset_index()
        )
        m_grp["Machine_Label"] = "M-" + m_grp["Machine_ID"].astype(str)

        fig_fleet = px.bar(
            m_grp.sort_values("High_Pct", ascending=False),
            x="Machine_Label",
            y=["High_Pct", "Medium_Pct", "Low_Pct"],
            title="Efficiency Class Distribution Across Industrial Machines (%)",
            labels={"value": "Share of Telemetry (%)", "Machine_Label": "Machine Identifier"},
            color_discrete_map={"High_Pct": "#059669", "Medium_Pct": "#d97706", "Low_Pct": "#dc2626"},
            height=380,
        )
        st.plotly_chart(fig_fleet, use_container_width=True)

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            target_machine = st.selectbox("Inspect Individual Machine Historical Trajectory", all_machines, index=23)
            m_hist = filtered_df[filtered_df["Machine_ID"] == target_machine].copy()
            m_daily = (
                m_hist.groupby("Date_Only")
                .agg(
                    High_Med_Pct=("Efficiency_Status", lambda s: s.isin(["High", "Medium"]).mean() * 100),
                    Mean_Speed=("Production_Speed_units_per_hr", "mean"),
                    Mean_Error=("Error_Rate_%", "mean"),
                )
                .reset_index()
            )
            fig_m_trend = px.line(
                m_daily,
                x="Date_Only",
                y="High_Med_Pct",
                markers=True,
                title=f"Machine {target_machine} — Daily Optimal Efficiency Share (High + Medium %)",
                labels={"High_Med_Pct": "Optimal Efficiency (%)", "Date_Only": "Date"},
                height=320,
            )
            st.plotly_chart(fig_m_trend, use_container_width=True)

        with c_m2:
            st.markdown("##### Top 10 Fleet Units by High-Efficiency State Share")
            st.dataframe(
                m_grp.sort_values("High_Pct", ascending=False).head(10).round(2),
                use_container_width=True,
                hide_index=True,
            )

    # === MODULE 3: EXPLAINABILITY PANEL (XAI) ===
    with tab3:
        st.subheader("Global Feature Importance & Local Prediction Attribution")
        xai_c1, xai_c2 = st.columns(2)

        with xai_c1:
            if hasattr(active_model, "feature_importances_"):
                imp_vals = active_model.feature_importances_ * 100.0
            else:
                imp_vals = np.mean(np.abs(active_model.coef_), axis=0)
                imp_vals = (imp_vals / imp_vals.sum()) * 100.0

            imp_df = (
                pd.DataFrame({"Feature": FEATURE_COLS, "Importance (%)": imp_vals})
                .sort_values("Importance (%)", ascending=True)
                .tail(10)
            )
            fig_imp = px.bar(
                imp_df,
                x="Importance (%)",
                y="Feature",
                orientation="h",
                title=f"Top 10 Global Feature Drivers — {selected_model_name}",
                color="Importance (%)",
                color_continuous_scale="Tealgrn",
                text_auto=".2f",
                height=380,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with xai_c2:
            # Local Feature Contribution (Standardized Deviation Weighted by Feature Importance)
            z_scores = (x_vec.iloc[0] - feat_means) / feat_stds
            directional_sign = pd.Series(1.0, index=FEATURE_COLS)
            for neg_col in [
                "Error_Rate_%",
                "Quality_Control_Defect_Rate_%",
                "Error_to_Output_Ratio",
                "Total_Fault_to_Output_Ratio",
                "Network_Latency_ms",
                "Packet_Loss_%",
                "Power_Consumption_kW",
            ]:
                directional_sign[neg_col] = -1.0

            local_contrib = z_scores * directional_sign * (imp_vals + 0.5)
            local_df = pd.DataFrame(
                {
                    "Feature": list(FEATURE_COLS),
                    "Efficiency Impact Score": [float(v) for v in local_contrib.values],
                }
            )
            local_df["_abs"] = local_df["Efficiency Impact Score"].abs()
            local_df = (
                local_df
                .nlargest(10, "_abs")
                .sort_values("_abs", ascending=True)
                .drop(columns=["_abs"])
                .reset_index(drop=True)
            )
            local_df["Direction"] = np.where(
                local_df["Efficiency Impact Score"] >= 0, "Positive Driver (Boosts Efficiency)", "Negative Driver (Drops Efficiency)"
            )
            fig_local = px.bar(
                local_df,
                x="Efficiency Impact Score",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={
                    "Positive Driver (Boosts Efficiency)": "#059669",
                    "Negative Driver (Drops Efficiency)": "#dc2626",
                },
                title=f"Local Explainability Attribution for Current Simulated State ({pred_label})",
                height=380,
            )
            st.plotly_chart(fig_local, use_container_width=True)

    # === MODULE 4: OPERATIONAL & 6G MONITORING VIEW ===
    with tab4:
        st.subheader("Operating Mode Breakdown & 6G Network vs. Sensor Impact Comparison")
        op_c1, op_c2 = st.columns(2)

        with op_c1:
            mode_ct = (
                pd.crosstab(filtered_df["Operation_Mode"], filtered_df["Efficiency_Status"], normalize="index")
                * 100.0
            ).reset_index()
            for _ec in ["High", "Medium", "Low"]:
                if _ec not in mode_ct.columns:
                    mode_ct[_ec] = 0.0
            fig_mode = px.bar(
                mode_ct,
                x="Operation_Mode",
                y=["High", "Medium", "Low"],
                barmode="group",
                color_discrete_map=COLOR_MAP,
                title="Efficiency Status Breakdown Across Operation Modes (%)",
                labels={"value": "Share (%)", "Operation_Mode": "Operation Mode"},
                height=350,
            )
            st.plotly_chart(fig_mode, use_container_width=True)

        with op_c2:
            sample_vis = filtered_df.sample(n=min(3000, len(filtered_df)), random_state=42)
            fig_phase = px.scatter(
                sample_vis,
                x="Production_Speed_units_per_hr",
                y="Error_Rate_%",
                color="Efficiency_Status",
                color_discrete_map=COLOR_MAP,
                opacity=0.65,
                title="Operational Regime Phase Space: Production Speed vs. Error Rate",
                labels={
                    "Production_Speed_units_per_hr": "Production Speed (units/hr)",
                    "Error_Rate_%": "Operational Error Rate (%)",
                },
                height=350,
            )
            fig_phase.add_vline(x=200, line_dash="dash", line_color="#334155")
            fig_phase.add_vline(x=400, line_dash="dot", line_color="#0f172a")
            fig_phase.add_hline(y=5.0, line_dash="dash", line_color="#334155")
            fig_phase.add_hline(y=2.0, line_dash="dot", line_color="#0f172a")
            st.plotly_chart(fig_phase, use_container_width=True)

        st.markdown("##### 6G Network Telemetry vs. Physical Sensor Metrics Across Efficiency States")
        comp_table = (
            filtered_df.groupby("Efficiency_Status")[
                [
                    "Production_Speed_units_per_hr",
                    "Error_Rate_%",
                    "Quality_Control_Defect_Rate_%",
                    "Energy_Efficiency_Ratio",
                    "Temperature_C",
                    "Vibration_Hz",
                    "Network_Latency_ms",
                    "Packet_Loss_%",
                    "Network_Reliability_Score",
                    "Sensor_Stability_Index",
                ]
            ]
            .mean()
            .round(3)
            .reset_index()
        )
        st.dataframe(comp_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()