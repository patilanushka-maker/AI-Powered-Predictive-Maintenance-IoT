"""Interactive Streamlit Monitoring Dashboard for AI-Predictive-Maintenance-IoT.

This module provides a production-grade industrial IoT monitoring interface for tracking
fleet sensor telemetry, evaluating machine learning positive-class scores, and triaging
rule-based maintenance alerts (Phases 8-10).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import alert_system, dashboard_helpers, predict, simulator
from src.preprocessing import BASELINE_PREDICTORS
from src.train_model import BASELINE_MODEL_NAMES

# Page Configuration
st.set_page_config(
    page_title="AI Predictive Maintenance | Industrial IoT Fleet Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Unified Production Industrial Design System (CSS)
CUSTOM_CSS = """
<style>
    /* Global layout & typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1440px;
    }
    
    /* Header Container */
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-box {
        background: #eff6ff;
        color: #2563eb;
        width: 44px;
        height: 44px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        border: 1px solid #dbeafe;
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin: 2px 0 0 0;
    }
    .header-meta {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .status-indicator-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-dot-active {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #16a34a;
    }
    .source-chip {
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
    }
    .disclaimer-icon {
        color: #2563eb;
        font-size: 1.15rem;
        line-height: 1;
        margin-top: 1px;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 14px 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card-total { border-top: 3px solid #2563eb; }
    .kpi-card-normal { border-top: 3px solid #16a34a; }
    .kpi-card-warning { border-top: 3px solid #d97706; }
    .kpi-card-critical { border-top: 3px solid #dc2626; }
    .kpi-card-invalid { border-top: 3px solid #64748b; }
    .kpi-card-machines { border-top: 3px solid #4f46e5; }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-label {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .kpi-badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.03em;
    }
    .kpi-badge-total { background: #eff6ff; color: #1e40af; }
    .kpi-badge-normal { background: #f0fdf4; color: #15803d; }
    .kpi-badge-warning { background: #fffbeb; color: #b45309; }
    .kpi-badge-critical { background: #fef2f2; color: #b91c1c; }
    .kpi-badge-invalid { background: #f1f5f9; color: #475569; }
    .kpi-badge-machines { background: #eef2ff; color: #4338ca; }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-footer {
        font-size: 0.74rem;
        color: #94a3b8;
    }

    /* Machine Card */
    .machine-card {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .machine-card-normal { border-left: 4px solid #16a34a; }
    .machine-card-warning { border-left: 4px solid #d97706; }
    .machine-card-critical { border-left: 4px solid #dc2626; }
    .machine-card-invalid { border-left: 4px solid #64748b; }

    .machine-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .machine-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .type-pill {
        font-size: 0.72rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #475569;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }

    /* Status Badges */
    .badge-status {
        padding: 3px 9px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.74rem;
        letter-spacing: 0.04em;
        display: inline-block;
    }
    .badge-normal {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-warning {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .badge-critical {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
    .badge-invalid {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    /* Score Bar */
    .score-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 12px;
    }
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
    }
    .score-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .score-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        font-variant-numeric: tabular-nums;
    }
    .score-track {
        height: 5px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Sensor Grid */
    .sensor-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px;
    }
    .sensor-cell {
        display: flex;
        flex-direction: column;
    }
    .sensor-name {
        font-size: 0.68rem;
        color: #64748b;
        margin-bottom: 1px;
    }
    .sensor-val {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e293b;
        font-variant-numeric: tabular-nums;
    }

    /* Alert Console Card */
    .alert-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .alert-card-normal { border-left: 4px solid #16a34a; }
    .alert-card-warning { border-left: 4px solid #d97706; }
    .alert-card-critical { border-left: 4px solid #dc2626; }
    .alert-card-invalid { border-left: 4px solid #64748b; }

    .alert-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .alert-meta-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-id {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.9rem;
    }
    .alert-machine {
        font-size: 0.84rem;
        color: #334155;
        font-weight: 600;
    }
    .alert-timestamp {
        font-size: 0.75rem;
        color: #64748b;
    }
    .alert-body-section {
        margin-top: 6px;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .alert-reason-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        margin-bottom: 6px;
    }
    .alert-action-box {
        background: #eff6ff;
        border: 1px solid #dbeafe;
        color: #1e40af;
        font-weight: 500;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px 0;
    }
    .section-caption {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0 0 14px 0;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #475569;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
def init_session_state() -> None:
    """Initialize persistent session state variables safely."""
    if "events_df" not in st.session_state:
        try:
            initial_events = simulator.run_simulation(
                num_records=4,
                num_machines=2,
                scenario="mixed",
                seed=3,
                model="random_forest",
            )
            st.session_state["events_df"] = pd.DataFrame(initial_events)
            st.session_state["source_label"] = "Demonstration Mixed Simulation (4 records, 2 machines, Seed 3)"
        except Exception as exc:
            st.session_state["events_df"] = pd.DataFrame(columns=dashboard_helpers.EVENT_COLUMNS)
            st.session_state["source_label"] = f"Failed to initialize simulation: {exc}"

    if "warning_threshold" not in st.session_state:
        st.session_state["warning_threshold"] = 0.30
    if "critical_threshold" not in st.session_state:
        st.session_state["critical_threshold"] = 0.70


init_session_state()


# Sidebar Navigation & Controls
def render_sidebar() -> None:
    """Render sidebar controls for data source selection, simulation parameters, and uploads."""
    st.sidebar.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span style="font-size:1.3rem;">⚙️</span>
            <div>
                <h3 style="margin:0; font-size:1.1rem; color:#0f172a; font-weight:700;">Telemetry Controls</h3>
                <span style="font-size:0.75rem; color:#64748b;">Fleet Data Ingestion & Models</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_source = st.sidebar.radio(
        "Data Source Mode",
        options=["Run IoT Simulation", "Upload Telemetry CSV", "Load Benchmark Sample"],
        index=0,
    )

    st.sidebar.markdown("<hr style='margin:14px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Common Model Selection
    model_choices = {
        "Random Forest (Unweighted - Reference)": "random_forest",
        "Random Forest (Balanced)": "random_forest_balanced",
        "Logistic Regression (Balanced)": "logistic_regression_balanced",
        "Logistic Regression (Unweighted)": "logistic_regression",
    }
    selected_model_label = st.sidebar.selectbox(
        "Classifier Model Pipeline",
        options=list(model_choices.keys()),
        index=0,
    )
    selected_model_key = model_choices[selected_model_label]

    # Alert Thresholds
    st.sidebar.markdown("<div style='font-size:0.85rem; font-weight:600; color:#334155; margin-top:10px;'>Alert Threshold Configuration</div>", unsafe_allow_html=True)
    warning_thresh = st.sidebar.slider(
        "Warning Threshold",
        min_value=0.05,
        max_value=0.85,
        value=float(st.session_state["warning_threshold"]),
        step=0.05,
        help="Scores at or above this trigger a WARNING alert.",
    )
    critical_thresh = st.sidebar.slider(
        "Critical Threshold",
        min_value=0.10,
        max_value=0.95,
        value=float(st.session_state["critical_threshold"]),
        step=0.05,
        help="Scores at or above this trigger a CRITICAL alert.",
    )

    if critical_thresh <= warning_thresh:
        st.sidebar.error("Critical threshold must be strictly greater than warning threshold.")
        return

    st.session_state["warning_threshold"] = warning_thresh
    st.session_state["critical_threshold"] = critical_thresh

    alert_cfg = alert_system.AlertConfig(
        warning_threshold=warning_thresh,
        critical_threshold=critical_thresh,
    )

    st.sidebar.markdown("<hr style='margin:14px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # Mode 1: Run Simulation
    if data_source == "Run IoT Simulation":
        st.sidebar.markdown("<div style='font-size:0.85rem; font-weight:600; color:#334155;'>Simulation Generator</div>", unsafe_allow_html=True)
        scenario = st.sidebar.selectbox(
            "Operational Scenario",
            options=["mixed", "nominal", "degradation", "fault_injection"],
            index=0,
            help="Mixed: multi-machine variation (demonstrates NORMAL + WARNING states). Nominal: baseline wear. Degradation: progressive wear. Fault: validation errors.",
        )
        records_count = st.sidebar.slider(
            "Number of Records",
            min_value=4,
            max_value=100,
            value=4,
            step=2,
            help="Bounded upper limit avoids UI unresponsiveness.",
        )
        machines_count = st.sidebar.slider(
            "Simulated Machines",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )
        interval_val = st.sidebar.slider(
            "Interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
        )
        seed_val = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=3,
            step=1,
        )

        if st.sidebar.button("Run Telemetry Simulation", use_container_width=True, type="primary"):
            with st.spinner("Generating simulated telemetry and computing inference..."):
                try:
                    events = simulator.run_simulation(
                        num_records=records_count,
                        num_machines=machines_count,
                        interval_seconds=interval_val,
                        scenario=scenario,
                        seed=int(seed_val),
                        model=selected_model_key,
                        alert_config=alert_cfg,
                    )
                    st.session_state["events_df"] = pd.DataFrame(events)
                    st.session_state["source_label"] = (
                        f"Simulation: {scenario.capitalize()} ({records_count} records, {machines_count} machines, seed {seed_val})"
                    )
                    st.sidebar.success(f"Generated {len(events)} events successfully.")
                except Exception as exc:
                    st.sidebar.error(f"Simulation failed: {exc}")

    # Mode 2: Upload CSV
    elif data_source == "Upload Telemetry CSV":
        st.sidebar.markdown("<div style='font-size:0.85rem; font-weight:600; color:#334155;'>CSV File Ingestion</div>", unsafe_allow_html=True)
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload previously saved simulation events or raw telemetry containing the 6 approved baseline predictors.",
        )

        if uploaded_file is not None:
            if st.sidebar.button("Process Uploaded File", use_container_width=True, type="primary"):
                with st.spinner("Validating schema and parsing CSV..."):
                    schema_type, df_parsed, errors = dashboard_helpers.validate_and_parse_uploaded_csv(uploaded_file)
                    if schema_type == "invalid":
                        for err in errors:
                            st.sidebar.error(err)
                    elif schema_type == "simulation_events":
                        st.session_state["events_df"] = df_parsed
                        st.session_state["source_label"] = f"Uploaded Event Log: {uploaded_file.name} ({len(df_parsed)} events)"
                        st.sidebar.success(f"Loaded {len(df_parsed)} simulation events.")
                    elif schema_type == "raw_telemetry":
                        try:
                            converted_events = dashboard_helpers.convert_raw_telemetry_to_events(
                                df=df_parsed,
                                model=selected_model_key,
                                alert_config=alert_cfg,
                            )
                            st.session_state["events_df"] = pd.DataFrame(converted_events)
                            st.session_state["source_label"] = f"Processed Raw Telemetry: {uploaded_file.name} ({len(converted_events)} rows)"
                            st.sidebar.success(f"Processed {len(converted_events)} records with inference & alerts.")
                        except Exception as exc:
                            st.sidebar.error(f"Inference on uploaded telemetry failed: {exc}")

    # Mode 3: Load Benchmark Sample
    elif data_source == "Load Benchmark Sample":
        st.sidebar.markdown("<div style='font-size:0.85rem; font-weight:600; color:#334155;'>Benchmark Sample Data</div>", unsafe_allow_html=True)
        sample_choice = st.sidebar.selectbox(
            "Select Sample Dataset",
            options=[
                "Warning Demonstration (3 Normal, 1 Warning)",
                "General Benchmark (3 Normal, 1 Critical)",
            ],
            index=0,
        )
        if sample_choice == "Warning Demonstration (3 Normal, 1 Warning)":
            chosen_file = "data/sample_warning_records.csv"
            desc = "Loads 4 verified records (3 Normal, 1 Warning with score 0.3400)."
        else:
            chosen_file = "data/sample_inference_records.csv"
            desc = "Loads 4 illustrative records from the general benchmark (3 Normal, 1 Critical)."

        st.sidebar.info(desc)
        if st.sidebar.button("Load Sample Records", use_container_width=True, type="primary"):
            sample_path = Path(chosen_file).resolve()
            if not sample_path.is_file():
                st.sidebar.error(f"Sample file not found at: {sample_path}")
            else:
                try:
                    df_sample = pd.read_csv(sample_path)
                    converted = dashboard_helpers.convert_raw_telemetry_to_events(
                        df=df_sample,
                        model=selected_model_key,
                        alert_config=alert_cfg,
                    )
                    st.session_state["events_df"] = pd.DataFrame(converted)
                    st.session_state["source_label"] = f"{sample_choice} ({len(converted)} records)"
                    st.sidebar.success(f"Loaded {len(converted)} records.")
                except Exception as exc:
                    st.sidebar.error(f"Failed to load sample: {exc}")


render_sidebar()

# Active Events Dataset & Source Label
events_df: pd.DataFrame = st.session_state.get("events_df", pd.DataFrame())
source_label_str: str = st.session_state.get("source_label", "Active Session")

# Header Section
st.markdown(
    f"""
    <div class="header-bar">
        <div class="header-brand">
            <div class="brand-icon-box">⚙️</div>
            <div>
                <h1 class="header-title">AI-Powered Predictive Maintenance</h1>
                <p class="header-subtitle">IoT Fleet Monitoring &amp; Predictive Risk Analytics</p>
            </div>
        </div>
        <div class="header-meta">
            <div class="status-indicator-pill">
                <span class="status-dot-active"></span>
                <span>System Operational</span>
            </div>
            <div class="source-chip">
                Source: <b>{source_label_str}</b>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Research & Portfolio Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        <div class="disclaimer-icon">ℹ️</div>
        <div>
            <strong>Research &amp; Portfolio Demonstration:</strong> Operating on the synthetic UCI AI4I 2020 predictive maintenance
            benchmark and simulated IoT telemetry. Demonstrates leak-free ML classification, dynamic telemetry simulation, and rule-based
            alerting. Does <em>not</em> connect to live physical machinery, does <em>not</em> predict Remaining Useful Life (RUL) or exact failure timing,
            and is <em>not</em> a certified industrial safety shutdown system.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI Summary Computation
kpi = dashboard_helpers.compute_kpi_summary(events_df)
warning_thresh_val = float(st.session_state["warning_threshold"])
critical_thresh_val = float(st.session_state["critical_threshold"])

# KPI Section (6 Responsive Cards)
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-total">
            <div class="kpi-header">
                <span class="kpi-label">Total Events</span>
                <span class="kpi-badge kpi-badge-total">TELEMETRY</span>
            </div>
            <div class="kpi-value">{kpi['total_events']}</div>
            <div class="kpi-footer">Active telemetry records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-normal">
            <div class="kpi-header">
                <span class="kpi-label">Normal</span>
                <span class="kpi-badge kpi-badge-normal">NOMINAL</span>
            </div>
            <div class="kpi-value">{kpi['normal_count']}</div>
            <div class="kpi-footer">Score &lt; {warning_thresh_val:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-warning">
            <div class="kpi-header">
                <span class="kpi-label">Warning</span>
                <span class="kpi-badge kpi-badge-warning">INSPECT</span>
            </div>
            <div class="kpi-value">{kpi['warning_count']}</div>
            <div class="kpi-footer">Score &ge; {warning_thresh_val:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-critical">
            <div class="kpi-header">
                <span class="kpi-label">Critical</span>
                <span class="kpi-badge kpi-badge-critical">REVIEW</span>
            </div>
            <div class="kpi-value">{kpi['critical_count']}</div>
            <div class="kpi-footer">Score &ge; {critical_thresh_val:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col5:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-invalid">
            <div class="kpi-header">
                <span class="kpi-label">Invalid</span>
                <span class="kpi-badge kpi-badge-invalid">DATA ERROR</span>
            </div>
            <div class="kpi-value">{kpi['invalid_count']}</div>
            <div class="kpi-footer">Validation / schema faults</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col6:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-machines">
            <div class="kpi-header">
                <span class="kpi-label">Distinct Machines</span>
                <span class="kpi-badge kpi-badge-machines">FLEET</span>
            </div>
            <div class="kpi-value">{kpi['distinct_machines']}</div>
            <div class="kpi-footer">Monitored units</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_fleet, tab_charts, tab_alerts, tab_export = st.tabs(
    [
        "Fleet Machine Monitoring",
        "Telemetry & Risk Visualizations",
        "Maintenance Alert Console",
        "Data Export & Architecture Specs",
    ]
)

# TAB 1: Fleet Machine Monitoring
with tab_fleet:
    st.markdown("<div class='section-title'>Fleet Machine Status Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Latest reported sensor telemetry and rule-based alert classifications per machine unit.</div>", unsafe_allow_html=True)

    if events_df.empty:
        st.warning("No telemetry events available in the active session. Launch a simulation from the sidebar control panel.")
    else:
        latest_df = dashboard_helpers.get_latest_machine_telemetry(events_df)
        
        machine_cols = st.columns(min(len(latest_df), 4) or 1)
        for idx, (_, m_row) in enumerate(latest_df.iterrows()):
            col_target = machine_cols[idx % len(machine_cols)]
            m_id = str(m_row.get("machine_id", f"Machine-{idx}"))
            m_lvl = str(m_row.get("alert_level", "UNKNOWN"))
            m_type = str(m_row.get("Type", "N/A"))
            m_score = m_row.get("positive_class_score")
            score_txt = f"{m_score:.4f}" if m_score is not None else "N/A"
            
            # Score fill width and color
            if m_score is not None:
                fill_pct = min(max(m_score * 100, 0), 100)
                if m_score >= critical_thresh_val:
                    fill_color = "#dc2626"
                elif m_score >= warning_thresh_val:
                    fill_color = "#d97706"
                else:
                    fill_color = "#16a34a"
            else:
                fill_pct = 0
                fill_color = "#94a3b8"

            m_wear = float(m_row.get("Tool wear [min]", 0.0) or 0.0)
            m_torque = float(m_row.get("Torque [Nm]", 0.0) or 0.0)
            m_speed = float(m_row.get("Rotational speed [rpm]", 0.0) or 0.0)
            m_air_temp = float(m_row.get("Air temperature [K]", 0.0) or 0.0)
            m_proc_temp = float(m_row.get("Process temperature [K]", 0.0) or 0.0)

            card_border_cls = {
                "NORMAL": "machine-card-normal",
                "WARNING": "machine-card-warning",
                "CRITICAL": "machine-card-critical",
                "INVALID": "machine-card-invalid",
            }.get(m_lvl, "machine-card-invalid")

            badge_cls = {
                "NORMAL": "badge-normal",
                "WARNING": "badge-warning",
                "CRITICAL": "badge-critical",
                "INVALID": "badge-invalid",
            }.get(m_lvl, "badge-invalid")

            with col_target:
                st.markdown(
                    f"""
                    <div class="machine-card {card_border_cls}">
                        <div class="machine-card-top">
                            <div class="machine-title">
                                <span>{m_id}</span>
                                <span class="type-pill">Type {m_type}</span>
                            </div>
                            <span class="badge-status {badge_cls}">{m_lvl}</span>
                        </div>
                        <div class="score-container">
                            <div class="score-row">
                                <span class="score-label">Positive-Class Score</span>
                                <span class="score-value">{score_txt}</span>
                            </div>
                            <div class="score-track">
                                <div class="score-fill" style="width:{fill_pct:.1f}%; background-color:{fill_color};"></div>
                            </div>
                        </div>
                        <div class="sensor-grid">
                            <div class="sensor-cell">
                                <span class="sensor-name">Speed</span>
                                <span class="sensor-val">{m_speed:.0f} rpm</span>
                            </div>
                            <div class="sensor-cell">
                                <span class="sensor-name">Torque</span>
                                <span class="sensor-val">{m_torque:.1f} Nm</span>
                            </div>
                            <div class="sensor-cell">
                                <span class="sensor-name">Tool Wear</span>
                                <span class="sensor-val">{m_wear:.1f} min</span>
                            </div>
                            <div class="sensor-cell">
                                <span class="sensor-name">Air Temp</span>
                                <span class="sensor-val">{m_air_temp:.1f} K</span>
                            </div>
                            <div class="sensor-cell">
                                <span class="sensor-name">Proc Temp</span>
                                <span class="sensor-val">{m_proc_temp:.1f} K</span>
                            </div>
                            <div class="sensor-cell">
                                <span class="sensor-name">Delta T</span>
                                <span class="sensor-val">{(m_proc_temp - m_air_temp):.1f} K</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Filterable Fleet Telemetry Log</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-caption'>Interactive tabular view of processed events with multi-criteria filtering.</div>", unsafe_allow_html=True)

        filter_col1, filter_col2 = st.columns(2)
        all_machines = sorted(events_df["machine_id"].dropna().unique().tolist()) if "machine_id" in events_df.columns else []
        with filter_col1:
            sel_machines = st.multiselect("Filter by Machine ID", options=all_machines, default=all_machines)
        with filter_col2:
            all_levels = [alert_system.ALERT_NORMAL, alert_system.ALERT_WARNING, alert_system.ALERT_CRITICAL, alert_system.ALERT_INVALID]
            sel_levels = st.multiselect("Filter by Alert Severity", options=all_levels, default=all_levels)

        filtered_table_df = dashboard_helpers.filter_events(events_df, machines=sel_machines, severities=sel_levels)

        display_cols = [
            "event_id",
            "timestamp",
            "machine_id",
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "positive_class_score",
            "predicted_label",
            "alert_level",
            "status",
        ]
        present_cols = [c for c in display_cols if c in filtered_table_df.columns]
        st.dataframe(filtered_table_df[present_cols], use_container_width=True, hide_index=True)


# TAB 2: Telemetry & Risk Visualizations
with tab_charts:
    st.markdown("<div class='section-title'>Sensor Telemetry &amp; Model Output Risk Trajectories</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Interactive time-series charts sorted chronologically by simulated timestamp.</div>", unsafe_allow_html=True)

    if events_df.empty:
        st.warning("No events available to plot. Run a simulation from the sidebar control panel.")
    else:
        machines_list = sorted(events_df["machine_id"].dropna().unique().tolist()) if "machine_id" in events_df.columns else []
        if not machines_list:
            st.error("Event dataset lacks machine identifiers.")
        else:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                selected_chart_machine = st.selectbox("Select Machine Unit", options=machines_list, index=0)
            with chart_col2:
                available_channels = [
                    "positive_class_score",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "Air temperature [K]",
                    "Process temperature [K]",
                ]
                channel_labels = {
                    "positive_class_score": "Positive-Class Score (Model Output)",
                    "Rotational speed [rpm]": "Rotational Speed [rpm]",
                    "Torque [Nm]": "Torque [Nm]",
                    "Tool wear [min]": "Tool Wear [min]",
                    "Air temperature [K]": "Air Temperature [K]",
                    "Process temperature [K]": "Process Temperature [K]",
                }
                selected_channel = st.selectbox(
                    "Select Telemetry Channel",
                    options=available_channels,
                    format_func=lambda c: channel_labels.get(c, c),
                    index=0,
                )

            try:
                ts_ready_df = dashboard_helpers.prepare_time_series_data(events_df, machine_id=selected_chart_machine)
                fig = dashboard_helpers.create_time_series_figure(
                    df=ts_ready_df,
                    machine_id=selected_chart_machine,
                    channel=selected_channel,
                    warning_thresh=float(st.session_state["warning_threshold"]),
                    critical_thresh=float(st.session_state["critical_threshold"]),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as err:
                st.error(f"Unable to render chart: {err}")


# TAB 3: Maintenance Alert Console
with tab_alerts:
    st.markdown("<div class='section-title'>Maintenance Alert &amp; Mitigation Console</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Rule-based operational triage recommendations categorized by severity level.</div>", unsafe_allow_html=True)

    if events_df.empty:
        st.warning("No events available to evaluate.")
    else:
        alert_filter = st.selectbox(
            "Filter Console by Alert Severity",
            options=["ALL", alert_system.ALERT_NORMAL, alert_system.ALERT_WARNING, alert_system.ALERT_CRITICAL, alert_system.ALERT_INVALID],
            index=0,
        )

        target_severities = None if alert_filter == "ALL" else [alert_filter]
        alerts_display_df = dashboard_helpers.filter_events(events_df, severities=target_severities)

        st.markdown(
            f"""
            <div style="margin-bottom:12px; font-size:0.85rem; color:#475569;">
                Displaying <b>{len(alerts_display_df)}</b> event(s) matching filter: <span class="type-pill">{alert_filter}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for _, a_row in alerts_display_df.iterrows():
            lvl = str(a_row.get("alert_level", "UNKNOWN"))
            score = a_row.get("positive_class_score")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            reason = a_row.get("reason", "No reason recorded.")
            action = a_row.get("recommended_action", "No action specified.")
            m_id = a_row.get("machine_id", "Unknown")
            evt_id = a_row.get("event_id", "N/A")
            ts = a_row.get("timestamp", "N/A")

            card_lvl_cls = {
                "NORMAL": "alert-card-normal",
                "WARNING": "alert-card-warning",
                "CRITICAL": "alert-card-critical",
                "INVALID": "alert-card-invalid",
            }.get(lvl, "alert-card-invalid")

            badge_cls = {
                "NORMAL": "badge-normal",
                "WARNING": "badge-warning",
                "CRITICAL": "badge-critical",
                "INVALID": "badge-invalid",
            }.get(lvl, "badge-invalid")

            is_urgent = (lvl in ["CRITICAL", "INVALID"])

            with st.expander(f"[{lvl}] {evt_id} | {m_id} | Score: {score_str} | {ts}", expanded=is_urgent):
                st.markdown(
                    f"""
                    <div class="alert-card {card_lvl_cls}">
                        <div class="alert-card-header">
                            <div class="alert-meta-left">
                                <span class="badge-status {badge_cls}">{lvl}</span>
                                <span class="alert-id">{evt_id}</span>
                                <span class="alert-machine">{m_id}</span>
                            </div>
                            <span class="alert-timestamp">UTC: {ts}</span>
                        </div>
                        <div class="alert-body-section alert-reason-box">
                            <strong>Operational Meaning / Reason:</strong><br>{reason}
                        </div>
                        <div class="alert-body-section alert-action-box">
                            <strong>Recommended Action:</strong><br>{action}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Status: `{a_row.get('status', 'unknown')}` | Model: `{a_row.get('model_name', 'N/A')}`")


# TAB 4: Data Export & Architecture Specs
with tab_export:
    st.markdown("<div class='section-title'>Data Export &amp; Architecture Specifications</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Download active simulation logs and review model benchmark performance.</div>", unsafe_allow_html=True)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown(
            """
            <div class="machine-card" style="border-left: 4px solid #2563eb; margin-bottom: 16px;">
                <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">Active Simulation Export</h4>
                <p style="margin:0 0 12px 0; font-size:0.82rem; color:#64748b;">
                    Export all currently loaded telemetry event records with model scores, alert levels, and sensor channels.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not events_df.empty:
            csv_data = events_df.to_csv(index=False)
            st.download_button(
                label="Download Events as CSV",
                data=csv_data,
                file_name="simulation_events.csv",
                mime="text/csv",
                use_container_width=True,
            )
            json_data = events_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download Events as JSON",
                data=json_data,
                file_name="simulation_events.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("No active events to export.")

    with col_exp2:
        st.markdown(
            """
            <div class="machine-card" style="border-left: 4px solid #4f46e5; margin-bottom: 16px;">
                <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:1rem;">Inference Pipeline Architecture</h4>
                <ul style="margin:0; padding-left:18px; font-size:0.82rem; color:#334155; line-height:1.55;">
                    <li><b>Provisional Reference Model</b>: <code>baseline_random_forest.joblib</code></li>
                    <li><b>Permitted Predictors (6)</b>: <code>Type</code>, <code>Air temperature [K]</code>, <code>Process temperature [K]</code>, <code>Rotational speed [rpm]</code>, <code>Torque [Nm]</code>, <code>Tool wear [min]</code></li>
                    <li><b>Strictly Excluded Leakage Columns</b>: <code>UDI</code>, <code>Product ID</code>, <code>TWF</code>, <code>HDF</code>, <code>PWF</code>, <code>OSF</code>, <code>RNF</code>, <code>Machine failure</code></li>
                    <li><b>Zero Data Leakage</b>: Transformers fitted exclusively on <code>X_train</code> (8,000 samples).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:18px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Held-Out Test Split Performance Benchmarks (N = 2,000, Threshold = 0.5)</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Empirical performance evaluated on untouched held-out test data without synthetic oversampling.</div>", unsafe_allow_html=True)

    benchmark_table = [
        {"Model Configuration": "Random Forest (Unweighted)", "Precision": 0.9000, "Recall": 0.5294, "F1-Score": 0.6667, "ROC-AUC": 0.9653, "Average Precision (AP)": 0.7892},
        {"Model Configuration": "Random Forest (Balanced)", "Precision": 0.9677, "Recall": 0.4412, "F1-Score": 0.6061, "ROC-AUC": 0.9635, "Average Precision (AP)": 0.7642},
        {"Model Configuration": "Logistic Regression (Balanced)", "Precision": 0.1418, "Recall": 0.8235, "F1-Score": 0.2419, "ROC-AUC": 0.9070, "Average Precision (AP)": 0.3818},
        {"Model Configuration": "Logistic Regression (Unweighted)", "Precision": 0.6364, "Recall": 0.1029, "F1-Score": 0.1772, "ROC-AUC": 0.8994, "Average Precision (AP)": 0.4205},
    ]
    st.table(pd.DataFrame(benchmark_table))
