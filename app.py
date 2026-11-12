"""
Executive Data Analyzer Pro
A local-first Streamlit app for exploratory analysis and management reporting.

Run:
    pip install -r requirements.txt
    streamlit run executive_data_analyzer.py

Notes:
- Core analysis runs locally; data is not sent to an AI provider.
- Optional Gemini narrative generation sends aggregated statistics and a
  bounded sample of the data profile, not the full raw dataset.
"""

from __future__ import annotations

import io
import json
import math
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page setup and visual defaults
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Data Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.07);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .insight-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0;
    }
    .small-muted {font-size: 0.88rem; opacity: 0.75;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📊 Executive Data Analyzer Pro")
st.caption(
    "Local-first data analysis for faster diagnosis, clearer insights, "
    "decision-oriented visualizations, and management-ready reporting."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_number(value: Any, digits: int = 2) -> str:
    """Format numeric values without displaying NaN or infinity."""
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return "—"
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.{digits}f}B"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:,.{digits}f}M"
        if abs(value) >= 1_000:
            return f"{value:,.{digits}f}"
        return f"{value:,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def infer_date_columns(df: pd.DataFrame) -> list[str]:
    """Detect datetime columns and object columns that parse reliably as dates."""
    found = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(300)
