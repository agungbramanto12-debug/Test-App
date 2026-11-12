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
        if len(sample) < 3:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        if parsed.notna().mean() >= 0.85:
            found.append(col)
    return list(dict.fromkeys(found))


def coerce_date_column(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", dayfirst=True)


def get_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    dates = infer_date_columns(df)
    return numeric, categorical, dates


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        rows.append({
            "Kolom": str(col),
            "Tipe Data": str(df[col].dtype),
            "Missing": missing,
            "Missing (%)": round(missing / total * 100, 2),
            "Unique Values": unique,
            "Unique (%)": round(unique / total * 100, 2),
        })
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    result = df[numeric_cols].describe().T
    result["median"] = df[numeric_cols].median()
    result["sum"] = df[numeric_cols].sum(min_count=1)
    result["missing"] = df[numeric_cols].isna().sum()
    result["missing_pct"] = df[numeric_cols].isna().mean() * 100
    result["zero_count"] = (df[numeric_cols] == 0).sum()
    return result.rename_axis("Metrik").reset_index()


def calculate_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append({
            "Metrik": col,
            "Q1": q1,
            "Median": series.median(),
            "Q3": q3,
            "Batas Bawah IQR": lower,
            "Batas Atas IQR": upper,
            "Outlier": count,
            "Outlier (%)": count / max(len(series), 1) * 100,
            "Valid N": len(series),
        })
    return pd.DataFrame(rows)


def top_category_insight(df: pd.DataFrame, cat_col: str, metric_col: str, agg: str):
    work = df[[cat_col, metric_col]].dropna()
    if work.empty:
        return None, pd.DataFrame()
    if agg == "Sum":
        grouped = work.groupby(cat_col, dropna=False)[metric_col].sum()
    elif agg == "Mean":
        grouped = work.groupby(cat_col, dropna=False)[metric_col].mean()
    else:
        grouped = work.groupby(cat_col, dropna=False)[metric_col].count()
    grouped = grouped.sort_values(ascending=False)
    return (grouped.index[0], grouped.iloc[0]), grouped.rename("value").reset_index()


def build_rule_based_summary(df: pd.DataFrame, raw_df: pd.DataFrame,
                             numeric_cols: list[str], categorical_cols: list[str],
                             date_cols: list[str], selected_metric: str | None):
    lines = []
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    missing_pct = missing_cells / max(n_rows * n_cols, 1) * 100
    duplicate_rows = int(df.duplicated().sum())
    filtered_share = n_rows / max(len(raw_df), 1) * 100

    lines.append(
        f"Dataset yang dianalisis berisi **{n_rows:,} baris dan {n_cols:,} kolom** "
        f"({filtered_share:.1f}% dari data awal setelah filter)."
    )
    lines.append(
        f"Kualitas data: **{missing_cells:,} sel kosong ({missing_pct:.2f}%)** "
        f"dan **{duplicate_rows:,} baris duplikat**."
    )

    if n_rows == 0:
        lines.append("Tidak ada baris yang tersisa setelah filter. Ubah filter sebelum menarik kesimpulan.")
        return lines

    if selected_metric and selected_metric in numeric_cols:
        s = pd.to_numeric(df[selected_metric], errors="coerce").dropna()
        if len(s):
            lines.append(
                f"Metrik utama **{selected_metric}** memiliki total **{safe_number(s.sum())}**, "
                f"rata-rata **{safe_number(s.mean())}**, median **{safe_number(s.median())}**, "
                f"dan rentang **{safe_number(s.min())}–{safe_number(s.max())}** "
                f"(berdasarkan {len(s):,} nilai valid)."
            )
            if s.mean() != 0:
                cv = s.std() / abs(s.mean()) * 100 if len(s) > 1 else 0
                lines.append(
                    f"Variasi relatif metrik (coefficient of variation) sekitar **{cv:.1f}%**. "
                    "Nilai ini membantu membaca konsistensi, tetapi tidak menjelaskan penyebab variasi."
                )

    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        metric_col = selected_metric if selected_metric in numeric_cols else numeric_cols[0]
        top, grouped = top_category_insight(df, cat_col, metric_col, "Sum")
        if top is not None:
            total = grouped["value"].sum()
            share = top[1] / total * 100 if total else np.nan
            lines.append(
                f"Pada **{cat_col}**, kontributor terbesar terhadap total **{metric_col}** "
                f"adalah **{top[0]}** dengan nilai **{safe_number(top[1])}**"
                + (f" (sekitar {share:.1f}% dari total kategori)." if pd.notna(share) else ".")
            )

    if date_cols:
        date_col = date_cols[0]
        dates = coerce_date_column(df, date_col).dropna()
        if len(dates):
            lines.append(
                f"Rentang waktu pada **{date_col}** adalah **{dates.min().date()} hingga "
                f"{dates.max().date()}** ({len(dates):,} tanggal valid)."
            )

    return lines


def make_markdown_report(df: pd.DataFrame, summary_lines: list[str],
                         quality_df: pd.DataFrame, outlier_df: pd.DataFrame,
                         numeric_cols: list[str], title: str) -> str:
    parts = [
        f"# {title}",
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        "## Executive Summary",
        *[f"- {line.replace('**', '')}" for line in summary_lines],
        "## Data Quality",
        quality_df.to_markdown(index=False) if not quality_df.empty else "Tidak tersedia.",
    ]
    if numeric_cols:
        profile = numeric_profile(df, numeric_cols)
        parts += ["## Numeric Profile", profile.round(2).to_markdown(index=False)]
    if not outlier_df.empty:
        parts += ["## Outlier Screening (IQR)", outlier_df.round(2).to_markdown(index=False)]
    parts += [
        "## Methodology & Limitations",
        "- Analisis deskriptif menggambarkan data yang tersedia; bukan bukti sebab-akibat.",
        "- Outlier menggunakan aturan 1.5 × IQR dan perlu ditinjau sesuai konteks bisnis.",
        "- Korelasi Pearson mengukur hubungan linear dan tidak membuktikan kausalitas.",
        "- Hasil dipengaruhi filter, missing values, definisi metrik, dan kualitas input.",
    ]
    return "\n\n".join(parts)


def call_gemini(api_key: str, prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Optional API call. Imported only when user explicitly invokes the feature."""
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 2200},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Upload and parsing
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Unggah dataset CSV atau Excel",
    type=["csv", "xlsx", "xls"],
    help="File diproses di sesi aplikasi. Untuk Excel, pilih sheet yang ingin dianalisis.",
)

if uploaded_file is None:
    st.info("Unggah file CSV atau Excel untuk memulai analisis.")
    st.markdown(
        """
        **Modul yang tersedia**
        - Data profiling, statistik deskriptif, dan audit kualitas data.
        - Visualisasi distribusi, perbandingan kategori, tren waktu, dan hubungan metrik.
        - Korelasi, screening outlier, serta segmentasi.
        - Executive summary berbasis data dan laporan yang dapat diunduh.
        """
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            raw_df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding="latin-1")
        sheet_name = None
    else:
        excel = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Pilih sheet Excel", excel.sheet_names)
        raw_df = pd.read_excel(excel, sheet_name=sheet_name)
except Exception as exc:
    st.error(f"File tidak dapat dibaca: {exc}")
    st.stop()

if raw_df.empty or len(raw_df.columns) == 0:
    st.error("Dataset kosong atau tidak memiliki kolom.")
    st.stop()

# Remove completely empty rows/columns only after preserving original counts for transparency.
empty_rows = int(raw_df.isna().all(axis=1).sum())
empty_cols = int(raw_df.isna().all(axis=0).sum())
raw_df = raw_df.dropna(how="all").dropna(axis=1, how="all")
raw_df.columns = [str(c).strip() if str(c).strip() else f"Unnamed_{i+1}" for i, c in enumerate(raw_df.columns)]
raw_df = raw_df.loc[:, ~pd.Index(raw_df.columns).duplicated()]

numeric_cols, categorical_cols, date_cols = get_column_types(raw_df)

# ---------------------------------------------------------------------------
# Sidebar controls and filters
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Analysis Controls")
st.sidebar.caption(f"File: {uploaded_file.name}")
st.sidebar.caption(f"Data awal: {len(raw_df):,} rows × {len(raw_df.columns):,} columns")

df = raw_df.copy()
with st.sidebar.expander("Filter dataset", expanded=True):
    filterable = categorical_cols.copy()
    if filterable:
        filter_col = st.selectbox("Filter berdasarkan kategori", ["Tidak ada"] + filterable)
        if filter_col != "Tidak ada":
            values = sorted(df[filter_col].dropna().astype(str).unique().tolist())
            selected_values = st.multiselect(
                f"Nilai {filter_col}",
                values,
                default=values,
            )
            df = df[df[filter_col].astype(str).isin(selected_values)]
    else:
        st.caption("Tidak ditemukan kolom kategori untuk filter.")

    if date_cols:
        date_filter_col = st.selectbox("Filter rentang tanggal (opsional)", ["Tidak ada"] + date_cols)
        if date_filter_col != "Tidak ada":
            date_values = coerce_date_column(df, date_filter_col)
            valid_dates = date_values.dropna()
            if len(valid_dates):
                min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
                selected_range = st.date_input(
                    "Rentang tanggal",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    start_date, end_date = selected_range
                    df = df[
                        (coerce_date_column(df, date_filter_col).dt.date >= start_date)
                        & (coerce_date_column(df, date_filter_col).dt.date <= end_date)
                    ]

if df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter. Sesuaikan filter di sidebar.")
    st.stop()

numeric_cols, categorical_cols, date_cols = get_column_types(df)
default_metric = numeric_cols[0] if numeric_cols else None
selected_metric = st.sidebar.selectbox(
    "Metrik utama untuk insight",
    numeric_cols,
    index=0 if numeric_cols else None,
    help="Digunakan untuk ringkasan kategori dan beberapa rekomendasi otomatis.",
) if numeric_cols else None

# ---------------------------------------------------------------------------
# Global KPIs
# ---------------------------------------------------------------------------
missing_cells = int(df.isna().sum().sum())
missing_pct = missing_cells / max(df.size, 1) * 100
duplicate_rows = int(df.duplicated().sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Records", f"{len(df):,}", delta=f"{len(df)-len(raw_df):+,} vs data awal")
k2.metric("Total Variables", f"{df.shape[1]:,}")
k3.metric("Missing Cells", f"{missing_cells:,}", delta=pct(missing_pct), delta_color="inverse")
k4.metric("Duplicate Rows", f"{duplicate_rows:,}")
k5.metric("Numeric Metrics", f"{len(numeric_cols):,}")

tabs = st.tabs([
    "Overview & Data Quality",
    "Visual Analytics",
    "Correlation & Outliers",
    "Segmentation",
    "Executive Summary",
])


# ---------------------------------------------------------------------------
# TAB 1: Overview and data quality
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Overview")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Data preview**")
        st.dataframe(df.head(15), use_container_width=True, height=380)
    with right:
        st.markdown("**Column inventory**")
        inventory = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Role": [
                "Numerik" if c in numeric_cols else "Tanggal" if c in date_cols else "Kategori / Teks"
                for c in df.columns
            ],
        })
        st.dataframe(inventory, use_container_width=True, height=380)

    st.divider()
    st.subheader("Data Quality Assessment")
    quality = data_quality_report(df)
    qa1, qa2, qa3 = st.columns(3)
    qa1.metric("Rows completely empty removed", f"{empty_rows:,}")
    qa2.metric("Columns completely empty removed", f"{empty_cols:,}")
    qa3.metric("Duplicate rate", pct(duplicate_rows / max(len(df), 1) * 100))
    st.dataframe(
        quality.style.format({"Missing (%)": "{:.2f}%", "Unique (%)": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    if missing_cells:
        missing_plot = quality[quality["Missing"] > 0].sort_values("Missing", ascending=False)
        fig_missing = px.bar(
            missing_plot.head(20), x="Missing", y="Kolom", orientation="h",
            title="Kolom dengan Missing Values Terbanyak",
            text="Missing",
        )
        fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_missing, use_container_width=True)

    if numeric_cols:
        st.divider()
        st.subheader("Descriptive Statistics")
        stats = numeric_profile(df, numeric_cols)
        st.dataframe(
            stats.style.format({
                c: "{:,.2f}" for c in stats.select_dtypes(include="number").columns
            }),
            use_container_width=True,
            hide_index=True,
        )
    if categorical_cols:
        st.divider()
        st.subheader("Categorical Profile")
        cat_rows = []
        for col in categorical_cols:
            mode = df[col].mode(dropna=True)
            cat_rows.append({
                "Kolom": col,
                "Unique": df[col].nunique(dropna=True),
                "Most Frequent": str(mode.iloc[0]) if len(mode) else "—",
                "Frequency": int((df[col] == mode.iloc[0]).sum()) if len(mode) else 0,
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: Visual analytics
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Visual Analytics")
    chart_options = [
        "Category Comparison",
        "Metric Distribution",
        "Metric Relationship",
        "Time Trend",
        "Top / Bottom Ranking",
    ]
    chart_type = st.selectbox("Pilih analisis visual", chart_options)

    if chart_type == "Category Comparison":
        if categorical_cols and numeric_cols:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("Category", categorical_cols)
            y_col = c2.selectbox("Metric", numeric_cols, index=numeric_cols.index(selected_metric) if selected_metric in numeric_cols else 0)
            agg = c3.selectbox("Aggregation", ["Sum", "Mean", "Count"], key="visual_category_aggregation")
            top_n = st.slider("Jumlah kategori yang ditampilkan", 5, 30, 15)
            work = df[[x_col, y_col]].dropna(subset=[x_col])
            if agg == "Sum":
                grouped = work.groupby(x_col, as_index=False)[y_col].sum()
                y_label = f"Total {y_col}"
            elif agg == "Mean":
                grouped = work.groupby(x_col, as_index=False)[y_col].mean()
                y_label = f"Average {y_col}"
            else:
                grouped = work.groupby(x_col, as_index=False)[y_col].count()
                y_label = "Record Count"
            grouped = grouped.sort_values(y_col, ascending=False).head(top_n)
            fig = px.bar(
                grouped, x=x_col, y=y_col, color=y_col, text_auto=".3s",
                title=f"{y_label} by {x_col}",
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_label, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(grouped, use_container_width=True, hide_index=True)
        else:
            st.info("Visual perbandingan membutuhkan minimal satu kolom kategori dan satu kolom numerik.")

    elif chart_type == "Metric Distribution":
        if numeric_cols:
            metric = st.selectbox("Pilih metrik", numeric_cols)
            bins = st.slider("Jumlah bins", 5, 80, 30)
            fig = px.histogram(
                df, x=metric, nbins=bins, marginal="box",
                title=f"Distribution & Spread — {metric}",
                opacity=0.85,
            )
            fig.update_layout(yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            s = df[metric].dropna()
            if len(s):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Mean", safe_number(s.mean()))
                d2.metric("Median", safe_number(s.median()))
                d3.metric("Std. Deviation", safe_number(s.std()))
                d4.metric("Skewness", safe_number(s.skew()))
        else:
            st.info("Tidak ditemukan kolom numerik.")

    elif chart_type == "Metric Relationship":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X axis", numeric_cols, index=0)
            y_col = st.selectbox("Y axis", numeric_cols, index=1)
            color_col = st.selectbox("Color by (optional)", ["Tidak ada"] + categorical_cols)
            size_col = st.selectbox("Bubble size (optional)", ["Tidak ada"] + numeric_cols)
            plot_df = df[[x_col, y_col] + ([color_col] if color_col != "Tidak ada" else []) + ([size_col] if size_col != "Tidak ada" else [])].dropna(subset=[x_col, y_col])
            fig = px.scatter(
                plot_df, x=x_col, y=y_col,
                color=None if color_col == "Tidak ada" else color_col,
                size=None if size_col == "Tidak ada" else size_col,
                trendline="ols" if len(plot_df) >= 3 else None,
                opacity=0.7,
                title=f"{x_col} vs {y_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
            if len(plot_df) >= 3:
                corr = plot_df[[x_col, y_col]].corr().iloc[0, 1]
                st.metric("Pearson correlation", f"{corr:.3f}")
                st.caption("Korelasi menunjukkan asosiasi linear, bukan hubungan sebab-akibat.")
        else:
            st.info("Scatter plot membutuhkan minimal dua kolom numerik.")

    elif chart_type == "Time Trend":
        if date_cols and numeric_cols:
            date_col = st.selectbox("Date column", date_cols)
            metric = st.selectbox("Metric", numeric_cols)
            frequency_label = st.selectbox("Time aggregation", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
            frequency_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
            trend = pd.DataFrame({
                "date": coerce_date_column(df, date_col),
                "value": pd.to_numeric(df[metric], errors="coerce"),
            }).dropna()
            if not trend.empty:
                trend = trend.set_index("date").resample(frequency_map[frequency_label])["value"].agg(["sum", "mean", "count"]).reset_index()
                measure = st.radio("Metric calculation", ["sum", "mean", "count"], horizontal=True)
                fig = px.line(trend, x="date", y=measure, markers=True, title=f"{metric} — {frequency_label} trend ({measure})")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(trend, use_container_width=True, hide_index=True)
        else:
            st.info("Time trend membutuhkan kolom tanggal dan numerik.")

    elif chart_type == "Top / Bottom Ranking":
        if categorical_cols and numeric_cols:
            cat = st.selectbox("Category", categorical_cols)
            metric = st.selectbox("Metric", numeric_cols)
            direction = st.radio("Ranking", ["Top", "Bottom"], horizontal=True)
            n = st.slider("Number of categories", 5, 30, 10)
            ranking = df.groupby(cat, dropna=False)[metric].sum().sort_values(ascending=(direction == "Bottom")).head(n).reset_index()
            fig = px.bar(ranking, x=metric, y=cat, orientation="h", text_auto=".3s", title=f"{direction} {n}: {metric} by {cat}")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ranking, use_container_width=True, hide_index=True)
        else:
            st.info("Ranking membutuhkan kolom kategori dan numerik.")


# ---------------------------------------------------------------------------
# TAB 3: Correlation and outliers
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Correlation Analysis")
    if len(numeric_cols) >= 2:
        method = st.radio("Correlation method", ["Pearson", "Spearman", "Kendall"], horizontal=True)
        corr_matrix = df[numeric_cols].corr(method=method.lower())
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"{method} Correlation Matrix",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr_matrix.iloc[i, j]
                if pd.notna(value):
                    pairs.append((cols[i], cols[j], value))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            st.markdown("**Strongest variable relationships**")
            st.dataframe(
                pd.DataFrame(pairs[:10], columns=["Variable 1", "Variable 2", "Correlation"])
                .assign(Strength=lambda x: x["Correlation"].abs())
                .sort_values("Strength", ascending=False)
                .drop(columns="Strength")
                .style.format({"Correlation": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("Korelasi tidak membuktikan kausalitas. Periksa ukuran sampel, outlier, dan kemungkinan variabel perancu.")
    else:
        st.info("Korelasi membutuhkan minimal dua kolom numerik.")

    st.divider()
    st.subheader("Outlier Screening — IQR Method")
    if numeric_cols:
        outlier_df = calculate_outliers(df, numeric_cols)
        if not outlier_df.empty:
            st.dataframe(
                outlier_df.style.format({
                    "Q1": "{:,.2f}", "Median": "{:,.2f}", "Q3": "{:,.2f}",
                    "Batas Bawah IQR": "{:,.2f}", "Batas Atas IQR": "{:,.2f}",
                    "Outlier (%)": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
            selected_outlier_metric = st.selectbox("Inspect outlier distribution", numeric_cols)
            s = df[selected_outlier_metric].dropna()
            if len(s):
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                flagged = df[(df[selected_outlier_metric] < low) | (df[selected_outlier_metric] > high)]
                st.metric("Flagged records", f"{len(flagged):,}", delta=pct(len(flagged) / max(len(s), 1) * 100))
                st.dataframe(flagged.head(100), use_container_width=True)
                if len(flagged) > 100:
                    st.caption("Preview dibatasi 100 baris; gunakan data hasil filter untuk mempersempit inspeksi.")
        else:
            st.info("Tidak ada nilai numerik valid untuk dihitung.")
    else:
        st.info("Outlier screening membutuhkan kolom numerik.")


# ---------------------------------------------------------------------------
# TAB 4: Segmentation
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Segment & Contribution Analysis")
    if categorical_cols and numeric_cols:
        c1, c2, c3 = st.columns(3)
        segment_col = c1.selectbox("Segment dimension", categorical_cols)
        metric_col = c2.selectbox("Performance metric", numeric_cols)
        segment_agg = c3.selectbox("Aggregation", ["Sum", "Mean", "Count"], key="visual_category_aggregation")
        segment_data = df[[segment_col, metric_col]].dropna(subset=[segment_col])
        if segment_agg == "Sum":
            seg = segment_data.groupby(segment_col)[metric_col].sum().rename("Value")
        elif segment_agg == "Mean":
            seg = segment_data.groupby(segment_col)[metric_col].mean().rename("Value")
        else:
            seg = segment_data.groupby(segment_col)[metric_col].count().rename("Value")
        seg = seg.sort_values(ascending=False).reset_index()
        total_value = seg["Value"].sum()
        seg["Contribution (%)"] = seg["Value"] / total_value * 100 if total_value else 0
        seg["Cumulative (%)"] = seg["Contribution (%)"].cumsum()

        top_n = st.slider("Segments to display", 5, 30, 10, key="segment_topn")
        display_seg = seg.head(top_n)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=display_seg[segment_col].astype(str), y=display_seg["Value"], name="Value"))
        fig.add_trace(go.Scatter(
            x=display_seg[segment_col].astype(str),
            y=display_seg["Cumulative (%)"],
            name="Cumulative contribution (%)",
            yaxis="y2",
            mode="lines+markers",
        ))
        fig.update_layout(
            title=f"Segment contribution: {metric_col} by {segment_col}",
            yaxis=dict(title="Value"),
            yaxis2=dict(title="Cumulative contribution (%)", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)
        if len(seg):
            st.info(
                f"Segmen terbesar adalah **{seg.iloc[0][segment_col]}** dengan nilai "
                f"**{safe_number(seg.iloc[0]['Value'])}** dan kontribusi "
                f"**{seg.iloc[0]['Contribution (%)']:.1f}%**."
            )
    else:
        st.info("Segment analysis membutuhkan minimal satu kolom kategori dan satu kolom numerik.")


# ---------------------------------------------------------------------------
# TAB 5: Executive summary and exports
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Executive Summary")
    st.caption("Ringkasan otomatis berbasis statistik dataset. Validasi konteks bisnis sebelum mengambil keputusan.")

    summary_lines = build_rule_based_summary(
        df, raw_df, numeric_cols, categorical_cols, date_cols, selected_metric
    )
    for idx, line in enumerate(summary_lines, start=1):
        st.markdown(f"<div class='insight-card'><b>{idx:02d}.</b> {line}</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Key Findings & Management Considerations")
    considerations = []
    if missing_pct >= 10:
        considerations.append(f"Missing value mencapai {missing_pct:.1f}%; periksa apakah pola missing terkonsentrasi pada kolom atau periode tertentu sebelum analisis lanjutan.")
    elif missing_cells:
        considerations.append(f"Terdapat {missing_cells:,} missing cells; evaluasi dampaknya pada metrik yang digunakan.")
    else:
        considerations.append("Tidak ditemukan missing cells pada data hasil filter.")
    if duplicate_rows:
        considerations.append(f"Ada {duplicate_rows:,} baris duplikat. Pastikan apakah ini duplikasi transaksi/record atau observasi yang memang sah.")
    else:
        considerations.append("Tidak ditemukan baris duplikat identik.")
    if numeric_cols:
        outlier_summary = calculate_outliers(df, numeric_cols)
        if not outlier_summary.empty:
            most_outlier = outlier_summary.sort_values("Outlier (%)", ascending=False).iloc[0]
            considerations.append(
                f"Proporsi outlier IQR tertinggi terdapat pada **{most_outlier['Metrik']}** "
                f"({most_outlier['Outlier']:,} nilai; {most_outlier['Outlier (%)']:.1f}%). "
                "Flag ini perlu ditinjau, bukan otomatis dihapus."
            )
    for item in considerations:
        st.markdown(f"- {item}")

    st.markdown("**Recommended management workflow**")
    st.markdown(
        "1. **Validate:** konfirmasi definisi KPI, satuan, periode, dan kelengkapan sumber data.  \n"
        "2. **Diagnose:** telusuri segmen, periode, atau variabel yang paling berkontribusi terhadap perubahan.  \n"
        "3. **Prioritize:** pilih 1–3 temuan dengan dampak bisnis terbesar dan tingkat keyakinan data memadai.  \n"
        "4. **Act & monitor:** tetapkan owner, target, tenggat, dan metrik pemantauan untuk setiap tindakan."
    )

    st.divider()
    st.subheader("AI-assisted Strategic Narrative (Optional)")
    st.warning(
        "Jika fitur AI digunakan, hanya ringkasan statistik, profil kolom, dan sampel terbatas "
        "yang dikirim ke Google Gemini. Jangan gunakan dataset berisi data pribadi, rahasia, "
        "atau informasi sensitif tanpa otorisasi dan kebijakan yang sesuai."
    )
    with st.expander("Generate AI narrative"):
        api_key = st.text_input("Gemini API key", type="password", key="gemini_key")
        ai_model = st.selectbox("Model", ["gemini-2.5-flash"], index=0)
        if st.button("Generate strategic executive summary", type="primary"):
            if not api_key.strip():
                st.error("Masukkan Gemini API key terlebih dahulu.")
            else:
                # Send aggregated profile only; do not transmit the full raw dataset.
                profile = {
                    "dataset_shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
                    "quality": {
                        "missing_cells": missing_cells,
                        "missing_percent": round(missing_pct, 2),
                        "duplicate_rows": duplicate_rows,
                    },
                    "numeric_summary": numeric_profile(df, numeric_cols).replace({np.nan: None}).to_dict(orient="records")[:30],
                    "categorical_summary": [
                        {
                            "column": col,
                            "unique": int(df[col].nunique(dropna=True)),
                            "top_values": df[col].value_counts(dropna=True).head(5).to_dict(),
                        }
                        for col in categorical_cols[:15]
                    ],
                    "date_ranges": {
                        col: {
                            "min": str(coerce_date_column(df, col).min()),
                            "max": str(coerce_date_column(df, col).max()),
                        }
                        for col in date_cols[:5]
                    },
                    "rule_based_findings": [re.sub(r"\*\*", "", x) for x in summary_lines],
                }
                prompt = f"""
Anda adalah konsultan strategi dan Chief Data Officer. Susun executive report
dalam Bahasa Indonesia yang lugas, berbasis bukti, dan ditujukan kepada direksi.

Gunakan hanya data yang tersedia di bawah. Jangan mengarang target, penyebab,
benchmark industri, atau angka. Bedakan fakta, interpretasi, hipotesis, dan
rekomendasi. Jika data tidak cukup, nyatakan keterbatasannya.

Struktur wajib:
1. Executive Brief (3-5 kalimat: kondisi utama, temuan paling penting, implikasi)
2. Business Performance & Key Findings (angka, kontribusi, tren/pola yang didukung)
3. Data Quality & Analytical Caveats
4. Risks / Opportunities (prioritaskan berdasarkan urgensi dan potensi dampak; tandai hipotesis)
5. Strategic Recommendations (3-5 tindakan konkret; untuk setiap tindakan sertakan rationale,
   prioritas High/Medium/Low, KPI pemantauan, dan data tambahan yang dibutuhkan)
6. Decisions Required from Management (keputusan yang perlu diambil; tulis 'Belum dapat ditentukan'
   jika data tidak mendukung)

Hindari klaim kausalitas dari korelasi. Jangan menyebut temuan sebagai tren jika tidak ada
dimensi waktu. Jangan membuat rekomendasi yang tidak berhubungan dengan temuan.

DATA PROFILE:
{json.dumps(profile, ensure_ascii=False, default=str)}
"""
                with st.spinner("Menyusun analisis strategis..."):
                    try:
                        ai_text = call_gemini(api_key.strip(), prompt, ai_model)
                        st.session_state["ai_executive_summary"] = ai_text
                    except Exception as exc:
                        st.error(f"Gagal menghasilkan analisis AI: {exc}")

        if st.session_state.get("ai_executive_summary"):
            st.markdown(st.session_state["ai_executive_summary"])

    # Report artifacts
    quality = data_quality_report(df)
    outlier_report = calculate_outliers(df, numeric_cols)
    report_md = make_markdown_report(
        df, summary_lines, quality, outlier_report, numeric_cols,
        title=f"Executive Data Report — {uploaded_file.name}",
    )
    ai_saved = st.session_state.get("ai_executive_summary")
    if ai_saved:
        report_md += "\n\n## AI-assisted Strategic Narrative\n\n" + ai_saved

    st.divider()
    st.subheader("Export & Download")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.download_button(
            "Download Executive Report (.md)",
            data=report_md.encode("utf-8"),
            file_name="executive_data_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp2:
        csv_buffer = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download Filtered Data (.csv)",
            data=csv_buffer,
            file_name="filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp3:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Filtered Data")
            quality.to_excel(writer, index=False, sheet_name="Data Quality")
            if not outlier_report.empty:
                outlier_report.to_excel(writer, index=False, sheet_name="Outlier Analysis")
            if numeric_cols:
                numeric_profile(df, numeric_cols).to_excel(writer, index=False, sheet_name="Numeric Profile")
        st.download_button(
            "Download Analysis Workbook (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name="executive_data_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.caption(
    "Analytical note: aplikasi ini membantu eksplorasi dan penyusunan hipotesis. "
    "Validasi definisi metrik, kualitas sumber, dan konteks operasional tetap diperlukan."
)
