from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Executive Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------ Styling -------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px;
       background: linear-gradient(120deg,#101b32,#142c4c); border:1px solid #244665;}
.hero h1 {margin:0; color:#f4f7fb; font-size:2rem;}
.hero p {margin:.45rem 0 0; color:#a9bed1;}
.step-card {background:rgba(60,190,210,.07); border:1px solid rgba(60,190,210,.3);
            border-radius:12px; padding:14px; min-height:100px;}
.kpi {background:#101b32; border:1px solid #244665; border-radius:12px; padding:12px;}
.kpi-label {font-size:.75rem;color:#9bb0c5;text-transform:uppercase;letter-spacing:.04em;}
.kpi-value {font-size:1.55rem;color:#43d5e6;font-weight:750;}
.kpi-note {font-size:.72rem;color:#9bb0c5;}
div[data-testid="stMetric"] {background:#101b32;border:1px solid #244665;padding:12px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Executive Data Analyzer</h1>
  <p>Upload data → pilih pertanyaan → dapatkan insight → unduh laporan eksekutif.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------ Helpers -------------------------------------
def fmt_num(v: Any, digits: int = 1) -> str:
    try:
        if pd.isna(v):
            return "—"
        x = float(v)
        if abs(x) >= 1_000_000_000: return f"{x/1_000_000_000:,.{digits}f}B"
        if abs(x) >= 1_000_000: return f"{x/1_000_000:,.{digits}f}M"
        if abs(x) >= 1_000: return f"{x:,.{digits}f}"
        return f"{x:,.{digits}f}"
    except Exception:
        return str(v)

def infer_dates(frame: pd.DataFrame) -> list[str]:
    found = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        sample = frame[col].dropna().astype(str).head(100)
        if len(sample) >= 5:
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            if parsed.notna().mean() >= .85:
                found.append(col)
    return list(dict.fromkeys(found))

def read_upload(file, sheet=None):
    if file.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    return pd.read_excel(file, sheet_name=sheet)

def category_summary(frame: pd.DataFrame, col: str, top_n: int = 6):
    s = frame[col].dropna().astype(str)
    if s.empty:
        return pd.DataFrame(columns=["Kategori", "Jumlah", "Persentase"])
    counts = s.value_counts().head(top_n)
    denominator = max(len(s), 1)
    return pd.DataFrame({
        "Kategori": counts.index.astype(str),
        "Jumlah": counts.values,
        "Persentase": counts.values / denominator * 100,
    })

def numeric_summary(frame: pd.DataFrame, col: str):
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "count": len(s), "sum": s.sum(), "mean": s.mean(), "median": s.median(),
        "min": s.min(), "max": s.max(), "std": s.std(),
    }

def quality_profile(frame: pd.DataFrame):
    n = max(len(frame), 1)
    rows = []
    for col in frame.columns:
        rows.append({
            "Kolom": str(col),
            "Tipe": str(frame[col].dtype),
            "Kosong": int(frame[col].isna().sum()),
            "Kosong (%)": frame[col].isna().mean() * 100,
            "Nilai unik": int(frame[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)



def detect_business_context(frame: pd.DataFrame):
    """Heuristic dataset classification; labels are suggestions, not guaranteed semantics."""
    names = [str(c).lower().replace("_", " ").replace("-", " ") for c in frame.columns]
    blob = " ".join(names)
    profiles = {
        "Audience & Survey": ["respondent", "responden", "gender", "jenis kelamin", "age", "usia", "umur", "occupation", "pekerjaan", "domicile", "domisili", "satisfaction", "kepuasan", "survey", "survei", "preference", "preferensi"],
        "Marketing & Digital Performance": ["reach", "impression", "engagement", "click", "ctr", "cpc", "cpm", "conversion", "campaign", "campaign", "social", "platform", "followers", "views", "profile visit"],
        "Sales & Revenue": ["revenue", "sales", "penjualan", " omzet", "transaction", "transaksi", "order", "quantity", "qty", "unit sold", "harga", "price", "product", "produk", "customer"],
        "Event Performance": ["attendee", "attendance", "visitor", "pengunjung", "ticket", "tiket", "event", "activation", "aktivasi", "venue", "gate", "registration", "registrasi"],
        "Finance & Budget": ["budget", "anggaran", "actual", "realisasi", "expense", "expenditure", "cost", "biaya", "spend", "variance", "margin", "profit", "laba", "cashflow"],
        "Operations & Productivity": ["duration", "durasi", "cycle time", "productivity", "produktivitas", "output", "downtime", "defect", "error", "sla", "completion", "completed", "status", "inventory", "stock"]
    }
    scores = {k: sum(1 for token in tokens if token in blob) for k, tokens in profiles.items()}

    # Keep the winning score before applying the General fallback.
    # "General Business Dataset" is not a key in `scores`, so looking it
    # up there caused a KeyError for datasets with unrecognized columns.
    detected_domain = max(scores, key=scores.get)
    winning_score = scores[detected_domain]
    total_score = sum(scores.values())

    if winning_score == 0:
        best = "General Business Dataset"
        confidence = 0.0
    else:
        best = detected_domain
        confidence = winning_score / max(total_score, 1)

    return best, confidence, scores


def infer_metric_roles(frame: pd.DataFrame, numeric_cols: list[str]):
    """Map likely business KPI roles from column names, preserving the original columns."""
    roles = {
        "Revenue / Value": ["revenue", "sales", "gmv", "amount", "value", "nilai", "nominal", "total", "price", "harga", "cost", "biaya", "budget", "actual", "spend", "expense", "profit", "margin"],
        "Volume / Count": ["count", "jumlah", "qty", "quantity", "unit", "volume", "visitor", "attendee", "ticket", "order", "transaction", "record", "respondent", "responden", "impression", "view", "click", "reach", "follower"],
        "Rate / Percentage": ["rate", "ratio", "percent", "percentage", "persen", "conversion", "ctr", "engagement rate", "growth", "retention", "share", "margin"],
        "Duration / Time": ["duration", "durasi", "time", "waktu", "latency", "hour", "minute", "second", "menit", "detik"]
    }
    mapped = {role: [] for role in roles}
    for col in numeric_cols:
        low = str(col).lower().replace("_", " ").replace("-", " ")
        matches = [role for role, tokens in roles.items() if any(t in low for t in tokens)]
        for role in matches:
            mapped[role].append(col)
    return mapped


def build_auto_insights(frame: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str], date_cols: list[str]):
    findings = []
    n, p = len(frame), len(frame.columns)
    missing = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    findings.append(("Dataset scale", f"Dataset berisi {n:,} records dan {p:,} variables.", "info"))
    if missing:
        top_missing = frame.isna().mean().sort_values(ascending=False)
        col = str(top_missing.index[0])
        findings.append(("Data completeness", f"Terdapat {missing:,} missing cells. Kolom dengan missing rate tertinggi adalah {col} ({top_missing.iloc[0]*100:.1f}%).", "warning"))
    else:
        findings.append(("Data completeness", "Tidak ditemukan missing values pada dataset ini.", "success"))
    if duplicates:
        findings.append(("Duplicate records", f"Ditemukan {duplicates:,} baris duplikat identik ({duplicates/max(n,1)*100:.1f}% dari records). Validasi sebelum menghapusnya.", "warning"))
    if cat_cols:
        for col in cat_cols[:4]:
            summ = category_summary(frame, col, 1)
            if not summ.empty:
                r = summ.iloc[0]
                findings.append((f"Largest segment · {col}", f"Kategori terbesar adalah '{r['Kategori']}' dengan {int(r['Jumlah']):,} records ({r['Persentase']:.1f}% dari nilai terisi).", "insight"))
    for col in numeric_cols[:4]:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s):
            findings.append((f"Metric · {col}", f"Rata-rata {fmt_num(s.mean())}; median {fmt_num(s.median())}; rentang {fmt_num(s.min())} hingga {fmt_num(s.max())}.", "insight"))
    if date_cols:
        col = date_cols[0]
        dates = pd.to_datetime(frame[col], errors="coerce", dayfirst=True).dropna()
        if len(dates):
            findings.append(("Time coverage", f"Rentang tanggal terdeteksi: {dates.min().strftime('%d %b %Y')} sampai {dates.max().strftime('%d %b %Y')}.", "info"))
    return findings


def build_pdf(frame, report_title, selected_categories, primary_metric):
    """Build a compact, dark editorial dashboard PDF using ReportLab Platypus."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable, KeepTogether
    )

    NAVY = colors.HexColor("#0B1425")
    PANEL = colors.HexColor("#111F35")
    PANEL2 = colors.HexColor("#172943")
    CYAN = colors.HexColor("#32D5E7")
    WHITE = colors.HexColor("#F3F6FA")
    MUTED = colors.HexColor("#A7B8CB")
    GRID = colors.HexColor("#29415D")
    GREEN = colors.HexColor("#58D6B0")

    class BarRows(Flowable):
        def __init__(self, data, width, height=86):
            Flowable.__init__(self)
            self.data, self.width, self.height = data, width, height
        def wrap(self, aw, ah):
            return self.width, self.height
        def draw(self):
            c = self.canv
            if not self.data:
                c.setFillColor(MUTED); c.setFont("Helvetica", 7)
                c.drawString(0, self.height-12, "Tidak ada nilai tersedia")
                return
            row_h = min(15, self.height / max(len(self.data), 1))
            max_v = max([x[1] for x in self.data] + [1])
            for i, (label, value, share) in enumerate(self.data):
                y = self.height - (i+1)*row_h + 2
                c.setFillColor(WHITE); c.setFont("Helvetica", 6.4)
                shown = label if len(label) <= 20 else label[:18] + "…"
                c.drawString(0, y+3, shown)
                bx = self.width * .48
                bw = self.width * .34
                c.setFillColor(GRID); c.roundRect(bx, y+2, bw, 5, 2, fill=1, stroke=0)
                c.setFillColor(CYAN); c.roundRect(bx, y+2, bw*float(value)/max_v, 5, 2, fill=1, stroke=0)
                c.setFillColor(CYAN); c.setFont("Helvetica-Bold", 6.4)
                c.drawRightString(self.width, y+3, f"{share:.1f}%")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontName="Helvetica-Bold",
                                 fontSize=17, leading=20, textColor=WHITE, alignment=TA_LEFT, spaceAfter=2)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=7.5, leading=10,
                               textColor=MUTED, spaceAfter=5)
    section_style = ParagraphStyle("SectionX", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=8.5, leading=11, textColor=CYAN, spaceBefore=4, spaceAfter=5)
    card_title = ParagraphStyle("CardTitleX", parent=styles["Normal"], fontName="Helvetica-Bold",
                                fontSize=7.3, leading=9, textColor=WHITE)
    body_style = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=7, leading=9,
                                textColor=WHITE)
    muted_style = ParagraphStyle("MutedX", parent=styles["Normal"], fontSize=6.2, leading=8,
                                 textColor=MUTED)
    kpi_value = ParagraphStyle("KpiValueX", parent=styles["Normal"], fontName="Helvetica-Bold",
                               fontSize=15, leading=17, textColor=CYAN)
    kpi_label = ParagraphStyle("KpiLabelX", parent=styles["Normal"], fontSize=6.2, leading=8,
                               textColor=MUTED)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=12*mm, leftMargin=12*mm,
                            topMargin=10*mm, bottomMargin=10*mm, title=report_title)
    usable = A4[0] - 24*mm
    story = []

    def page_bg(canvas, doc_obj):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(.5)
        canvas.line(12*mm, A4[1]-27*mm, A4[0]-12*mm, A4[1]-27*mm)
        canvas.setFillColor(MUTED); canvas.setFont("Helvetica", 6)
        canvas.drawString(12*mm, 6*mm, "EXECUTIVE DATA ANALYZER  •  GENERATED REPORT")
        canvas.drawRightString(A4[0]-12*mm, 6*mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(
        f"Dataset: {len(frame):,} records · {len(frame.columns):,} variables · "
        f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style
    ))

    missing = int(frame.isna().sum().sum())
    dup = int(frame.duplicated().sum())
    kpis = [
        ("TOTAL RECORDS", f"{len(frame):,}", "Basis data yang dianalisis"),
        ("TOTAL VARIABLES", f"{len(frame.columns):,}", "Kolom dalam dataset"),
        ("MISSING CELLS", f"{missing:,}", f"{missing/max(frame.size,1)*100:.1f}% dari seluruh sel"),
        ("DUPLICATE ROWS", f"{dup:,}", "Baris identik"),
    ]
    kpi_cells = []
    for label, value, note in kpis:
        kpi_cells.append([
            Paragraph(label, kpi_label), Paragraph(value, kpi_value),
            Paragraph(note, muted_style)
        ])
    kpi_table = Table([[Table(cell, colWidths=[usable/4-8], rowHeights=None) for cell in kpi_cells]],
                      colWidths=[usable/4]*4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),PANEL),("BOX",(0,0),(-1,-1),.5,GRID),
        ("INNERGRID",(0,0),(-1,-1),.5,GRID),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    story += [kpi_table, Spacer(1, 5), Paragraph("CATEGORY & DATA PROFILE", section_style)]

    cats = selected_categories[:6]
    if not cats:
        cats = [c for c in frame.select_dtypes(include=["object","category","string","bool"]).columns][:6]
    card_width = (usable - 10) / 3
    card_tables = []
    for col in cats:
        summary = category_summary(frame, col, 5)
        data = [(str(r["Kategori"]), float(r["Jumlah"]), float(r["Persentase"]))
                for _, r in summary.iterrows()]
        chart = BarRows(data, card_width-14, height=74)
        card = Table([
            [Paragraph(str(col).upper(), card_title)],
            [Paragraph(f"{int(frame[col].notna().sum()):,} valid responses", muted_style)],
            [chart],
        ], colWidths=[card_width-8])
        card.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),PANEL),("BOX",(0,0),(-1,-1),.6,GRID),
            ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        card_tables.append(card)
    if card_tables:
        while len(card_tables) % 3:
            card_tables.append(Paragraph("", body_style))
        for i in range(0, len(card_tables), 3):
            row = Table([[card_tables[i], card_tables[i+1], card_tables[i+2]]],
                        colWidths=[card_width]*3)
            row.setStyle(TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),
                ("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),2),
                ("BOTTOMPADDING",(0,0),(-1,-1),3),
            ]))
            story.append(row)
    else:
        story.append(Paragraph("Tidak ditemukan kolom kategori untuk profil distribusi.", muted_style))

    if primary_metric and primary_metric in frame.columns:
        stats = numeric_summary(frame, primary_metric)
        if stats:
            story += [Spacer(1, 4), Paragraph("KEY METRIC PERFORMANCE", section_style)]
            metric_cells = [
                [Paragraph(primary_metric, card_title), Paragraph("TOTAL", kpi_label),
                 Paragraph("AVERAGE", kpi_label), Paragraph("MEDIAN", kpi_label),
                 Paragraph("MIN – MAX", kpi_label)],
                [Paragraph("Selected metric", muted_style),
                 Paragraph(fmt_num(stats["sum"]), kpi_value),
                 Paragraph(fmt_num(stats["mean"]), kpi_value),
                 Paragraph(fmt_num(stats["median"]), kpi_value),
                 Paragraph(f"{fmt_num(stats['min'])} – {fmt_num(stats['max'])}", body_style)],
            ]
            mt = Table(metric_cells, colWidths=[usable*.22, usable*.19, usable*.19, usable*.19, usable*.21])
            mt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),PANEL),("BOX",(0,0),(-1,-1),.5,GRID),
                ("INNERGRID",(0,0),(-1,-1),.4,GRID),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ]))
            story.append(mt)

    story += [Spacer(1, 5), Paragraph("EXECUTIVE SIGNALS", section_style)]
    signals = []
    if len(frame):
        signals.append(f"Dataset mencakup <b>{len(frame):,} records</b> dan <b>{len(frame.columns)} variables</b>.")
    if missing:
        signals.append(f"Ada <b>{missing:,} missing cells</b> ({missing/max(frame.size,1)*100:.1f}%); periksa kolom dengan tingkat missing tertinggi.")
    else:
        signals.append("Tidak ditemukan missing cells pada dataset yang dianalisis.")
    if dup:
        signals.append(f"Ditemukan <b>{dup:,} duplicate rows</b>; validasi apakah merupakan duplikasi aktual atau observasi yang sah.")
    else:
        signals.append("Tidak ditemukan baris duplikat identik.")
    for col in cats[:3]:
        summ = category_summary(frame, col, 1)
        if not summ.empty:
            r = summ.iloc[0]
            signals.append(f"Pada <b>{col}</b>, kategori terbesar adalah <b>{r['Kategori']}</b> ({r['Persentase']:.1f}%).")
    if primary_metric and primary_metric in frame.columns:
        s = numeric_summary(frame, primary_metric)
        if s:
            signals.append(f"Metrik <b>{primary_metric}</b>: total {fmt_num(s['sum'])}, rata-rata {fmt_num(s['mean'])}, median {fmt_num(s['median'])}.")
    signal_table = Table([[Paragraph("• " + x, body_style)] for x in signals[:6]], colWidths=[usable])
    signal_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),PANEL2),("BOX",(0,0),(-1,-1),.6,GRID),
        ("LINEBEFORE",(0,0),(0,-1),2,CYAN),("LEFTPADDING",(0,0),(-1,-1),9),
        ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(signal_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Interpretation note: hasil ini bersifat deskriptif. Distribusi dan korelasi tidak membuktikan sebab-akibat. "
        "Validasi definisi variabel, representativitas sampel, dan konteks operasional sebelum keputusan.",
        muted_style
    ))
    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    return buffer.getvalue()


# ------------------------------ Upload --------------------------------------
uploaded = st.file_uploader(
    "① Upload dataset kamu",
    type=["csv", "xlsx", "xls"],
    help="Format CSV atau Excel. Data diproses di sesi aplikasi.",
    key="main_dataset_uploader",
)

if uploaded is None:
    st.markdown("### Mulai dalam 3 langkah")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="step-card"><b>1. Upload data</b><br><small>Masukkan file CSV atau Excel.</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-card"><b>2. Pilih insight</b><br><small>Pilih pertanyaan bisnis yang ingin dijawab.</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-card"><b>3. Download report</b><br><small>Hasilkan laporan visual siap dibagikan.</small></div>', unsafe_allow_html=True)
    st.info("Belum ada dataset. Upload file untuk membuka menu analisis.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df_raw = read_upload(uploaded)
        sheet = None
    else:
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Pilih sheet Excel", xls.sheet_names, key="excel_sheet_select")
        df_raw = read_upload(uploaded, sheet)
except Exception as exc:
    st.error(f"File gagal dibaca: {exc}")
    st.stop()

if df_raw.empty:
    st.error("Dataset kosong.")
    st.stop()

df = df_raw.dropna(how="all").dropna(axis=1, how="all").copy()
df.columns = [str(c).strip() or f"Kolom_{i+1}" for i, c in enumerate(df.columns)]
df = df.loc[:, ~pd.Index(df.columns).duplicated()]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
date_cols = infer_dates(df)

# ------------------------------ Snapshot ------------------------------------
st.success(f"Dataset berhasil dibaca: {len(df):,} baris dan {len(df.columns):,} kolom.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total records", f"{len(df):,}")
k2.metric("Total variables", f"{len(df.columns):,}")
k3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
k4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

st.divider()
st.header("② Apa yang ingin kamu ketahui dari data ini?")
st.caption("Pilih salah satu kebutuhan. Aplikasi akan menampilkan analisis yang relevan tanpa perlu mengatur grafik satu per satu.")

domain, domain_confidence, domain_scores = detect_business_context(df)
metric_roles = infer_metric_roles(df, numeric_cols)
auto_findings = build_auto_insights(df, numeric_cols, cat_cols, date_cols)

st.markdown("### Dataset recognition")
dc1, dc2, dc3 = st.columns([1.4, 1, 1])
dc1.metric("Detected data context", domain)
dc2.metric("Numeric variables", len(numeric_cols))
dc3.metric("Category / text variables", len(cat_cols))
st.caption("Klasifikasi otomatis berdasarkan nama kolom dan tipe data. Anggap sebagai rekomendasi awal; verifikasi konteks bisnis sebelum mengambil keputusan.")

question_options = ["Automatic insights", "Gambaran umum dataset"]
if cat_cols:
    question_options += ["Profil kategori / segmen", "Profil audiens / pelanggan", "Kategori atau segmen terbesar"]
if numeric_cols:
    question_options += ["Performa metrik", "Perbandingan metrik berdasarkan segmen"]
if date_cols and (numeric_cols or len(df)):
    question_options += ["Tren berdasarkan waktu"]
if len(numeric_cols) >= 2:
    question_options += ["Hubungan antarvariabel"]
question_options += ["Kualitas data dan anomali", "Buat laporan eksekutif lengkap"]
question = st.selectbox("Apa yang ingin kamu ketahui?", question_options, key="insight_question")
st.caption("Pilih satu kebutuhan. Aplikasi akan menampilkan analisis yang sesuai dengan kolom yang tersedia.")

# Keep controls and analysis in the same clear guided area.
if question == "Automatic insights":
    st.subheader("Temuan awal yang terdeteksi otomatis")
    st.caption("Insight ini bersifat deskriptif dan dihasilkan dari pola statistik dasar, bukan kesimpulan kausal.")
    for title, detail, level in auto_findings:
        if level == "warning":
            st.warning(f"**{title}** — {detail}")
        elif level == "success":
            st.success(f"**{title}** — {detail}")
        else:
            st.info(f"**{title}** — {detail}")
    if metric_roles:
        detected_roles = {k: v for k, v in metric_roles.items() if v}
        if detected_roles:
            st.markdown("**Kandidat KPI berdasarkan nama kolom**")
            role_rows = [{"Kandidat KPI": role, "Kolom terdeteksi": ", ".join(map(str, cols))} for role, cols in detected_roles.items()]
            st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)
    st.markdown("**Rekomendasi eksplorasi berikutnya**")
    suggestions = []
    if cat_cols: suggestions.append("Profil kategori / segmen untuk melihat komposisi dan konsentrasi data.")
    if numeric_cols: suggestions.append("Performa metrik untuk memahami skala, rata-rata, median, dan sebaran nilai.")
    if date_cols and numeric_cols: suggestions.append("Tren berdasarkan waktu untuk mengidentifikasi pola temporal.")
    if len(numeric_cols) >= 2: suggestions.append("Hubungan antarvariabel untuk memeriksa asosiasi antarindikator.")
    for item in suggestions: st.markdown(f"- {item}")

elif question == "Profil kategori / segmen":
    st.subheader("Profil kategori / segmen")
    selected_profile_cols = st.multiselect("Pilih dimensi yang ingin dibandingkan", cat_cols, default=cat_cols[:min(6,len(cat_cols))], max_selections=9, key="adaptive_profile_dims")
    if selected_profile_cols:
        chart_cols = st.columns(2)
        for i, col in enumerate(selected_profile_cols):
            summary = category_summary(df, col, 10)
            with chart_cols[i % 2]:
                st.markdown(f"**{col}**")
                fig = px.bar(summary.sort_values("Jumlah"), x="Jumlah", y="Kategori", orientation="h", text=summary.sort_values("Jumlah")["Persentase"].map(lambda x: f"{x:.1f}%"), title=f"Distribusi {col}")
                fig.update_layout(height=320, margin=dict(l=5,r=5,t=35,b=5))
                st.plotly_chart(fig, use_container_width=True, key=f"adaptive_profile_chart_{i}")

elif question == "Perbandingan metrik berdasarkan segmen":
    st.subheader("Perbandingan metrik berdasarkan segmen")
    metric = st.selectbox("Pilih KPI / metrik", numeric_cols, key="adaptive_compare_metric")
    dimension = st.selectbox("Kelompokkan berdasarkan", cat_cols, key="adaptive_compare_dimension")
    agg_label = st.selectbox("Cara perhitungan", ["Total", "Rata-rata", "Jumlah data"], key="adaptive_compare_agg")
    agg_func = {"Total":"sum", "Rata-rata":"mean", "Jumlah data":"count"}[agg_label]
    result = df.groupby(dimension, dropna=False)[metric].agg(agg_func).reset_index(name="Hasil").sort_values("Hasil", ascending=False).head(20)
    st.plotly_chart(px.bar(result, x=dimension, y="Hasil", title=f"{agg_label} {metric} berdasarkan {dimension}"), use_container_width=True)
    st.dataframe(result, use_container_width=True, hide_index=True)

if question == "Gambaran umum dataset":
    st.subheader("Gambaran umum")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("**Komposisi tipe data**")
    type_df = pd.DataFrame({
        "Tipe": ["Numerik", "Kategori / teks", "Tanggal"],
        "Jumlah kolom": [len(numeric_cols), len(cat_cols), len(date_cols)]
    })
    st.plotly_chart(px.bar(type_df, x="Tipe", y="Jumlah kolom", color="Tipe",
                           title="Komposisi variabel"), use_container_width=True)
    st.markdown("**Statistik numerik**")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)
    else:
        st.info("Tidak ditemukan kolom numerik.")

elif question == "Profil audiens / pelanggan":
    st.subheader("Profil audiens / pelanggan")
    st.caption("Pilih dimensi seperti usia, gender, pekerjaan, domisili, preferensi, atau kanal informasi.")
    if not cat_cols:
        st.warning("Tidak ada kolom kategori yang dapat dianalisis.")
    else:
        selected_profile_cols = st.multiselect(
            "Dimensi profil yang ingin ditampilkan",
            cat_cols,
            default=cat_cols[:min(6, len(cat_cols))],
            max_selections=9,
            key="audience_profile_dimensions",
        )
        if selected_profile_cols:
            cols = st.columns(3)
            for i, col in enumerate(selected_profile_cols):
                with cols[i % 3]:
                    st.markdown(f"**{col}**")
                    summary = category_summary(df, col, 8)
                    fig = px.bar(summary.sort_values("Jumlah"), x="Jumlah", y="Kategori",
                                 orientation="h", text=summary.sort_values("Jumlah")["Persentase"].map(lambda x: f"{x:.1f}%"),
                                 title=None)
                    fig.update_layout(height=260, margin=dict(l=5,r=5,t=5,b=5),
                                      showlegend=False, yaxis_title=None, xaxis_title="Jumlah")
                    st.plotly_chart(fig, use_container_width=True, key=f"audience_chart_{i}")
                    st.dataframe(summary, use_container_width=True, hide_index=True)

elif question == "Kategori atau segmen terbesar":
    st.subheader("Analisis kategori / segmen")
    if not cat_cols:
        st.warning("Tidak ada kolom kategori.")
    else:
        chosen_cat = st.selectbox("Kategori yang ingin dibandingkan", cat_cols, key="segment_category_select")
        top_n = st.slider("Jumlah kategori", 3, 20, min(10, max(3, df[chosen_cat].nunique())), key="segment_topn_select")
        summary = category_summary(df, chosen_cat, top_n)
        fig = px.bar(summary.sort_values("Jumlah"), x="Jumlah", y="Kategori", orientation="h",
                     text=summary.sort_values("Jumlah")["Persentase"].map(lambda x: f"{x:.1f}%"),
                     title=f"Distribusi {chosen_cat}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)

elif question == "Performa metrik":
    st.subheader("Analisis metrik")
    if not numeric_cols:
        st.warning("Tidak ada kolom numerik.")
    else:
        metric = st.selectbox("Pilih metrik", numeric_cols, key="metric_performance_select")
        stats = numeric_summary(df, metric)
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        a,b,c,d = st.columns(4)
        a.metric("Total", fmt_num(stats["sum"]))
        b.metric("Average", fmt_num(stats["mean"]))
        c.metric("Median", fmt_num(stats["median"]))
        d.metric("Valid values", f"{stats['count']:,}")
        fig = px.histogram(df, x=metric, marginal="box", title=f"Distribusi {metric}")
        st.plotly_chart(fig, use_container_width=True)
        if cat_cols:
            split = st.selectbox("Bandingkan berdasarkan kategori (opsional)", ["Tidak ada"] + cat_cols, key="metric_split_select")
            if split != "Tidak ada":
                agg = df.groupby(split, dropna=False)[metric].agg(["sum","mean","count"]).reset_index()
                st.plotly_chart(px.bar(agg.sort_values("sum", ascending=False).head(15),
                                       x=split, y="sum", title=f"Total {metric} per {split}"),
                                use_container_width=True)

elif question == "Tren berdasarkan waktu":
    st.subheader("Analisis tren waktu")
    if not date_cols or not numeric_cols:
        st.warning("Dibutuhkan setidaknya satu kolom tanggal dan satu kolom numerik.")
    else:
        date_col = st.selectbox("Kolom tanggal", date_cols, key="trend_date_select")
        metric = st.selectbox("Metrik", numeric_cols, key="trend_metric_select")
        freq_label = st.selectbox("Periode", ["Harian","Mingguan","Bulanan","Kuartalan","Tahunan"], key="trend_freq_select")
        freq = {"Harian":"D","Mingguan":"W","Bulanan":"MS","Kuartalan":"QS","Tahunan":"YS"}[freq_label]
        calc = st.radio("Perhitungan", ["Total","Rata-rata","Jumlah record"], horizontal=True, key="trend_calc_select")
        tmp = pd.DataFrame({"tanggal":pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
                            "nilai":pd.to_numeric(df[metric], errors="coerce")}).dropna()
        if not tmp.empty:
            resample_agg = {"Total":"sum","Rata-rata":"mean","Jumlah record":"count"}[calc]
            trend = tmp.set_index("tanggal").resample(freq)["nilai"].agg(resample_agg).reset_index()
            st.plotly_chart(px.line(trend, x="tanggal", y="nilai", markers=True,
                                    title=f"{calc} {metric} — {freq_label}"), use_container_width=True)
            st.dataframe(trend, use_container_width=True, hide_index=True)

elif question == "Hubungan antarvariabel":
    st.subheader("Hubungan antarvariabel")
    if len(numeric_cols) < 2:
        st.warning("Dibutuhkan minimal dua kolom numerik.")
    else:
        x = st.selectbox("Variabel X", numeric_cols, key="relationship_x_select")
        y = st.selectbox("Variabel Y", [c for c in numeric_cols if c != x], key="relationship_y_select")
        method = st.radio("Metode korelasi", ["Pearson","Spearman","Kendall"], horizontal=True, key="relationship_corr_method")
        corr = df[[x,y]].corr(method=method.lower()).iloc[0,1]
        st.metric(f"Korelasi {method}", f"{corr:.3f}" if pd.notna(corr) else "Tidak tersedia")
        fig = px.scatter(df, x=x, y=y, trendline="ols" if df[[x,y]].dropna().shape[0] >= 3 else None,
                         title=f"{x} vs {y}", opacity=.7)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Korelasi menunjukkan asosiasi, bukan hubungan sebab-akibat.")

elif question == "Kualitas data dan anomali":
    st.subheader("Audit kualitas data")
    quality = quality_profile(df)
    st.dataframe(quality, use_container_width=True, hide_index=True)
    missing = quality[quality["Kosong"] > 0].sort_values("Kosong", ascending=False)
    if not missing.empty:
        st.plotly_chart(px.bar(missing, x="Kosong", y="Kolom", orientation="h",
                               title="Missing values per kolom"), use_container_width=True)
    if numeric_cols:
        rows = []
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                q1, q3 = s.quantile([.25,.75]); iqr=q3-q1
                count=int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
                rows.append({"Metrik":col,"Outlier IQR":count,"Outlier (%)":count/max(len(s),1)*100})
        st.markdown("**Screening outlier (IQR)**")
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)
    st.caption("Outlier adalah penanda untuk diperiksa, bukan otomatis kesalahan atau data yang harus dihapus.")

# ------------------------------ Report builder -----------------------------
st.divider()
st.header("③ Buat laporan dan download")
st.caption("Laporan adaptif: KPI, profil kategori, metrik utama, dan executive signals disusun berdasarkan kolom yang tersedia pada dataset.")

with st.container(border=True):
    report_title = st.text_input("Judul laporan", value=f"{domain.upper()} — EXECUTIVE DATA OVERVIEW", key="report_title_input")
    r1, r2 = st.columns(2)
    with r1:
        report_categories = st.multiselect(
            "Dimensi kategori untuk laporan (maksimal 6)",
            cat_cols,
            default=cat_cols[:min(6,len(cat_cols))],
            max_selections=6,
            key="report_category_fields",
            help="Contoh: gender, usia, pekerjaan, domisili, kanal promosi, preferensi brand."
        )
    with r2:
        preferred_metrics = [c for role in ["Revenue / Value", "Volume / Count", "Rate / Percentage", "Duration / Time"] for c in metric_roles.get(role, [])]
        preferred_metrics = list(dict.fromkeys(preferred_metrics + numeric_cols))
        report_metric = st.selectbox(
            "Metrik numerik utama (opsional)",
            ["Tidak ada"] + preferred_metrics,
            key="report_primary_metric",
        )

    if st.button("Generate Executive Overview PDF", type="primary", use_container_width=True, key="generate_pdf_report"):
        try:
            metric_arg = None if report_metric == "Tidak ada" else report_metric
            pdf_bytes = build_pdf(df, report_title.strip() or "EXECUTIVE DATA OVERVIEW",
                                  report_categories, metric_arg)
            st.session_state["generated_pdf"] = pdf_bytes
            st.session_state["generated_pdf_name"] = "adaptive_executive_data_overview.pdf"
            st.success("Laporan PDF berhasil dibuat.")
        except Exception as exc:
            st.error(f"Gagal membuat PDF: {exc}")

    if st.session_state.get("generated_pdf"):
        st.download_button(
            "⬇️ Download Executive Overview (PDF)",
            data=st.session_state["generated_pdf"],
            file_name=st.session_state.get("generated_pdf_name","adaptive_executive_data_overview.pdf"),
            mime="application/pdf",
            use_container_width=True,
            key="download_generated_pdf",
        )

# Other useful exports
with st.expander("Download data dan tabel analisis"):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download filtered dataset (CSV)", csv_bytes,
                       "analyzed_dataset.csv", "text/csv", key="download_dataset_csv")
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
        quality_profile(df).to_excel(writer, index=False, sheet_name="Data Quality")
        if numeric_cols:
            df[numeric_cols].describe().T.to_excel(writer, sheet_name="Numeric Summary")
    st.download_button("Download analysis workbook (Excel)", xbuf.getvalue(),
                       "analysis_workbook.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="download_analysis_xlsx")

st.caption("Catatan: analisis bersifat deskriptif. Pastikan definisi variabel, kualitas sampel, dan konteks bisnis tervalidasi sebelum mengambil keputusan.")
