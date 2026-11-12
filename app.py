import pandas as pd
import streamlit as st
import plotly.express as px
import json
import urllib.request

# Konfigurasi Halaman
st.set_page_config(page_title="Executive Data Suite", layout="wide")

st.title("📊 Executive Data Analyzer Pro")
st.markdown("Platform analisis data komprehensif, deteksi anomali, dan *Executive Summary* otomatis.")

# 1. Modul Unggah & Baca Data
uploaded_file = st.file_uploader("📂 Unggah file CSV atau Excel (.xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca CSV atau Excel (Mendukung Multi-Sheet)
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.selectbox("📑 Pilih Sheet Excel:", sheet_names)
        df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Identifikasi Tipe Kolom
    num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter Data Cepat (Expander agar rapi di HP)
    df = df_raw.copy()
    with st.expander("🔍 Filter Data (Opsional)"):
        if len(cat_cols) > 0:
            filter_col = st.selectbox("Pilih Kolom Kategori untuk Filter:", ["Semua Data"] + cat_cols)
            if filter_col != "Semua Data":
                unique_vals = df_raw[filter_col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"Pilih Nilai {filter_col}:", unique_vals, default=unique_vals)
                df = df_raw[df_raw[filter_col].isin(selected_vals)]
        st.caption(f"Menampilkan {len(df):,} dari {len(df_raw):,} baris data.")

    # Pembagian Tab Navigasi
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Profil & Statistik", 
        "📊 Visualisasi", 
        "🔗 Korelasi & Anomali", 
        "🧠 Executive Summary"
    ])

    # ================= TAB 1: PROFIL & STATISTIK =================
    with tab1:
        st.subheader("Ringkasan Kualitas Data")
        c1, c2, c3, c4 = st.columns(4)
        missing_total = int(df.isna().sum().sum())
        dup_total = int(df.duplicated().sum())
        
        c1.metric("Total Baris", f"{df.shape[0]:,}")
        c2.metric("Total Kolom", f"{df.shape[1]:,}")
        c3.metric("Data Kosong (Missing)", f"{missing_total:,}")
        c4.metric("Baris Duplikat", f"{dup_total:,}")

        st.subheader("Pratinjau Data")
        st.dataframe(df.head(10), use_container_width=True)

        if len(num_cols) > 0:
            st.subheader("Statistik Deskriptif (Numerik)")
            stats_df = df[num_cols].describe().T
            stats_df['median'] = df[num_cols].median()
            stats_df['total (sum)'] = df[num_cols].sum()
            st.dataframe(stats_df[['count', 'mean', 'median', 'std', 'min', 'max', 'total (sum)']].round(2), use_container_width=True)

        if len(cat_cols) > 0:
            st.subheader("Ringkasan Kategori / Teks")
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)

    # ================= TAB 2: VISUALISASI DINAMIS =================
    with tab2:
        st.subheader("Eksplorasi Grafik Interaktif")
        chart_type = st.radio("Pilih Jenis Analisis Grafik:", ["Perbandingan Kategori", "Distribusi & Box Plot", "Hubungan 2 Variabel (Scatter)"], horizontal=True)

        if chart_type == "Perbandingan Kategori" and len(cat_cols) > 0 and len(num_cols) > 0:
            col_x = st.selectbox("Pilih Kategori (Sumbu X):", cat_cols)
            col_y = st.selectbox("Pilih Metrik Angka (Sumbu Y):", num_cols)
            agg_func = st.selectbox("Metode Perhitungan:", ["Total (Sum)", "Rata-rata (Mean)", "Jumlah Data (Count)"])
            
            if agg_func == "Total (Sum)":
                grouped = df.groupby(col_x, as_index=False)[col_y].sum()
            elif agg_func == "Rata-rata (Mean)":
                grouped = df.groupby(col_x, as_index=False)[col_y].mean()
            else:
                grouped = df.groupby(col_x, as_index=False)[col_y].count()
            
            grouped = grouped.sort_values(by=col_y, ascending=False).head(15)
            fig_bar = px.bar(grouped, x=col_x, y=col_y, color=col_x, title=f"Top 15 {col_x} berdasarkan {agg_func} {col_y}")
            st.plotly_chart(fig_bar, use_container_width=True)

        elif chart_type == "Distribusi & Box Plot" and len(num_cols) > 0:
            target_num = st.selectbox("Pilih Kolom Numerik:", num_cols)
            fig_hist = px.histogram(df, x=target_num, marginal="box", title=f"Distribusi & Sebaran Outlier: {target_num}")
            st.plotly_chart(fig_hist, use_container_width=True)

        elif chart_type == "Hubungan 2 Variabel (Scatter)" and len(num_cols) >= 2:
            sc_x = st.selectbox("Sumbu X:", num_cols, index=0)
            sc_y = st.selectbox("Sumbu Y:", num_cols, index=1)
            color_opt = st.selectbox("Warna Kategori (Opsional):", ["Tidak Ada"] + cat_cols)
            fig_sc = px.scatter(
                df, x=sc_x, y=sc_y, 
                color=None if color_opt == "Tidak Ada" else color_opt,
                title=f"Korelasi {sc_x} vs {sc_y}"
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Data tidak memiliki cukup kolom kategori/numerik untuk jenis grafik ini.")

    # ================= TAB 3: KORELASI & ANOMALI =================
    with tab3:
        if len(num_cols) >= 2:
            st.subheader("🔥 Peta Korelasi Antar Variabel (Heatmap)")
            corr_matrix = df[num_cols].corr().round(2)
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Matriks Korelasi (-1 s/d 1)")
            st.plotly_chart(fig_corr, use_container_width=True)

        if len(num_cols) > 0:
            st.subheader("⚠️ Deteksi Anomali / Outlier Otomatis (Metode IQR)")
            outlier_summary = []
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_summary.append({
                    "Kolom": col,
                    "Batas Bawah": round(lower_bound, 2),
                    "Batas Atas": round(upper_bound, 2),
                    "Jumlah Outlier": len(outliers),
                    "Persentase (%)": round((len(outliers) / len(df)) * 100, 2) if len(df) > 0 else 0
                })
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)

    # ================= TAB 4: EXECUTIVE SUMMARY =================
    with tab4:
        st.subheader("📝 Automated Executive Summary")
        
        # 1. Ringkasan Berbasis Aturan Statistik (Instan tanpa API)
        auto_insights = []
        auto_insights.append(f"**1. Skala & Kualitas Data:** Dataset terdiri dari **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**. Ditemukan **{missing_total:,} sel kosong** dan **{dup_total:,} baris duplikat**.")
        
        if len(num_cols) > 0:
            for col in num_cols[:3]:
                mean_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                sum_val = df[col].sum()
                auto_insights.append(f"**Metrik `{col}`:** Total keseluruhan mencapai **{sum_val:,.2f}** dengan rata-rata **{mean_val:,.2f}** (Rentang nilai: {min_val:,.2f} hingga {max_val:,.2f}).")

        if len(cat_cols) > 0 and len(num_cols) > 0:
            top_cat_col = cat_cols[0]
            top_num_col = num_cols[0]
            top_group = df.groupby(top_cat_col)[top_num_col].sum().sort_values(ascending=False)
            if len(top_group) > 0:
                best_cat = top_group.index[0]
                best_val = top_group.iloc[0]
                auto_insights.append(f"**Dominasi Kategori:** Pada kategori `{top_cat_col}`, kontributor terbesar terhadap `{top_num_col}` adalah **{best_cat}** dengan nilai total **{best_val:,.2f}**.")

        report_text = "### 📌 Ringkasan Eksekutif Statistik\n" + "\n\n".join([f"- {item}" for item in auto_insights])
        st.markdown(report_text)

        st.divider()

        # 2. Integrasi AI Mendalam (Opsional menggunakan Gemini API)
        st.subheader("🤖 Buat Analisa Strategis Mendalam dengan AI (Opsional)")
        st.caption("Masukkan Gemini API Key gratis dari Google AI Studio (aistudio.google.com) untuk menghasilkan narasi eksekutif dan rekomendasi bisnis mendalam.")
        
        api_key = st.text_input("🔑 Masukkan Gemini API Key:", type="password")
        
        if st.button("🚀 Generate AI Executive Summary"):
            if not api_key:
                st.warning("Silakan masukkan Gemini API Key terlebih dahulu untuk fitur narasi AI, atau gunakan Ringkasan Eksekutif Statistik di atas.")
            else:
                with st.spinner("AI sedang menganalisis pola data dan menyusun rekomendasi strategis..."):
                    try:
                        # Menyiapkan konteks data untuk dikirim ke AI
                        stats_context = df.describe(include='all').to_string()
                        prompt = f"""
                        Anda adalah Chief Data Officer dan Konsultan Strategi Bisnis. 
                        Analisis ringkasan statistik dataset berikut dan buatkan laporan dalam Bahasa Indonesia yang profesional dengan struktur:
                        1. Executive Summary (Kesimpulan utama kondisi data secara makro)
                        2. Key Findings & Anomali (Temuan pola penting, ketimpangan, atau potensi masalah kualitas data)
                        3. Actionable Recommendations (3-4 rekomendasi strategis konkret yang bisa langsung dieksekusi manajemen)

                        Data Statistik:
                        {stats_context}
                        """
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                        headers = {"Content-Type": "application/json"}
                        payload = {"contents": [{"parts": [{"text": prompt}]}]}
                        
                        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                        with urllib.request.urlopen(req) as response:
                            res_data = json.loads(response.read().decode("utf-8"))
                            ai_output = res_data["candidates"][0]["content"]["parts"][0]["text"]
                            
                            st.success("Analisis Selesai!")
                            st.markdown(ai_output)
                            report_text += "\n\n---\n\n### 🤖 Analisis & Rekomendasi Strategis AI\n" + ai_output
                    except Exception as e:
                        st.error(f"Gagal memanggil AI API. Pastikan API Key valid. Detail: {e}")

        st.divider()
        st.download_button(
            label="📥 Download Laporan Executive Summary (.txt)",
            data=report_text,
            file_name="Executive_Summary_Report.txt",
            mime="text/plain"
        )
