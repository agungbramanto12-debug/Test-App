
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Upload Test",
    layout="wide"
)

st.title("UPLOAD COMPONENT TEST")
st.success("Streamlit is running correctly!")

st.write("Jika kamu melihat halaman ini, deployment berhasil.")

uploaded_file = st.file_uploader(
    "UPLOAD YOUR CSV OR EXCEL FILE",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")
    st.dataframe(df, use_container_width=True)
