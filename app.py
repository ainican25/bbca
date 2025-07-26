import streamlit as st
import pandas as pd
import joblib # Untuk memuat model .pkl
import numpy as np # Mungkin dibutuhkan untuk preprocessing data input

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Harga Penutupan Saham",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("📈 Prediksi Harga Penutupan Saham")
st.write("""
Aplikasi ini memprediksi harga penutupan saham (`Close`) berdasarkan
harga pembukaan (`Open`), harga tertinggi (`High`), harga terendah (`Low`),
dan volume perdagangan (`Volume`).
""")

st.markdown("---")

# --- Muat Model yang Telah Disimpan ---
# Pastikan 'model_saham.pkl' ada di direktori yang sama dengan 'app.py'
try:
    # Model diasumsikan adalah model regresi (misal: LinearRegression, RandomForestRegressor, dll.)
    # yang menerima 4 fitur dan mengembalikan 1 prediksi numerik.
    model = joblib.load('model_saham.pkl')
    st.sidebar.success("Model prediksi saham berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: 'model_saham.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Input Fitur dari Pengguna di Sidebar ---
st.sidebar.header("Parameter Input Prediksi Saham")
st.sidebar.write("Masukkan nilai untuk setiap parameter:")

def user_input_features():
    # Asumsi harga saham dan volume dalam nilai positif
    new_open = st.sidebar.number_input("Harga Pembukaan (Open)", min_value=0.0, value=100.0, step=0.01)
    new_high = st.sidebar.number_input("Harga Tertinggi (High)", min_value=0.0, value=105.0, step=0.01)
    new_low = st.sidebar.number_input("Harga Terendah (Low)", min_value=0.0, value=98.0, step=0.01)
    new_volume = st.sidebar.number_input("Volume Perdagangan (Juta Unit)", min_value=0.0, value=1000000.0, step=1000.0)

    # Buat DataFrame dari input pengguna. Nama kolom harus sama persis dengan yang digunakan saat melatih model.
    data = {
        'Open': new_open,
        'High': new_high,
        'Low': new_low,
        'Volume': new_volume
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Dapatkan input dari pengguna
df_input = user_input_features()

# --- Tampilkan Data Input Pengguna ---
st.subheader("Data Input untuk Prediksi")
st.write(df_input)

# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Lakukan Prediksi Harga Penutupan"):
    st.subheader("Hasil Prediksi")
    try:
        # Lakukan prediksi menggunakan model
        # Output model diasumsikan adalah harga numerik
        predicted_close = model.predict(df_input)

        # predicted_close adalah array. Ambil nilai tunggalnya
        # Perhatikan: jika model.predict menghasilkan array 2D seperti [[value]], gunakan [0][0]
        # Jika model.predict menghasilkan array 1D seperti [value], gunakan [0]
        # Saya asumsikan ini menghasilkan 2D array berdasarkan kode Anda sebelumnya
        predicted_value = predicted_close[0][0]

        st.success(f"Prediksi Harga Penutupan Saham (Close) adalah: **IDR {predicted_value:,.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("""
---
*Disclaimer: Prediksi ini bersifat hipotetis dan tidak mewakili saran finansial. Pasar saham sangat fluktuatif dan berinvestasi memiliki risiko.*
""")
