import streamlit as st
import pandas as pd
import joblib # Untuk memuat model .pkl
import numpy as np # Mungkin dibutuhkan untuk preprocessing data input

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Harga Saham",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("📈 Prediksi Harga Saham Sederhana")
st.write("""
Aplikasi ini memprediksi harga saham (atau tren) berdasarkan beberapa parameter input.
**Penting:** Aplikasi ini hanya contoh demonstrasi dan tidak boleh digunakan untuk keputusan investasi riil.
""")

st.markdown("---")

# --- Muat Model yang Telah Disimpan ---
# Pastikan 'model_saham.pkl' ada di direktori yang sama dengan 'app.py'
try:
    model = joblib.load('model_saham.pkl')
    st.sidebar.success("Model prediksi saham berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: 'model_saham.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- Input Fitur dari Pengguna di Sidebar ---
st.sidebar.header("Parameter Input Prediksi")
st.sidebar.write("Sesuaikan nilai di bawah untuk melihat prediksi:")

def user_input_features():
    # Contoh fitur. Anda perlu menyesuaikan ini berdasarkan fitur yang digunakan oleh model Anda.
    # Misalnya, Anda mungkin punya fitur seperti:
    # - Harga penutupan hari sebelumnya
    # - Volume perdagangan
    # - Indikator teknikal (RSI, MACD)
    # - Sentimen berita (jika model Anda memproses teks)

    harga_penutupan_kemarin = st.sidebar.number_input("Harga Penutupan Kemarin ($)", min_value=0.01, value=150.00, step=0.01)
    volume_perdagangan = st.sidebar.number_input("Volume Perdagangan (Juta)", min_value=0.0, value=5.0, step=0.1)
    indikator_teknikal_1 = st.sidebar.slider("Indikator X (0-100)", 0, 100, 50)
    indikator_teknikal_2 = st.sidebar.slider("Indikator Y (-1 s/d 1)", -1.0, 1.0, 0.0, 0.01)

    # Buat DataFrame dari input pengguna. Nama kolom harus sama persis dengan yang digunakan saat melatih model.
    data = {
        'harga_penutupan_kemarin': harga_penutupan_kemarin,
        'volume_perdagangan': volume_perdagangan,
        'indikator_teknikal_1': indikator_teknikal_1,
        'indikator_teknikal_2': indikator_teknikal_2
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
if st.button("Lakukan Prediksi Harga Saham"):
    st.subheader("Hasil Prediksi")
    try:
        # Lakukan prediksi menggunakan model
        # Output model bisa berupa harga prediksi, kategori (naik/turun), dll.
        prediction = model.predict(df_input)

        # Contoh: Jika model memprediksi harga numerik
        st.success(f"Harga Saham Prediksi: **${prediction[0]:.2f}**")

        # Contoh: Jika model memprediksi kategori (misal: 0=Turun, 1=Stabil, 2=Naik)
        # label_mapping = {0: 'Turun', 1: 'Stabil', 2: 'Naik'}
        # predicted_label = label_mapping.get(prediction[0], 'Tidak diketahui')
        # st.info(f"Tren Saham Prediksi: **{predicted_label}**")

        # Jika model Anda mendukung predict_proba (untuk klasifikasi)
        if hasattr(model, 'predict_proba'):
            st.write("Probabilitas:")
            # Sesuaikan indeks probabilitas dengan kelas model Anda
            st.write(f"Probabilitas Turun: **{model.predict_proba(df_input)[0][0]:.2f}**")
            st.write(f"Probabilitas Naik: **{model.predict_proba(df_input)[0][1]:.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("""
---
*Disclaimer: Prediksi ini bersifat hipotetis dan tidak mewakili saran finansial. Pasar saham sangat fluktuatif.*
""")
