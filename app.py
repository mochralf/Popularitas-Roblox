import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Roblox Popularity Classifier 🌸", layout="wide")

# ==============================================
# PINK BLOSSOM HEADER (CSS CUSTOM)
# ==============================================
st.markdown("""
    <style>
    .main { background-color: #fff5f8; }
    .stButton>button {
        background-color: #ffb3d2;
        color: #5a2a41;
        border-radius: 20px;
        border: 2px solid #d14a7c;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #d14a7c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color:#ffe6f0; padding:20px; border-radius:12px; border:2px solid #ffb3d2; margin-bottom:20px;">
        <h1 style="color:#d14a7c; text-align:center;">🌸 Roblox Game Popularity Classifier 🌸</h1>
        <p style="color:#5a2a41; text-align:center;">Masukkan statistik game di bawah ini untuk memprediksi tingkat popularitasnya.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================
# LOAD MODEL & ASSETS
# ==============================================
@st.cache_resource
def load_all():
    try:
        # Pastikan file-file ini ada di folder yang sama dengan app.py
        svm = joblib.load("svm_model.pkl")
        knn = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        # Kita asumsikan target_names juga disimpan saat training, jika tidak kita definisikan manual
        return svm, knn, scaler
    except Exception as e:
        return None, None, None

svm_model, knn_model, scaler = load_all()

# ==============================================
# SIDEBAR - INPUT DATA
# ==============================================
st.sidebar.markdown("### 📝 Input Statistik Game")
st.sidebar.info("Gunakan data dari dashboard Roblox Developer kamu.")

active = st.sidebar.number_input("Active Players", min_value=0, value=0, help="Jumlah pemain online saat ini")
visits = st.sidebar.number_input("Total Visits", min_value=0, value=0)
favourites = st.sidebar.number_input("Total Favourites", min_value=0, value=0)
likes = st.sidebar.number_input("Total Likes", min_value=0, value=0)
dislikes = st.sidebar.number_input("Total Dislikes", min_value=0, value=0)

btn_predict = st.sidebar.button("🌸 Jalankan Prediksi")

# ==============================================
# KONTEN UTAMA
# ==============================================
if svm_model is None or knn_model is None:
    st.warning("⚠️ Model tidak ditemukan! Pastikan file `.pkl` (svm_model, knn_model, scaler) sudah ada di direktori aplikasi.")
    st.info("Pastikan Anda sudah menjalankan script training di Google Colab dan mendownload file modelnya.")
else:
    if btn_predict:
        # 1. Siapkan Data (Sesuaikan dengan urutan fitur saat training)
        feature_cols = ["Active", "Visits", "Favourites", "Likes", "Dislikes"]
        input_data = pd.DataFrame([[active, visits, favourites, likes, dislikes]], columns=feature_cols)

        # 2. Scaling
        input_scaled = scaler.transform(input_data)

        # 3. Prediksi
        pred_svm = svm_model.predict(input_scaled)[0]
        pred_knn = knn_model.predict(input_scaled)[0]

        # 4. Mapping Label (Sesuaikan dengan LabelEncoder: High, Low, Medium biasanya 0, 1, 2 secara alfabetis)
        # Catatan: Periksa kembali urutan le.classes_ di notebook Anda.
        # Umumnya: 0 = High, 1 = Low, 2 = Medium (karena urutan alfabetis)
        label_map = {0: "High", 1: "Low", 2: "Medium"} 

        # Tampilkan Hasil
        st.markdown("### 🔍 Analisis Input")
        st.write(input_data)

        st.markdown("---")
        st.subheader("🔮 Hasil Klasifikasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div style="background-white; padding:20px; border-radius:10px; border-left: 5px solid #d14a7c; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color:#d14a7c;">Model SVM</h4>
                    <h2 style="margin:0;">{label_map.get(pred_svm, "Unknown")}</h2>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background-white; padding:20px; border-radius:10px; border-left: 5px solid #ffb3d2; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color:#5a2a41;">Model KNN</h4>
                    <h2 style="margin:0;">{label_map.get(pred_knn, "Unknown")}</h2>
                </div>
            """, unsafe_allow_html=True)

        # Tambahkan Tips berdasarkan hasil
        st.write("")
        if label_map.get(pred_svm) == "Low":
            st.info("💡 **Tips:** Cobalah untuk memperbarui thumbnail game atau mengadakan event untuk meningkatkan 'Active Players'.")
        elif label_map.get(pred_svm) == "High":
            st.success("🌟 **Luar Biasa!** Game Anda memiliki performa yang sangat kuat di komunitas.")

    else:
        # Tampilan Awal sebelum tombol diklik
        st.write("### Silakan masukkan angka statistik di sidebar dan klik tombol **Prediksi**.")
        st.image("https://img.freepik.com/free-vector/cute-girl-playing-video-game-concept_23-2148535451.jpg", width=400)

# Footer
st.markdown("---")
st.caption("🌸 © 2025 — Roblox Popularity Predictor | Dibuat dengan Streamlit & Scikit-Learn")
