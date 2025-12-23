import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Popularity Classifier 🌸", layout="wide")

# ==============================================
# PINK BLOSSOM HEADER
# ==============================================
st.markdown("""
    <div style="background-color:#ffe6f0; padding:20px; border-radius:12px; border:2px solid #ffb3d2; margin-bottom:20px;">
        <h1 style="color:#d14a7c; text-align:center;">🌸 Roblox Game Popularity Classifier 🌸</h1>
        <p style="color:#5a2a41; text-align:center;">Prediksi tingkat popularitas game Roblox menggunakan model SVM & K-NN.</p>
    </div>
""", unsafe_allow_html=True)

# ==============================================
# LOAD MODEL SAFELY
# ==============================================
@st.cache_resource
def load_all():
    try:
        svm = joblib.load("svm_model.pkl")
        knn = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        evaluation = joblib.load("evaluation.pkl")
        return svm, knn, scaler, evaluation
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        return None, None, None, None

svm_model, knn_model, scaler, evaluation = load_all()

# ==============================================
# CEK VALIDITAS MODEL
# ==============================================
def model_invalid(model, name):
    if model is None:
        return True
    if isinstance(model, np.ndarray):
        st.error(f"❌ ERROR: {name} adalah numpy.ndarray — file PKL salah. Harus model sklearn, bukan array.")
        return True
    if not hasattr(model, "predict"):
        st.error(f"❌ ERROR: {name} tidak punya method .predict() — file PKL rusak.")
        return True
    return False

invalid_svm = model_invalid(svm_model, "svm_model.pkl")
invalid_knn = model_invalid(knn_model, "knn_model.pkl")

if scaler is None:
    st.error("❌ ERROR: scaler.pkl gagal dimuat.")
    invalid_svm = invalid_knn = True

# ==============================================
# FITUR (5 FITUR SAJA — sesuai scaler fit time)
# ==============================================
feature_cols = ["Active", "Visits", "Favourites", "Likes", "Dislikes"]

st.sidebar.write("### Scaler expects:")
st.sidebar.write(feature_cols)

# ==============================================
# INPUT USER
# ==============================================
active = st.sidebar.number_input("Active", min_value=0)
visits = st.sidebar.number_input("Visits", min_value=0)
favourites = st.sidebar.number_input("Favourites", min_value=0)
likes = st.sidebar.number_input("Likes", min_value=0)
dislikes = st.sidebar.number_input("Dislikes", min_value=0)

# ==============================================
# PREDIKSI
# ==============================================
if st.sidebar.button("🌸 Prediksi"):

    if invalid_svm or invalid_knn:
        st.error("❌ Tidak dapat melakukan prediksi karena (model/dataset) yang digunakan tidak valid.")
    else:

        # DataFrame SESUAI DENGAN SCALER (5 FITUR)
        x_df = pd.DataFrame([[
            active, visits, favourites, likes, dislikes
        ]], columns=feature_cols)

        st.write("### 🔍 Input DataFrame:")
        st.write(x_df)

        # TRANSFORM AMAN (TIDAK ADA LAGI FITUR TAMBAHAN)
        x_scaled = scaler.transform(x_df)

        svm_pred = svm_model.predict(x_scaled)[0]
        knn_pred = knn_model.predict(x_scaled)[0]

        label_map = {0: "Low", 1: "Medium", 2: "High"}

        st.subheader("🔮 Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"**SVM:** {label_map[svm_pred]}")

        with col2:
            st.info(f"**KNN:** {label_map[knn_pred]}")


# ==============================================
# VISUALISASI EVALUASI (JIKA ADA)
# ==============================================
if evaluation:
    st.header("📊 Visualisasi Evaluasi Model SVM & K-NN")

    svm_matrix = evaluation.get("svm_matrix")
    knn_matrix = evaluation.get("knn_matrix")

    def plot_matrix(matrix, title):
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap="pink")
        ax.set_title(title, color="#d14a7c")
        ax.set_xlabel("Predicted", color="#5a2a41")
        ax.set_ylabel("Actual", color="#5a2a41")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="#5a2a41")
        st.pyplot(fig)

    colA, colB = st.columns(2)
    with colA:
        if svm_matrix is not None:
            plot_matrix(svm_matrix, "Confusion Matrix - SVM")
    with colB:
        if knn_matrix is not None:
            plot_matrix(knn_matrix, "Confusion Matrix - KNN")

    st.header("📈 Perbandingan Metrik Evaluasi")
    st.subheader("SVM Classification Report")
    st.code(evaluation.get("svm_report", "Tidak ada."))

    st.subheader("KNN Classification Report")
    st.code(evaluation.get("knn_report", "Tidak ada."))

st.write("---")
st.caption("🌸 © 2025 — Roblox Popularity ML Deployment | Pink Blossom Theme 🌸")
