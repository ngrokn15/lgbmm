import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# LOAD MODEL & ARTEFAK (di-cache agar tidak load ulang tiap interaksi)
# =========================================================
@st.cache_resource
def load_artifacts():
    # Model dimuat dari format native (.txt) LightGBM -> aman lintas versi
    model = lgb.Booster(model_file="model_lgb.txt")
    scaler = joblib.load("scaler.pkl")
    le_dict = joblib.load("le_dict.pkl")
    le_target = joblib.load("le_target.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    # metrics.pkl bersifat opsional -> kalau belum ada, tab performa akan
    # menampilkan pesan, bukan error yang menghentikan seluruh aplikasi
    try:
        metrics = joblib.load("metrics.pkl")
    except FileNotFoundError:
        metrics = None

    return model, scaler, le_dict, le_target, feature_columns, metrics


try:
    model, scaler, le_dict, le_target, feature_columns, metrics = load_artifacts()
except (FileNotFoundError, lgb.basic.LightGBMError) as e:
    st.error(
        "File model tidak ditemukan atau gagal dimuat. Pastikan file "
        "model_lgb.txt, scaler.pkl, le_dict.pkl, le_target.pkl, "
        "dan feature_columns.pkl ada di folder yang sama dengan app.py.\n\n"
        f"Detail error: {e}"
    )
    st.stop()


@st.cache_resource
def get_shap_explainer(_model):
    # TreeExplainer untuk model berbasis pohon (LightGBM) tidak wajib
    # membutuhkan data latar belakang, jadi cukup dibuat sekali dari model
    return shap.TreeExplainer(_model)


explainer = get_shap_explainer(model)


# =========================================================
# HALAMAN: PREDIKSI
# =========================================================
def halaman_prediksi():
    st.subheader("Masukkan Data")

    user_input = {}

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        # --- Kolom numerik ---
        with col1:
            user_input["Usia"] = st.number_input(
                "Usia", min_value=1, max_value=120, value=25, step=1
            )
            user_input["Frekuensi_Makan_Sayur"] = st.slider(
                "Frekuensi Makan Sayur (1 = jarang, 3 = sering)", 1.0, 3.0, 2.0, 0.1
            )
            user_input["Jumlah_Makan_Utama"] = st.slider(
                "Jumlah Makan Utama per Hari", 1.0, 4.0, 3.0, 0.1
            )

        with col2:
            user_input["Konsumsi_Air_Putih"] = st.slider(
                "Konsumsi Air Putih per Hari (liter)", 1.0, 3.0, 2.0, 0.1
            )
            user_input["Frekuensi_Olahraga"] = st.slider(
                "Frekuensi Olahraga per Minggu (0 = tidak pernah, 3 = sering)", 0.0, 3.0, 1.0, 0.1
            )
            user_input["Durasi_Layar_Gadget"] = st.slider(
                "Durasi Pemakaian Gadget per Hari (0 = rendah, 2 = tinggi)", 0.0, 2.0, 1.0, 0.1
            )

        st.markdown("---")
        col3, col4 = st.columns(2)

        # --- Kolom kategorikal, opsi diambil dari encoder yang tersimpan ---
        with col3:
            for kolom in ["Jenis_Kelamin", "Riwayat_Keluarga", "Makan_Kalori_Tinggi", "Merokok"]:
                if kolom in le_dict:
                    opsi = list(le_dict[kolom].classes_)
                    user_input[kolom] = st.selectbox(kolom.replace("_", " "), opsi)

        with col4:
            for kolom in ["Sering_Ngemil", "Monitor_Kalori", "Konsumsi_Alkohol", "Transportasi_Utama"]:
                if kolom in le_dict:
                    opsi = list(le_dict[kolom].classes_)
                    user_input[kolom] = st.selectbox(kolom.replace("_", " "), opsi)

        submitted = st.form_submit_button("Prediksi Sekarang", use_container_width=True)

    if submitted:
        # 1. Buat DataFrame dari input, urutan kolom harus sama seperti saat training
        df_input = pd.DataFrame([user_input])
        df_input = df_input[feature_columns]

        # 2. Encode kolom kategorikal menggunakan encoder yang sama dari training
        for kolom, le in le_dict.items():
            if kolom in df_input.columns:
                df_input[kolom] = le.transform(df_input[kolom])

        # 3. Scaling menggunakan scaler yang sama dari training
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

        # 4. Prediksi -> pastikan array float64 & contiguous untuk LightGBM
        data_np = np.ascontiguousarray(df_scaled.values, dtype=np.float64)
        proba = model.predict(data_np)[0]
        pred_encoded = int(proba.argmax())
        pred_label = le_target.inverse_transform([pred_encoded])[0]

        proba_df = pd.DataFrame({
            "Kelas": le_target.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False).reset_index(drop=True)

        st.divider()
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi tingkat obesitas: **{pred_label}**")

        st.write("Probabilitas untuk tiap kelas:")
        st.bar_chart(proba_df.set_index("Kelas"))
        st.dataframe(proba_df, use_container_width=True, hide_index=True)

        # =========================================================
        # SHAP: kenapa model memprediksi kelas ini untuk input tersebut
        # =========================================================
        st.divider()
        st.subheader("Kenapa Model Memprediksi Ini? (SHAP)")

        shap_values = explainer.shap_values(data_np)

        # Format output shap berbeda antar versi library:
        # - versi lama -> list berisi 1 array per kelas
        # - versi baru -> 1 array 3D (baris, fitur, kelas)
        if isinstance(shap_values, list):
            shap_kelas = shap_values[pred_encoded][0]
        else:
            shap_kelas = shap_values[0, :, pred_encoded]

        shap_df = pd.DataFrame({
            "Fitur": feature_columns,
            "Kontribusi SHAP": shap_kelas
        }).sort_values("Kontribusi SHAP", key=abs, ascending=False)

        st.caption(
            f"Kontribusi tiap fitur terhadap prediksi kelas **{pred_label}** untuk data yang kamu "
            "masukkan. Nilai positif mendorong prediksi ke arah kelas ini, nilai negatif menariknya menjauh."
        )
        st.bar_chart(shap_df.set_index("Fitur"))
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        # =========================================================
        # ANALISIS SENSITIVITAS FITUR (PDP sederhana, per-individu)
        # =========================================================
        st.divider()
        st.subheader("Analisis Sensitivitas Fitur")
        st.caption(
            "Lihat bagaimana probabilitas prediksi berubah jika satu fitur diubah-ubah, "
            "sementara semua fitur lain tetap sama seperti input di atas."
        )

        fitur_numerik_range = {
            "Usia": (1, 120),
            "Frekuensi_Makan_Sayur": (1.0, 3.0),
            "Jumlah_Makan_Utama": (1.0, 4.0),
            "Konsumsi_Air_Putih": (1.0, 3.0),
            "Frekuensi_Olahraga": (0.0, 3.0),
            "Durasi_Layar_Gadget": (0.0, 2.0),
        }

        fitur_dipilih = st.selectbox(
            "Pilih fitur yang ingin diuji sensitivitasnya",
            list(fitur_numerik_range.keys()),
            key="fitur_sensitivitas"
        )

        nilai_min, nilai_max = fitur_numerik_range[fitur_dipilih]
        rentang_nilai = np.linspace(nilai_min, nilai_max, 20)

        hasil_sensitivitas = []
        for nilai in rentang_nilai:
            df_variasi = df_input.copy()
            df_variasi[fitur_dipilih] = nilai
            df_variasi_scaled = pd.DataFrame(
                scaler.transform(df_variasi), columns=df_variasi.columns
            )
            data_variasi_np = np.ascontiguousarray(df_variasi_scaled.values, dtype=np.float64)
            proba_variasi = model.predict(data_variasi_np)[0]
            hasil_sensitivitas.append({
                fitur_dipilih: round(float(nilai), 2),
                f"Probabilitas '{pred_label}'": proba_variasi[pred_encoded]
            })

        df_sensitivitas = pd.DataFrame(hasil_sensitivitas)
        st.line_chart(df_sensitivitas.set_index(fitur_dipilih))
        st.caption(
            f"Garis di atas menunjukkan probabilitas kelas **{pred_label}** seiring **{fitur_dipilih.replace('_', ' ')}** "
            "berubah, dengan semua fitur lain tetap seperti input kamu."
        )


# =========================================================
# HALAMAN: PERFORMA MODEL
# =========================================================
def halaman_performa():
    if metrics is None:
        st.info(
            "Metrik performa model belum tersedia. Jalankan cell "
            "`simpan_metrics.py` di notebook, lalu letakkan file "
            "`metrics.pkl` yang dihasilkan sejajar dengan app.py."
        )
        return

    st.subheader("Ringkasan Performa Model (Data Test)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision_weighted']:.2%}")
    c3.metric("Recall", f"{metrics['recall_weighted']:.2%}")
    c4.metric("F1-Score", f"{metrics['f1_weighted']:.2%}")
    st.caption(
        "Precision, Recall, dan F1-Score di atas dihitung dengan "
        "*weighted average* (mempertimbangkan proporsi tiap kelas)."
    )

    st.divider()
    st.subheader("Detail per Kelas")

    report = metrics["classification_report"]
    label_names = metrics["label_names"]

    rows = []
    for label in label_names:
        if label in report:
            rows.append({
                "Kelas": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1-Score": report[label]["f1-score"],
                "Jumlah Data": int(report[label]["support"]),
            })
    report_df = pd.DataFrame(rows)
    st.dataframe(
        report_df.style.format({
            "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"
        }),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.subheader("Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    st.dataframe(
        cm_df.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )
    st.caption("Baris = label asli, Kolom = label hasil prediksi model.")


# =========================================================
# JUDUL & DESKRIPSI
# =========================================================
st.title("🩺 Prediksi Tingkat Obesitas")
st.write(
    "Aplikasi ini memprediksi tingkat obesitas seseorang berdasarkan "
    "kebiasaan makan dan gaya hidup, menggunakan model **LightGBM**."
)
st.divider()

tab_prediksi, tab_performa = st.tabs(["🔮 Prediksi", "📊 Performa Model"])

with tab_prediksi:
    halaman_prediksi()

with tab_performa:
    halaman_performa()
