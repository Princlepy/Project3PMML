import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import seaborn as sns # Hanya import seaborn untuk plot

# ======================================================================================
# KONFIGURASI PUSAT
# ======================================================================================
MODEL_FILENAME = 'best_random_forest_model.pkl' # Nama model diubah

FEATURE_ORDER = [
    'num__login_attempts', 'num__failed_logins', 'num__ip_reputation_score',
    'num__network_packet_size', 'cat__unusual_time_access', 'cat__browser_type_Chrome',
    'cat__browser_type_Firefox', 'cat__browser_type_Edge', 'cat__browser_type_Safari',
    'cat__browser_type_Unknown', 'cat__protocol_type_TCP', 'cat__protocol_type_UDP',
    'cat__protocol_type_ICMP', 'cat__encryption_used_AES', 'cat__encryption_used_DES',
    'cat__encryption_used_None', '17', '18', '19', '20'
]

ORIGINAL_NUMERIC_FEATURES = [
    'num__login_attempts', 'num__failed_logins',
    'num__ip_reputation_score', 'num__network_packet_size'
]

ORIGINAL_CATEGORICAL_FEATURES_MAP = {
    'cat__unusual_time_access_input': 'cat__unusual_time_access',
    'cat__browser_type_input': 'cat__browser_type',
    'cat__protocol_type_input': 'cat__protocol_type',
    'cat__encryption_used_input': 'cat__encryption_used'
}
BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "None"]

# ======================================================================================
# FUNGSI UNTUK MEMUAT MODEL
# ======================================================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        return None
    try:
        model = joblib.load(model_path)
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(FEATURE_ORDER):
            st.error(f"FATAL: Model mengharapkan {model.n_features_in_} fitur, "
                       f"tetapi FEATURE_ORDER memiliki {len(FEATURE_ORDER)} fitur.")
            return None
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# ======================================================================================
# FUNGSI UNTUK PRA-PEMROSESAN DATASET
# ======================================================================================
def preprocess_dataframe(df_input):
    processed_rows = []
    for _, row in df_input.iterrows():
        final_model_inputs = {}
        for feature in ORIGINAL_NUMERIC_FEATURES:
            final_model_inputs[feature] = row.get(feature, 0)

        user_choice_unusual_time = row.get('cat__unusual_time_access_input', "Tidak")
        final_model_inputs['cat__unusual_time_access'] = 1 if user_choice_unusual_time == "Ya" else 0

        user_choice_browser = row.get('cat__browser_type_input', "Unknown")
        for browser_option in BROWSER_OPTIONS:
            feature_name = f"cat__browser_type_{browser_option}"
            if feature_name in FEATURE_ORDER:
                 final_model_inputs[feature_name] = 1 if user_choice_browser == browser_option else 0

        user_choice_protocol = row.get('cat__protocol_type_input', "TCP")
        for protocol_option in PROTOCOL_OPTIONS:
            feature_name = f"cat__protocol_type_{protocol_option}"
            if feature_name in FEATURE_ORDER:
                final_model_inputs[feature_name] = 1 if user_choice_protocol == protocol_option else 0

        user_choice_encryption = row.get('cat__encryption_used_input', "None")
        for encryption_option in ENCRYPTION_OPTIONS:
            feature_name = f"cat__encryption_used_{encryption_option}"
            if feature_name in FEATURE_ORDER:
                final_model_inputs[feature_name] = 1 if user_choice_encryption == encryption_option else 0
        
        for feature in FEATURE_ORDER:
            if feature not in final_model_inputs:
                final_model_inputs[feature] = 0
        processed_rows.append(final_model_inputs)
    return pd.DataFrame(processed_rows, columns=FEATURE_ORDER)

# ======================================================================================
# FUNGSI UNTUK PREDIKSI PADA DATASET
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed):
    if df_processed.empty:
        return pd.DataFrame()
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)
    df_hasil = pd.DataFrame({
        'Status Deteksi Numerik': predictions,
        'Probabilitas Normal (%)': (prediction_probas[:, 0] * 100).round(2),
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })
    df_hasil['Status Deteksi'] = df_hasil['Status Deteksi Numerik'].apply(lambda x: 'Terancam' if x == 1 else 'Aman')
    df_hasil['Tingkat Kepercayaan Ancaman (%)'] = df_hasil['Probabilitas Ancaman (%)']
    return df_hasil

# ======================================================================================
# ANTARMUKA PENGGUNA (UI)
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard Lanjutan", layout="wide", page_icon="🛡️")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard - Analisis Dataset")
model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV (pemisah ';') atau Excel", type=["csv", "xlsx", "xls"]) # Ditambahkan info pemisah
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Pastikan dataset Anda memiliki kolom berikut:
        - `num__login_attempts`
        - `num__failed_logins`
        - `num__ip_reputation_score`
        - `num__network_packet_size`
        - `cat__unusual_time_access_input` (Teks: 'Ya'/'Tidak')
        - `cat__browser_type_input` (Teks: e.g., 'Chrome', 'Firefox')
        - `cat__protocol_type_input` (Teks: e.g., 'TCP', 'UDP')
        - `cat__encryption_used_input` (Teks: e.g., 'AES', 'None')
        Untuk file CSV, pastikan pemisah kolom adalah titik koma (`;`).
        """
    ) # Ditambahkan info pemisah

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Pembacaan CSV dengan separator titik koma
                df_input_original = pd.read_csv(uploaded_file, sep=';') 
            elif uploaded_file.name.endswith(('.xls', '.xlsx')): # .xls, .xlsx
                df_input_original = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
                st.stop()


            st.subheader("📄 Data Awal yang Diunggah (Contoh 5 Baris Pertama):")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis pada Seluruh Dataset", type="primary", use_container_width=True):
                with st.spinner("Menganalisis data... ⏳"):
                    df_processed_for_model = preprocess_dataframe(df_input_original.copy())
                    if df_processed_for_model.empty:
                        st.warning("Tidak ada data yang bisa diproses.")
                    else:
                        df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)
                        df_tampilan_akhir = pd.concat([df_input_original.reset_index(drop=True),
                                                       df_prediksi_hasil.reset_index(drop=True)], axis=1)

                        st.subheader("📊 Hasil Analisis Lengkap per Data:")
                        kolom_tampil = list(df_input_original.columns) + ['Status Deteksi', 'Tingkat Kepercayaan Ancaman (%)']
                        st.dataframe(df_tampilan_akhir[kolom_tampil], use_container_width=True)
                        st.markdown("---")
                        st.subheader("📈 Ringkasan dan Statistik Hasil Analisis:")

                        jumlah_total_data = len(df_tampilan_akhir)
                        jumlah_terancam = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam'])
                        jumlah_aman = len(df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Aman'])
                        persentase_terancam = (jumlah_terancam / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0
                        persentase_aman = (jumlah_aman / jumlah_total_data) * 100 if jumlah_total_data > 0 else 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Total Data Dianalisis", value=f"{jumlah_total_data} baris")
                        with col2:
                            st.metric(label="🚨 Data Terdeteksi Ancaman", value=f"{jumlah_terancam} baris",
                                      delta=f"{persentase_terancam:.2f}% dari total", delta_color="inverse")
                        with col3:
                            st.metric(label="✅ Data Terdeteksi Aman", value=f"{jumlah_aman} baris",
                                      delta=f"{persentase_aman:.2f}% dari total", delta_color="normal")

                        if jumlah_total_data > 0:
                            st.subheader("Visualisasi Hasil:")
                            
                            st.write("Perbandingan Status Deteksi:")
                            data_grafik_status_df = pd.DataFrame({ # Pastikan DataFrame ini dibuat untuk st.bar_chart
                                'Status': ['Terancam', 'Aman'],
                                'Jumlah': [jumlah_terancam, jumlah_aman]
                            })
                            st.bar_chart(data_grafik_status_df.set_index('Status'), color=["#FF4B4B", "#3DDC97"])


                            if jumlah_terancam > 0:
                                st.write("Distribusi Tingkat Kepercayaan untuk Data Terancam:")
                                data_terancam_probs = df_tampilan_akhir[
                                    df_tampilan_akhir['Status Deteksi'] == 'Terancam'
                                ]['Tingkat Kepercayaan Ancaman (%)']
                                
                                g_probs = sns.displot(data_terancam_probs, kind="hist", kde=True, color="#FF4B4B")
                                g_probs.set_axis_labels("Tingkat Kepercayaan Ancaman (%)", "Jumlah")
                                g_probs.fig.suptitle('Distribusi Tingkat Kepercayaan (Data Terancam)', y=1.03)
                                st.pyplot(g_probs.fig)
                            
                            if 'num__network_packet_size' in df_tampilan_akhir.columns:
                                st.subheader("Analisis 'Ukuran Paket Jaringan (KB)':")
                                
                                g_packet = sns.displot(data=df_tampilan_akhir, 
                                                       x='num__network_packet_size', 
                                                       hue='Status Deteksi', 
                                                       kind="hist",
                                                       kde=True,
                                                       palette={"Aman":"#3DDC97", "Terancam":"#FF4B4B"})
                                g_packet.set_axis_labels("Ukuran Paket Jaringan (KB)", "Jumlah")
                                g_packet.fig.suptitle('Distribusi Ukuran Paket Jaringan berdasarkan Status Deteksi', y=1.03)
                                st.pyplot(g_packet.fig)

        except pd.errors.ParserError:
            st.error("Gagal mem-parsing file. Pastikan format CSV benar dan pemisah kolom adalah titik koma (';').") # Pesan error disesuaikan
        except KeyError as e:
            st.error(f"Kolom yang dibutuhkan tidak ditemukan: {e}. Periksa nama kolom.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("Unggah dataset (CSV dengan pemisah ';' atau Excel) melalui panel kiri.") # Pesan info disesuaikan
else:
    st.warning("Model tidak berhasil dimuat.")
st.markdown("---")
st.caption("Dashboard IDS v0.5 | Dibuat dengan Streamlit & Seaborn")
