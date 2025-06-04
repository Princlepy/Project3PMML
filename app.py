import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
# import datetime # Tidak digunakan secara langsung di bagian ini, tapi bisa relevan untuk logging

# ======================================================================================
# KONFIGURASI PUSAT (DARI KODE ANDA)
# ======================================================================================
MODEL_FILENAME = 'best_random_forest_model(1).pkl' # Pastikan file ini ada di direktori yang sama atau berikan path lengkap

# DAFTAR FITUR YANG DIHARAPKAN MODEL (SETELAH ONE-HOT ENCODING)
FEATURE_ORDER = [
    'num__login_attempts',
    'num__failed_logins',
    'num__ip_reputation_score',
    'num__network_packet_size',
    'cat__unusual_time_access', # Ini akan jadi 1 jika 'Ya', 0 jika 'Tidak'
    'cat__browser_type_Chrome',
    'cat__browser_type_Firefox',
    'cat__browser_type_Edge',
    'cat__browser_type_Safari',
    'cat__browser_type_Unknown', # Perhatikan ini
    'cat__protocol_type_TCP',
    'cat__protocol_type_UDP',
    'cat__protocol_type_ICMP',
    'cat__encryption_used_AES',
    'cat__encryption_used_DES',
    'cat__encryption_used_None' # Perhatikan ini
]

# Fitur asli sebelum di-encode (digunakan untuk input pengguna dan pemetaan)
# Nama kolom ini HARUS SESUAI dengan nama kolom di file CSV/Excel yang diunggah pengguna
ORIGINAL_NUMERIC_FEATURES = [
    'num__login_attempts',
    'num__failed_logins',
    'num__ip_reputation_score',
    'num__network_packet_size'
]

ORIGINAL_CATEGORICAL_FEATURES_MAP = {
    'cat__unusual_time_access_input': 'cat__unusual_time_access', # Input pengguna akan 'Ya'/'Tidak'
    'cat__browser_type_input': 'cat__browser_type', # Input pengguna: Chrome, Firefox, dll.
    'cat__protocol_type_input': 'cat__protocol_type', # Input pengguna: TCP, UDP, ICMP
    'cat__encryption_used_input': 'cat__encryption_used' # Input pengguna: AES, DES, None
}
# Pastikan nilai-nilai ini ada di FEATURE_ORDER (misal cat__browser_type_Unknown)
BROWSER_OPTIONS = ["Chrome", "Firefox", "Edge", "Safari", "Unknown"]
PROTOCOL_OPTIONS = ["TCP", "UDP", "ICMP"]
ENCRYPTION_OPTIONS = ["AES", "DES", "None"]


# ======================================================================================
# FUNGSI UNTUK MEMUAT MODEL (DARI KODE ANDA)
# ======================================================================================
@st.cache_resource # Menggunakan cache_resource untuk model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan. Pastikan file berada di path yang benar.")
        return None
    try:
        model = joblib.load(model_path)
        # Verifikasi jumlah fitur yang diharapkan model
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(FEATURE_ORDER):
            st.error(f"FATAL: Model mengharapkan {model.n_features_in_} fitur, "
                       f"tetapi FEATURE_ORDER memiliki {len(FEATURE_ORDER)} fitur.")
            return None
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# ======================================================================================
# FUNGSI UNTUK PRA-PEMROSESAN DATASET (BARU)
# ======================================================================================
def preprocess_dataframe(df_input):
    """
    Mengubah DataFrame input pengguna (dengan kolom asli) menjadi
    DataFrame yang siap untuk diprediksi oleh model (dengan fitur one-hot encoded).
    """
    processed_rows = []

    for _, row in df_input.iterrows():
        final_model_inputs = {}
        # 1. Fitur Numerik Langsung
        for feature in ORIGINAL_NUMERIC_FEATURES:
            final_model_inputs[feature] = row.get(feature, 0) # Default ke 0 jika kolom hilang

        # 2. Fitur Kategorikal - Perlu One-Hot Encoding manual berdasarkan FEATURE_ORDER
        # cat__unusual_time_access
        user_choice_unusual_time = row.get('cat__unusual_time_access_input', "Tidak") # Ambil dari kolom input
        final_model_inputs['cat__unusual_time_access'] = 1 if user_choice_unusual_time == "Ya" else 0

        # cat__browser_type
        user_choice_browser = row.get('cat__browser_type_input', "Unknown") # Ambil dari kolom input
        for browser_option in BROWSER_OPTIONS:
            feature_name = f"cat__browser_type_{browser_option}"
            if feature_name in FEATURE_ORDER: # Pastikan fitur ini ada di FEATURE_ORDER
                 final_model_inputs[feature_name] = 1 if user_choice_browser == browser_option else 0

        # cat__protocol_type
        user_choice_protocol = row.get('cat__protocol_type_input', "TCP") # Default bisa disesuaikan
        for protocol_option in PROTOCOL_OPTIONS:
            feature_name = f"cat__protocol_type_{protocol_option}"
            if feature_name in FEATURE_ORDER:
                final_model_inputs[feature_name] = 1 if user_choice_protocol == protocol_option else 0

        # cat__encryption_used
        user_choice_encryption = row.get('cat__encryption_used_input', "None") # Default bisa disesuaikan
        for encryption_option in ENCRYPTION_OPTIONS:
            feature_name = f"cat__encryption_used_{encryption_option}"
            if feature_name in FEATURE_ORDER:
                final_model_inputs[feature_name] = 1 if user_choice_encryption == encryption_option else 0
        
        # Pastikan semua fitur di FEATURE_ORDER ada, jika tidak ada di atas, set 0
        for feature in FEATURE_ORDER:
            if feature not in final_model_inputs:
                final_model_inputs[feature] = 0 # Safety net

        processed_rows.append(final_model_inputs)

    df_processed = pd.DataFrame(processed_rows, columns=FEATURE_ORDER) # Pastikan urutan kolomnya benar
    return df_processed

# ======================================================================================
# FUNGSI UNTUK PREDIKSI PADA DATASET (BARU)
# ======================================================================================
def model_prediksi_ancaman_dataset(model, df_processed):
    """
    Melakukan prediksi pada DataFrame yang sudah diproses.
    """
    if df_processed.empty:
        return pd.DataFrame()

    # Prediksi
    predictions = model.predict(df_processed)
    prediction_probas = model.predict_proba(df_processed)

    # Buat DataFrame hasil
    df_hasil = pd.DataFrame({
        'Status Deteksi Numerik': predictions, # 0 atau 1
        'Probabilitas Normal (%)': (prediction_probas[:, 0] * 100).round(2),
        'Probabilitas Ancaman (%)': (prediction_probas[:, 1] * 100).round(2)
    })

    df_hasil['Status Deteksi'] = df_hasil['Status Deteksi Numerik'].apply(lambda x: 'Terancam' if x == 1 else 'Aman')
    df_hasil['Tingkat Kepercayaan Ancaman (%)'] = df_hasil['Probabilitas Ancaman (%)'] # Kolom target untuk metrik

    return df_hasil


# ======================================================================================
# ANTARMUKA PENGGUNA (UI) - DIMODIFIKASI UNTUK UNGGAH DATASET
# ======================================================================================
st.set_page_config(page_title="IDS Dashboard Lanjutan", layout="wide", page_icon="🛡️")
st.title("🛡️ Intrusion Detection System (IDS) Dashboard - Analisis Dataset")

# Muat model
model = load_model(MODEL_FILENAME)

if model:
    st.sidebar.header("Unggah Dataset Aktivitas Jaringan")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx", "xls"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Pastikan dataset Anda memiliki kolom berikut untuk analisis:
        - `num__login_attempts` (Numerik)
        - `num__failed_logins` (Numerik)
        - `num__ip_reputation_score` (Numerik, 0-100)
        - `num__network_packet_size` (Numerik, KB)
        - `cat__unusual_time_access_input` (Teks: 'Ya' atau 'Tidak')
        - `cat__browser_type_input` (Teks: e.g., 'Chrome', 'Firefox', 'Unknown')
        - `cat__protocol_type_input` (Teks: e.g., 'TCP', 'UDP', 'ICMP')
        - `cat__encryption_used_input` (Teks: e.g., 'AES', 'DES', 'None')
        """
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input_original = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df_input_original = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
                st.stop() # Hentikan eksekusi lebih lanjut jika format salah

            st.subheader("📄 Data Awal yang Diunggah (Contoh 5 Baris Pertama):")
            st.dataframe(df_input_original.head(), use_container_width=True)

            if st.button("🚀 Jalankan Analisis pada Seluruh Dataset", type="primary", use_container_width=True):
                with st.spinner("Mempersiapkan data dan menjalankan analisis... Ini mungkin memakan waktu. ⏳"):
                    # 1. Pra-pemrosesan dataset input
                    df_processed_for_model = preprocess_dataframe(df_input_original.copy())
                    
                    # (Opsional) Tampilkan data yang sudah diproses untuk debugging
                    # st.subheader("Data Setelah Pra-pemrosesan (untuk Model):")
                    # st.dataframe(df_processed_for_model.head(), use_container_width=True)

                    if df_processed_for_model.empty:
                        st.warning("Tidak ada data yang bisa diproses. Periksa format dataset Anda.")
                    else:
                        # 2. Lakukan prediksi pada data yang sudah diproses
                        df_prediksi_hasil = model_prediksi_ancaman_dataset(model, df_processed_for_model)

                        # 3. Gabungkan hasil prediksi dengan data input asli untuk ditampilkan
                        # Pastikan indeks cocok jika ingin menggabungkan
                        df_tampilan_akhir = pd.concat([df_input_original.reset_index(drop=True),
                                                       df_prediksi_hasil.reset_index(drop=True)], axis=1)

                        st.subheader("📊 Hasil Analisis Lengkap per Data:")
                        # Pilih kolom yang relevan untuk ditampilkan
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
                            data_grafik_status = pd.DataFrame({
                                'Status': ['Terancam', 'Aman'],
                                'Jumlah': [jumlah_terancam, jumlah_aman]
                            })
                            st.bar_chart(data_grafik_status.set_index('Status'), color=["#FF4B4B", "#3DDC97"])

                            if jumlah_terancam > 0:
                                st.write("Distribusi Tingkat Kepercayaan untuk Data Terancam:")
                                data_terancam_probs = df_tampilan_akhir[df_tampilan_akhir['Status Deteksi'] == 'Terancam']['Tingkat Kepercayaan Ancaman (%)']
                                st.bar_chart(data_terancam_probs, color="#FF4B4B")
                            
                            # Tambahan statistik deskriptif untuk salah satu fitur numerik
                            # Bisa diperluas untuk fitur lain atau visualisasi lain
                            if 'num__network_packet_size' in df_tampilan_akhir.columns:
                                st.subheader("Analisis 'Ukuran Paket Jaringan (KB)':")
                                fig, ax = plt.subplots() # Menggunakan matplotlib untuk kustomisasi lebih
                                sns.histplot(data=df_tampilan_akhir, x='num__network_packet_size', hue='Status Deteksi', kde=True, ax=ax, palette={"Aman":"#3DDC97", "Terancam":"#FF4B4B"})
                                ax.set_title('Distribusi Ukuran Paket Jaringan berdasarkan Status Deteksi')
                                st.pyplot(fig)


        except pd.errors.ParserError:
            st.error("Gagal mem-parsing file. Pastikan format CSV/Excel benar dan tidak korup.")
        except KeyError as e:
            st.error(f"Kolom yang dibutuhkan tidak ditemukan di dataset: {e}. "
                       "Mohon periksa nama kolom di file Anda sesuai dengan yang diinstruksikan di sidebar.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.error("Pastikan format file dan nama kolom sudah benar.")
    else:
        st.info("Silakan unggah dataset (file .csv atau .xlsx) melalui panel di sebelah kiri untuk memulai analisis.")

else:
    st.warning("Model tidak berhasil dimuat. Fungsi analisis tidak tersedia.")

st.markdown("---")
st.caption("Dashboard IDS v0.3 | Dibuat dengan Streamlit")

# Untuk grafik tambahan, mungkin perlu import:
import matplotlib.pyplot as plt
import seaborn as sns
