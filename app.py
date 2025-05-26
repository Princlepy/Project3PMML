import streamlit as st
import pandas as pd
import numpy as np

# Judul dan deskripsi
st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.title("📡 Intrusion Detection System Dashboard")
st.markdown("Dashboard sederhana untuk mendeteksi aktivitas jaringan mencurigakan pada UMKM e-commerce.")

# Sidebar untuk input manual
st.sidebar.header("🛠️ Input Fitur")
duration = st.sidebar.slider("Durasi koneksi (ms)", 0, 10000, 500)
protocol_type = st.sidebar.selectbox("Protocol", ["tcp", "udp", "icmp"])
service = st.sidebar.selectbox("Service", ["http", "ftp", "smtp", "domain_u"])
flag = st.sidebar.selectbox("Flag", ["SF", "S0", "REJ", "RSTR"])

# Contoh input numerik lainnya
src_bytes = st.sidebar.number_input("Jumlah byte dari sumber", 0, step=1)
dst_bytes = st.sidebar.number_input("Jumlah byte ke tujuan", 0, step=1)

# Tombol prediksi (sementara dummy)
if st.sidebar.button("🔍 Deteksi Sekarang"):
    # Dummy prediksi
    hasil = np.random.choice(["Normal", "Mencurigakan"], p=[0.7, 0.3])
    
    st.subheader("🧾 Hasil Deteksi")
    if hasil == "Normal":
        st.success("✅ Aktivitas terdeteksi sebagai NORMAL")
    else:
        st.error("🚨 Aktivitas MENCURIGAKAN terdeteksi!")

# (Opsional) Tampilkan tabel data dummy
st.markdown("---")
st.subheader("📊 Contoh Data Traffic Log")
data_dummy = pd.DataFrame({
    "durasi": np.random.randint(0, 10000, 10),
    "protocol": np.random.choice(["tcp", "udp", "icmp"], 10),
    "service": np.random.choice(["http", "ftp", "smtp", "domain_u"], 10),
    "flag": np.random.choice(["SF", "S0", "REJ", "RSTR"], 10),
    "src_bytes": np.random.randint(0, 5000, 10),
    "dst_bytes": np.random.randint(0, 5000, 10),
    "label": np.random.choice(["Normal", "Mencurigakan"], 10)
})
st.dataframe(data_dummy)

# Footer
st.markdown("---")
st.caption("© 2025 - IDS Dashboard by Mas dan Lya ✨")
