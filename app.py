"""Network Anomaly Detector — Streamlit Dashboard. Run: streamlit run app.py"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_generator import generate_dataset
from src.detectors import get_detector
from src.evaluator import evaluate

st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛡️", layout="wide")
st.sidebar.title("🛡️ Network Anomaly Detector")
mode = st.sidebar.radio("Mode", ["Demo (synthetic data)", "Upload CSV", "About"])
method = st.sidebar.selectbox("Detection method", ["hybrid", "statistical", "isolation_forest", "rule_based"])
with st.sidebar.expander("Advanced"):
    z_threshold = st.slider("Statistical threshold", 2.0, 6.0, 3.5, 0.1)
    contamination = st.slider("IForest contamination", 0.01, 0.20, 0.05, 0.01)
    min_votes = st.slider("Hybrid min votes", 1, 3, 2)

@st.cache_data
def load_or_generate(n=4000, ratio=0.05, seed=42):
    return generate_dataset(n_samples=n, anomaly_ratio=ratio, seed=seed)

def run_detection(df, method, z_threshold, contamination, min_votes):
    kwargs = {}
    if method == "statistical": kwargs["threshold"] = z_threshold
    elif method == "isolation_forest": kwargs["contamination"] = contamination
    elif method == "hybrid": kwargs.update({"z_threshold": z_threshold, "contamination": contamination, "min_votes": min_votes})
    return get_detector(method, **kwargs).fit_predict(df)

st.title("🛡️ Network Anomaly Detector")
st.caption("Statistical · Isolation Forest · Rule-based · Hybrid")

if mode == "About":
    st.markdown("Detect anomalous network flows with four complementary methods. Built for defensive cybersecurity research.")
    st.stop()

if mode == "Demo (synthetic data)":
    n_samples = st.slider("Flows", 1000, 10000, 4000, 500)
    anomaly_ratio = st.slider("Anomaly ratio", 0.01, 0.15, 0.05, 0.01)
    if st.button("Generate & Detect", type="primary") or "df" not in st.session_state:
        st.session_state["df"] = load_or_generate(n_samples, anomaly_ratio)
        st.session_state["has_labels"] = True
    df = st.session_state["df"]
else:
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a flow CSV with packet_count, byte_count, etc.")
        st.stop()
    df = pd.read_csv(up)
    st.session_state["has_labels"] = "is_anomaly" in df.columns

has_labels = st.session_state.get("has_labels", False)
result = run_detection(df, method, z_threshold, contamination, min_votes)
df_view = df.copy()
df_view["anomaly"] = result.labels
df_view["anomaly_score"] = result.scores
n_anom = int(result.labels.sum())
pct = 100 * n_anom / len(df) if len(df) else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total flows", f"{len(df):,}")
k2.metric("Anomalies", f"{n_anom:,}", f"{pct:.1f}%")
k3.metric("Method", method.replace("_", " ").title())
if has_labels:
    metrics = evaluate(df["is_anomaly"].values, result.labels, result.scores)
    k4.metric("F1", f"{metrics['f1']:.3f}")
else:
    k4.metric("Max score", f"{result.scores.max():.3f}")

tab1, tab2, tab3 = st.tabs(["Overview", "Top Anomalies", "Data"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = pd.Series(result.labels).value_counts().reindex([0, 1], fill_value=0)
        st.plotly_chart(px.pie(names=["Normal", "Anomaly"], values=counts.values,
            color_discrete_sequence=["#2ecc71", "#e74c3c"], title="Breakdown"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df_view, x="anomaly_score",
            color=df_view["anomaly"].map({0: "Normal", 1: "Anomaly"}), nbins=40,
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"}, opacity=0.7),
            use_container_width=True)
    if has_labels:
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{metrics['precision']:.3f}")
        m2.metric("Recall", f"{metrics['recall']:.3f}")
        m3.metric("F1", f"{metrics['f1']:.3f}")
with tab2:
    top = df_view[df_view["anomaly"] == 1].nlargest(15, "anomaly_score")
    cols = [c for c in ["src_ip", "dst_ip", "dst_port", "protocol", "packet_count", "byte_count", "anomaly_score"] if c in top.columns]
    st.dataframe(top[cols] if cols else top, use_container_width=True)
with tab3:
    st.dataframe(df_view.head(300), use_container_width=True)
    st.download_button("Download results CSV", df_view.to_csv(index=False).encode(), "anomaly_results.csv", "text/csv")
