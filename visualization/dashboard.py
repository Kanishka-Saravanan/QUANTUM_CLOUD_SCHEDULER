import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

pio.templates.default = "plotly_dark"

st.set_page_config(page_title="Q-Cloud Scheduler", page_icon="⚛️", layout="wide")

# ================= UI =================
st.markdown("""
<style>
    .main, .stApp, .block-container { background-color: #0a0e17 !important; }
    .quantum-text { color: #00f5ff; text-shadow: 0 0 15px #00f5ff; }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Q-Cloud Scheduler")
st.markdown('<h3 class="quantum-text">Hybrid Quantum-Classical Cloud Task Scheduling Dashboard</h3>', unsafe_allow_html=True)

# ================= Sidebar =================
st.sidebar.header("🔧 Controls")

if st.sidebar.button("🚀 Run Experiment Pipeline"):
    with st.spinner("Running..."):
        os.system("python main.py")
    st.success("✅ Done!")

# ================= Load Results =================
RESULT_FILE = "results/experiment_results.json"

if not os.path.exists(RESULT_FILE):
    st.error("❌ No results found. Run pipeline first.")
    st.stop()

with open(RESULT_FILE) as f:
    raw_data = json.load(f)

# Convert list → dict
data = {f"Workload_{i+1}": item for i, item in enumerate(raw_data)}

# ================= Workload Selection =================
selected_workload = st.sidebar.selectbox("Select Workload", list(data.keys()))
results = data[selected_workload]["algorithm_results"]

# ================= Flatten Data =================
flat_data = []

for algo, values in results.items():
    cost = values.get("cost", {})
    perf = values.get("performance", {})
    sla = values.get("sla", {})

    flat_data.append({
        "Algorithm": algo,
        "Total Cost": cost.get("total_cost", 0),
        "Energy": cost.get("total_energy", 0),
        "Makespan": perf.get("makespan", 0),
        "Utilization": perf.get("mean_utilization", 0),
        "SLA Violations": sla.get("sla_violations", 0)
    })

df = pd.DataFrame(flat_data)

# ================= Metrics =================
st.subheader(f"📊 Results for {selected_workload}")

cols = st.columns(len(df.columns) - 1)

for i, metric in enumerate(df.columns[1:]):
    with cols[i]:
        if "Cost" in metric or "Violations" in metric:
            best = df.loc[df[metric].idxmin()]
            label = "↓ Best"
        else:
            best = df.loc[df[metric].idxmax()]
            label = "↑ Best"

        st.metric(metric, f"{best[metric]:.2f}", f"{best['Algorithm']} {label}")

# ================= Tabs =================
tab1, tab2, tab3 = st.tabs(["📊 Charts", "📈 Trends", "📋 Data"])

# -------- TAB 1 --------
with tab1:
    for metric in df.columns[1:]:
        fig = px.bar(df, x="Algorithm", y=metric, color="Algorithm", title=metric)
        st.plotly_chart(fig, use_container_width=True)

# -------- TAB 2 --------
with tab2:
    all_rows = []

    for wl, exp in data.items():
        for algo, val in exp["algorithm_results"].items():
            all_rows.append({
                "Workload": wl,
                "Algorithm": algo,
                "Cost": val["cost"]["total_cost"],
                "Makespan": val["performance"]["makespan"],
                "Utilization": val["performance"]["mean_utilization"]
            })

    big_df = pd.DataFrame(all_rows)

    metric = st.selectbox("Select Metric", ["Cost", "Makespan", "Utilization"])

    fig = px.line(big_df, x="Workload", y=metric, color="Algorithm", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# -------- TAB 3 --------
with tab3:
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")

# ================= Footer =================
st.caption("⚛️ Quantum Cloud Scheduler • 2026")