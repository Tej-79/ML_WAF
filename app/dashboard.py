import streamlit as st
import pandas as pd
import requests
from streamlit_autorefresh import st_autorefresh
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = BASE_DIR / "Dataset" / "Testing.csv"
LOG_FILE = BASE_DIR / "Dataset" / "log.csv"
INTERVAL_MS = 1000

FEATURES = [
    "dttl",
    "swin",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "proto_tcp",
    "proto_udp",
    "service_dns",
    "state_FIN"
]


# PAGE SETUP

st.set_page_config(page_title="Indian Navy ML-WAF Dashboard", layout="wide")
st.title("ML-Based Web Application Firewall")
st.caption("Live Traffic • ML Decisions • Secure Logging")


# SESSION STATE

if "running" not in st.session_state:
    st.session_state.running = False

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "allow" not in st.session_state:
    st.session_state.allow = 0

if "block" not in st.session_state:
    st.session_state.block = 0

# persistent log storage
if "log_df" not in st.session_state:
    if os.path.exists(LOG_FILE):
        st.session_state.log_df = pd.read_csv(LOG_FILE)
    else:
        st.session_state.log_df = pd.DataFrame(
            columns=["Index", "Ground Truth", "Prediction", "Decision", "Confidence"]
        )


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    df["proto_tcp"] = 0
    df["proto_udp"] = 0
    df["service_dns"] = 0
    df["state_FIN"] = 0

    df.loc[df["proto"] == "tcp", "proto_tcp"] = 1
    df.loc[df["proto"] == "udp", "proto_udp"] = 1
    df.loc[df["service"] == "dns", "service_dns"] = 1
    df.loc[df["state"] == "FIN", "state_FIN"] = 1

    return df

df = load_data()
df_model = df[FEATURES]


# CONTROLS

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Start Live Traffic"):
        st.session_state.running = True

with c2:
    if st.button("Stop"):
        st.session_state.running = False

with c3:
    st.metric("Processed Packets", len(st.session_state.log_df))

st.divider()


# METRICS

m1, m2, m3 = st.columns(3)
m1.metric("ALLOW", st.session_state.allow)
m2.metric("BLOCK", st.session_state.block)
m3.metric("TOTAL", st.session_state.allow + st.session_state.block)

st.divider()


# PLACEHOLDERS

decision_box = st.empty()
table_box = st.empty()


# AUTO REFRESH

if st.session_state.running:
    st_autorefresh(interval=INTERVAL_MS, key="traffic_refresh")


# PROCESS ONE PACKET

if st.session_state.running and st.session_state.idx < len(df_model):

    i = st.session_state.idx
    row = df_model.iloc[i]

    res = requests.post(API_URL, json=row.to_dict()).json()

    decision = res["decision"]
    prediction = res["prediction"]
    confidence = round(res["confidence"], 2)
    gt = df.loc[i, "attack_cat"]

    if decision == "ALLOW":
        st.session_state.allow += 1
    else:
        st.session_state.block += 1

    log_row = {
        "Index": i,
        "Ground Truth": gt,
        "Prediction": prediction,
        "Decision": decision,
        "Confidence": confidence
    }

    # append incrementally 
    st.session_state.log_df.loc[len(st.session_state.log_df)] = log_row
    st.session_state.log_df.to_csv(LOG_FILE, index=False)

    # DECISION CARD (CALM UPDATE)

    with decision_box:
        bg = "#d4edda" if decision == "ALLOW" else "#f8d7da"
        fg = "#155724" if decision == "ALLOW" else "#721c24"
        status = "ALLOWED" if decision == "ALLOW" else "BLOCKED"

        st.markdown(
            f"""
            <div style="
                background-color:{bg};
                padding:14px;
                border-radius:8px;
                transition: all 0.3s ease-in-out;
            ">
                <h4 style="color:{fg}; margin-bottom:6px;">
                    Packet #{i} — {status}
                </h4>
                <div style="font-size:14px;">
                    <b>Prediction:</b> {prediction}<br>
                    <b>Confidence:</b> {confidence}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.session_state.idx += 1

# LOG TABLE 

with table_box:
    if not st.session_state.log_df.empty:
        st.dataframe(
            st.session_state.log_df.tail(15),
            use_container_width=True,
            height=380
        )


# PERFORMANCE METRICS

st.divider()
st.subheader("Model Performance (Live Replay)")

if not st.session_state.log_df.empty:

    perf_df = st.session_state.log_df

    y_true = perf_df["Ground Truth"].apply(
        lambda x: "Normal" if x == "Normal" else "Attack"
    )
    y_pred = perf_df["Decision"].apply(
        lambda x: "Normal" if x == "ALLOW" else "Attack"
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Attack")
    rec = recall_score(y_true, y_pred, pos_label="Attack")
    f1 = f1_score(y_true, y_pred, pos_label="Attack")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1-score", f"{f1:.2f}")

    cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Attack"])

    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 8}
    )

    ax.set_title("Confusion Matrix", fontsize=9, pad=6)
    ax.set_xticklabels(["Pred Normal", "Pred Attack"], fontsize=8)
    ax.set_yticklabels(["True Normal", "True Attack"], fontsize=8, rotation=0)

    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=False)
