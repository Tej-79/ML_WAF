import pandas as pd
import requests
import time
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = BASE_DIR/"Dataset/Testing.csv"
DELAY = 0.5

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

df = pd.read_csv(CSV_PATH)

# Reconstruct one-hot features

df["proto_tcp"] = 0
df["proto_udp"] = 0
df["service_dns"] = 0
df["state_FIN"] = 0

# encode
df.loc[df["proto"] == "tcp", "proto_tcp"] = 1
df.loc[df["proto"] == "udp", "proto_udp"] = 1
df.loc[df["service"] == "dns", "service_dns"] = 1
df.loc[df["state"] == "FIN", "state_FIN"] = 1

# Keep only model features

df_model = df[FEATURES]

print("Replay started...")

# Replay loop
for i, row in df_model.iterrows():

    payload = row.to_dict()

    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        res = r.json()

        gt = df.loc[i, "attack_cat"] if "attack_cat" in df.columns else "NA"

        print(
            f"[{i}] "
            f"GT: {gt:<15} | "
            f"PRED: {res['prediction']:<15} | "
            f"DECISION: {res['decision']:<5} | "
            f"CONF: {res['confidence']:.2f}"
        )

    except Exception as e:
        print(f"[{i}] ERROR:", e)

    time.sleep(DELAY)
