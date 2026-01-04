from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent



# App Initialization

app = FastAPI(
    title="UNSW ML-WAF Inference API",
    description="Multiclass RandomForest-based WAF (UNSW-NB15)",
    version="1.0"
)


# Load Model 

rf_model = joblib.load(BASE_DIR/"Model/rf_multiclass_unsw.pkl")
scaler = joblib.load(BASE_DIR/"Model/minmaxscaler.pkl")
label_classes = np.load(BASE_DIR/"Model/le2_classes.npy", allow_pickle=True)

print("Model loaded")
print("Expected features:", rf_model.n_features_in_)
print("Classes:", label_classes)


# FEATURE ORDER 

FEATURE_ORDER = [
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


# Input Schema

class UNSWInput(BaseModel):
    dttl: float
    swin: float
    dwin: float
    tcprtt: float
    synack: float
    ackdat: float
    proto_tcp: float
    proto_udp: float
    service_dns: float
    state_FIN: float


# Prediction Endpoint

@app.post("/predict")
def predict(packet: UNSWInput):

    # Convert input to numpy array 
    x = np.array([[getattr(packet, f) for f in FEATURE_ORDER]])

    # Scale
    x_scaled = scaler.transform(x)

    # Predict
    pred_idx = rf_model.predict(x_scaled)[0]
    pred_proba = rf_model.predict_proba(x_scaled).max()

    pred_label = label_classes[pred_idx]

    # WAF Decision
    decision = "ALLOW" if pred_label == "Normal" else "BLOCK"

    return {
        "prediction": str(pred_label),
        "decision": decision,
        "confidence": float(pred_proba)
    }
