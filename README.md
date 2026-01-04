# ML-Based Web Application Firewall (ML-WAF)

A Machine Learning–driven Web Application Firewall designed to detect malicious network traffic in real time using behavioral features.  
The system integrates a trained RandomForest model with a FastAPI inference backend and an interactive Streamlit dashboard for live traffic analysis, decision visualization, and logging.

---

## Problem Statement

Traditional rule-based Web Application Firewalls (WAFs) rely heavily on static signatures and predefined rules. While effective for known threats, they suffer from:

- High false alarm rates on legitimate traffic  
- Inability to detect zero-day or unseen attack patterns  
- Limited adaptability to evolving attack behaviors  

This project addresses these limitations by leveraging **machine learning–based behavioral analysis** instead of static rules.

---

## Proposed Solution

We propose an **ML-based WAF** that:

- Learns traffic behavior from the UNSW-NB15 dataset  
- Classifies traffic using a trained RandomForest model  
- Converts multiclass predictions into actionable WAF decisions (ALLOW / BLOCK)  
- Provides a live dashboard for monitoring, logging, and performance evaluation  

The system is designed to be **modular, explainable, and extensible**.

---

## System Architecture

**High-level flow:**

1. Network traffic samples are replayed from a dataset  
2. Feature vectors are sent to a FastAPI inference service  
3. The ML model predicts the traffic class  
4. Predictions are mapped to WAF decisions (ALLOW / BLOCK)  
5. Results are visualized in real time on a Streamlit dashboard  

---

## Machine Learning Model

- **Dataset:** UNSW-NB15  
- **Model:** RandomForest Classifier (multiclass)  
- **Features:** Selected behavioral and protocol-level features  
- **Decision Logic:**  
  - Normal → ALLOW  
  - Any attack class → BLOCK  

The model is trained to balance attack detection while minimizing false alarms on legitimate traffic.

> **Note:**  
> Trained model binaries (`.pkl`) are excluded from this repository due to GitHub file size limits.  
> The full inference pipeline is provided, and the model can be loaded locally for execution.

---

## Live Dashboard Features

The Streamlit dashboard provides:

- **Live Traffic Replay** – packet-by-packet simulation  
- **Decision Card** – current packet decision with confidence  
- **Traffic Log Table** – incremental logging of past packets  
- **Performance Metrics** – accuracy, precision, recall, F1-score  
- **Confusion Matrix** – WAF-level evaluation (Normal vs Attack)  

The dashboard is designed to resemble a **Security Operations Center (SOC)** monitoring interface.

---

## Zero-Day and Bot Attack Handling

- The system does not rely on static signatures.  
- Detection is based on **behavioral deviations from learned normal traffic patterns**.  
- Unknown or unseen attack behaviors are flagged as malicious if they significantly differ from normal traffic.  
- Explicit bot labeling is not implemented due to dataset limitations and is identified as future work.

---

## Tech Stack used 

- Python
- Scikit-learn
- FastAPI
- Streamlit
- Pandas / NumPy

---

## How to Run the Project

### 1. Install dependencies
pip install -r requirements.txt

### 2. Start FastAPI backend
- From the project root in first terminal:
- python -m uvicorn app.main:app --reload
- API will be available at :http://127.0.0.1:8000

### 3. Start Streamlit 
- From the project root in Second terminal:
- python -m streamlit run app/dashboard.py

---

## Run Live Traffic Simulation

- Open the dashboard in browser
- Click Start Live Traffic
- Observe real-time decisions, logs, and metrics

---

> **Integration Note:**
    >The FastAPI inference endpoint can be invoked by external WAF engines to make real-time ALLOW/BLOCK decisions.

---

## Results

- High accuracy on replayed UNSW-NB15 traffic

- Effective separation of normal and attack traffic at WAF level

- Reduced false alarms through feature selection and model tuning

- Performance is visualized live using metrics and a confusion matrix.

---

## Limitations

- Model binary not included in the repository
- Real network packet capture is not integrated (dataset replay used)
- Bot traffic is not explicitly labeled in the dataset

---

## Future Work

- Integration with live packet capture (PCAP / network interface)
- Threshold-based confidence tuning for false alarm reduction
- Explicit bot and DDoS traffic classification
- Deployment using Docker and reverse proxy integration

---

## Conclusion

- This project demonstrates how machine learning can enhance traditional Web Application Firewalls by enabling behavioral detection, adaptability to unknown attacks, and real-time visibility.
- The modular architecture allows easy extension toward production-ready deployments.