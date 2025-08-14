import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("transaction_anomalies_dataset.csv")

data = load_data()

# Features used for anomaly detection
FEATURES = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
X = data[FEATURES]

# Train or load model and scaler
model_path = "model.pkl"
scaler_path = "scaler.pkl"

def train_and_save_model():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler

def load_model_and_scaler():
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        model, scaler = train_and_save_model()
    return model, scaler

model, scaler = load_model_and_scaler()

# UI
st.title("🔍 Transaction Anomaly Detection (Improved)")

st.markdown("Enter transaction details to detect potential anomalies.")

with st.form("anomaly_form"):
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=10.0)
    avg_transaction = st.number_input("Average Transaction Amount", min_value=0.0, value=950.0, step=10.0)
    frequency = st.number_input("Frequency of Transactions (per month)", min_value=0.0, value=15.0, step=1.0)
    threshold = st.slider("Anomaly Score Threshold (adjust sensitivity)", -0.2, 0.1, -0.02, 0.01)
    submitted = st.form_submit_button("Check Anomaly")

if submitted:
    input_data = pd.DataFrame([[transaction_amount, avg_transaction, frequency]], columns=FEATURES)
    input_scaled = scaler.transform(input_data)

    anomaly_score = model.decision_function(input_scaled)[0]
    prediction = anomaly_score < threshold

    st.subheader("🔎 Result:")
    if prediction:
        st.error(f"⚠️ ALERT: This transaction is flagged as ANOMALOUS!\n\n**Anomaly Score:** {anomaly_score:.4f}")
    else:
        st.success(f"✅ This transaction appears NORMAL.\n\n**Anomaly Score:** {anomaly_score:.4f}")

st.markdown("---")
st.markdown("ℹ️ Adjust the slider above to make the model more or less sensitive to unusual behavior.")
