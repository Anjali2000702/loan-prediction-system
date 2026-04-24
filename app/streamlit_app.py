# import streamlit as st
# import requests

# st.title("Loan Default Prediction")

# st.write("Enter customer details:")

# # simple inputs (just demo for now)
# income = st.number_input("Income", value=200000.0)
# credit = st.number_input("Credit Amount", value=500000.0)
# annuity = st.number_input("Annuity", value=20000.0)

# if st.button("Predict"):
    
#     # simple dummy feature list (same size as model)
#     features = [0.0]*83
    
#     # put important values
#     features[5] = income
#     features[6] = credit
#     features[7] = annuity

#     response = requests.post(
#         "http://127.0.0.1:5000/predict",
#         json={"features": features}
#     )
    
#     result = response.json()
    
#     st.write("Prediction:", result["prediction"])
#     st.write("Probability:", result["probability"])



import streamlit as st
import requests
import joblib
import numpy as np
import os

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "lgb_model.pkl")

model = joblib.load(model_path)

st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

st.title("💰 Loan Default Prediction System")
st.markdown("### Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Income", value=200000.0)
    credit = st.number_input("Credit Amount", value=500000.0)

with col2:
    annuity = st.number_input("Annuity", value=20000.0)

if st.button("Predict Risk"):

    if st.button("Predict Risk"):

    features = [0.0]*83
    features[5] = income
    features[6] = credit
    features[7] = annuity

    with st.spinner("🔄 Predicting... please wait"):

        try:
            response = requests.post(
                "https://loan-api-z5me.onrender.com/predict",
                json={"features": features}
            )

            result = response.json()

            if result["prediction"] == 1:
                st.error(f"⚠️ High Risk of Default\n\nProbability: {result['probability']:.2f}")
            else:
                st.success(f"✅ Low Risk (Safe)\n\nProbability: {result['probability']:.2f}")

        except:
            st.error("❌ Server not responding. Please try again later.")