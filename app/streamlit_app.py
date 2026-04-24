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

import shap
import joblib
import numpy as np

#model = joblib.load("../models/lgb_model.pkl")
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "lgb_model.pkl")

model = joblib.load(model_path)
explainer = shap.Explainer(model)

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

    features = [0.0]*83
    features[5] = income
    features[6] = credit
    features[7] = annuity

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"features": features}
    )

    result = response.json()

    if result["prediction"] == 1:
        st.error(f"⚠️ High Risk of Default\nProbability: {result['probability']:.2f}")
    else:
        st.success(f"✅ Low Risk (Safe)\nProbability: {result['probability']:.2f}")
    #         # SHAP explanation
    # shap_values = explainer(np.array(features).reshape(1, -1))

    # st.markdown("### 🔍 Model Explanation")

    # shap.plots.waterfall(shap_values[0], show=False)
    # st.pyplot(bbox_inches='tight')
        # SHAP explanation
    import matplotlib.pyplot as plt

    shap_values = explainer(np.array(features).reshape(1, -1))

    st.markdown("### 🔍 Model Explanation")

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)