import streamlit as st
import joblib
import numpy as np


model = joblib.load("gbc_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction", layout="centered")


st.markdown("<h1 style='text-align: center;'> Customer Churn Prediction</h1>", unsafe_allow_html=True)

st.markdown("---")


st.header("Customer Info")

credit_score = st.slider("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=50000.0)
salary = st.number_input("Estimated Salary", value=50000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

is_active = st.selectbox("Active Member", [1, 0])
has_card = st.selectbox("Has Credit Card", [1, 0])
products = st.selectbox("Number of Products", [1, 2, 3, 4])

st.markdown("## Prediction Result")


geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0


input_data = np.array([[credit_score, age, tenure, balance, products,
                        has_card, is_active, salary,
                        geo_germany, geo_spain, gender_male]])


input_data[:, :4] = scaler.transform(input_data[:, :4])


if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error(f"⚠️ Customer likely to churn")
    else:
        st.success(f"✅ Customer likely to stay")

st.markdown("---")