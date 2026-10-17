import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Demo", page_icon="💳")
st.title("Credit default risk predictor")
st.caption("Educational demo built on the UCI credit card default dataset.")

@st.cache_resource
def load_model():
    return joblib.load("credit_model.joblib")

model = load_model()

limit_bal = st.number_input("Credit limit (NT$)", 10000, 1000000, 100000, step=10000)
age = st.slider("Age", 21, 79, 35)
sex = st.selectbox("Sex", [1, 2], format_func=lambda x: {1: "Male", 2: "Female"}[x])
education = st.selectbox("Education", [1, 2, 3, 4],
    format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}[x])
marriage = st.selectbox("Marital status", [1, 2, 3],
    format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

st.subheader("Repayment status per month")
st.caption("-2 = no consumption, -1 = paid in full, 0 = minimum payment, 1+ = months late")
months = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
pay_values = [st.selectbox(m, list(range(-2, 9)), index=2) for m in months]

avg_bill = st.number_input("Average monthly bill (NT$)", 0, 1000000, 50000, step=5000)
avg_paid = st.number_input("Average monthly payment (NT$)", 0, 1000000, 5000, step=1000)
threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

if st.button("Predict"):
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    row = {"LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
           "MARRIAGE": marriage, "AGE": age}
    row.update(dict(zip(pay_cols, pay_values)))
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = avg_bill
        row[f"PAY_AMT{i}"] = avg_paid
    X_new = pd.DataFrame([row])

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    X_new["avg_delay"] = X_new[pay_cols].mean(axis=1)
    X_new["max_delay"] = X_new[pay_cols].max(axis=1)
    X_new["months_late"] = (X_new[pay_cols] > 0).sum(axis=1)
    X_new["avg_bill"] = X_new[bill_cols].mean(axis=1)
    X_new["total_paid"] = X_new[amt_cols].sum(axis=1)
    X_new["utilization"] = X_new["avg_bill"] / X_new["LIMIT_BAL"]
    X_new["pay_ratio"] = (X_new["total_paid"] / X_new[bill_cols].sum(axis=1).clip(lower=1)).clip(upper=5)

    proba = model.predict_proba(X_new)[0, 1]
    st.metric("Estimated default probability", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= threshold:
        st.error("Flagged as HIGH risk at this threshold")
    else:
        st.success("Flagged as LOW risk at this threshold")