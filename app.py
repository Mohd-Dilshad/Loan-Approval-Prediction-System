

import streamlit as st
import pickle
import pandas as pd


# 1. LOAD TRAINED MODEL
with open("model/loan_model.pkl", "rb") as file:
    model = pickle.load(file)


# 2. PAGE SETTINGS
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)


# 3. TITLE
st.title("🏦 Loan Approval Prediction System")
st.write("Enter customer details to predict loan approval.")


# 4. CUSTOMER DETAILS
st.subheader("👤 Customer Details")
col1, col2 = st.columns(2)
with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=5,
        value=0
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=200
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )



# 5. PREDICTION BUTTON
st.divider()

if st.button("🔍 Predict Loan Status", use_container_width=True):

    # Convert categorical values
    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    education = 1 if education == "Graduate" else 0

    self_employed = 1 if self_employed == "Yes" else 0


    # Create input DataFrame
    input_data = pd.DataFrame({

        "Gender": [gender],

        "Married": [married],

        "Dependents": [dependents],

        "Education": [education],

        "Self_Employed": [self_employed],

        "ApplicantIncome": [applicant_income],

        "LoanAmount": [loan_amount],

        "Credit_History": [credit_history],

        "Property_Area_Rural": [
            1 if property_area == "Rural" else 0
        ],

        "Property_Area_Semiurban": [
            1 if property_area == "Semiurban" else 0
        ],

        "Property_Area_Urban": [
            1 if property_area == "Urban" else 0
        ]
    })


    # 6. PREDICTION
    prediction = model.predict(input_data)


    # 7. RESULT
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")