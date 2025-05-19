import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('Streamlit_Bank_Loan_Prediction-master/Streamlit_Bank_Loan_Prediction-master/Model/ML_Model.pkl', 'rb'))

SAVE_FILE = 'loan_applications.csv'

# Mapping dictionaries
duration_map = {
    '2 Month': 60,
    '6 Month': 180,
    '8 Month': 240,
    '1 Year': 360,
    '16 Month': 480
}
gender_map = {'Female': 0, 'Male': 1}
married_map = {'No': 0, 'Yes': 1}
dependents_map = {'No': 0, 'One': 1, 'Two': 2, 'More than Two': 3}
education_map = {'Not Graduate': 0, 'Graduate': 1}
employment_map = {'Job': 0, 'Business': 1}
credit_map = {'Between 300 to 500': 0, 'Above 500': 1}
property_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}

st.set_page_config(page_title="Bank Loan Predictor", layout="wide", page_icon="💰")

# Encodes user inputs
def encode_input(data):
    return [
        gender_map[data['Gender']],
        married_map[data['Married']],
        dependents_map[data['Dependents']],
        education_map[data['Education']],
        employment_map[data['Employment']],
        data['Income'],
        data['CoIncome'],
        data['LoanAmount'],
        duration_map[data['Duration']],
        credit_map[data['CreditScore']],
        property_map[data['PropertyArea']]
    ]

# Save single prediction to CSV
def save_input(data):
    df = pd.DataFrame([data])
    if os.path.exists(SAVE_FILE):
        df.to_csv(SAVE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(SAVE_FILE, index=False)

# Single prediction
def predict_single(data):
    features = [encode_input(data)]
    return model.predict(features)[0]

# Render logo
def render_logo():
    img = Image.open('logo bank.jpeg').resize((160, 145))
    st.image(img, use_column_width=False)

# Simple interest calculator function
def simple_interest(principal, rate, time):
    return (principal * rate * time) / 100

def main():
    # Sidebar Navigation
    st.sidebar.title("💼 Navigation")
    choice = st.sidebar.radio("Go to", ['🏠 Home', '📂 Batch Prediction', '📊 Insights', '💵 Interest Calculator'])

    if choice == '🏠 Home':
        render_logo()
        st.markdown("## 💰 Bank Loan Prediction")
        st.markdown("Predict whether a customer is eligible for a loan based on their details.")
        st.markdown("---")

        with st.form("loan_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Full Name")
                account_no = st.text_input("🔢 Account Number")
                gender = st.selectbox("👫 Gender", list(gender_map.keys()))
                married = st.selectbox("💍 Marital Status", list(married_map.keys()))
                dependents = st.selectbox("👶 Dependents", list(dependents_map.keys()))
                education = st.selectbox("🎓 Education", list(education_map.keys()))

            with col2:
                employment = st.selectbox("💼 Employment", list(employment_map.keys()))
                property_area = st.selectbox("🏘️ Property Area", list(property_map.keys()))
                credit_score = st.selectbox("💳 Credit Score", list(credit_map.keys()))
                income = st.number_input("📈 Monthly Income ($)", min_value=0)
                co_income = st.number_input("📉 Co-applicant Income ($)", min_value=0)
                loan_amount = st.number_input("💵 Loan Amount", min_value=0)
                duration = st.selectbox("⏱️ Loan Duration", list(duration_map.keys()))

            submitted = st.form_submit_button("Submit")

        if submitted:
            if not name.strip() or not account_no.strip():
                st.warning("⚠️ Please fill in both name and account number.")
            else:
                user_data = {
                    'AccountNumber': account_no,
                    'Name': name,
                    'Gender': gender,
                    'Married': married,
                    'Dependents': dependents,
                    'Education': education,
                    'Employment': employment,
                    'PropertyArea': property_area,
                    'CreditScore': credit_score,
                    'Income': income,
                    'CoIncome': co_income,
                    'LoanAmount': loan_amount,
                    'Duration': duration
                }

                try:
                    result = predict_single(user_data)
                    save_input(user_data)
                    if result == 1:
                        st.success(f"🎉 Hello {name}, your loan has been approved!")
                    else:
                        st.error(f"🙁 Sorry {name}, your loan has not been approved.")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    elif choice == '📂 Batch Prediction':
        st.header("📂 Batch Loan Prediction from CSV")
        st.markdown("Upload a CSV with multiple applicants to predict loan approval in bulk.")
        sample = pd.DataFrame([{
            'Gender': 'Male', 'Married': 'Yes', 'Dependents': 'Two',
            'Education': 'Graduate', 'Employment': 'Job', 'Income': 5000,
            'CoIncome': 1000, 'LoanAmount': 150, 'Duration': '1 Year',
            'CreditScore': 'Above 500', 'PropertyArea': 'Urban'
        }])
        st.download_button("📥 Download Sample CSV", sample.to_csv(index=False), "sample_loan_data.csv")

        uploaded_file = st.file_uploader("📤 Upload your CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df['Prediction'] = df.apply(lambda row: model.predict([encode_input(row)])[0], axis=1)
                df['Prediction Result'] = df['Prediction'].map({1: 'Approved ✅', 0: 'Rejected ❌'})
                st.dataframe(df)

                st.download_button("📥 Download Predictions", df.to_csv(index=False), "loan_predictions.csv")
            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")

    elif choice == '📊 Insights':
        st.header("📊 Insights from Submitted Applications")

        if os.path.exists(SAVE_FILE):
            df = pd.read_csv(SAVE_FILE)
            if not df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 💰 Income Distribution")
                    fig, ax = plt.subplots()
                    df['Income'].hist(bins=20, color='skyblue', edgecolor='black')
                    ax.set_xlabel("Monthly Income")
                    st.pyplot(fig)

                with col2:
                    st.markdown("### 🏘️ Loan Amount vs Income")
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(df['Income'], df['LoanAmount'], alpha=0.6, color='orange')
                    ax2.set_xlabel("Income")
                    ax2.set_ylabel("Loan Amount")
                    st.pyplot(fig2)
            else:
                st.info("📂 No applications submitted yet.")
        else:
            st.info("📂 No applications submitted yet.")

    elif choice == '💵 Interest Calculator':
        st.header("💵 Simple Interest Calculator")
        st.markdown("Calculate the Simple Interest based on principal amount, rate of interest, and time.")

        with st.form("interest_form"):
            principal = st.number_input("💰 Principal Amount ($)", min_value=0.0, format="%.2f")
            rate = st.number_input("📈 Annual Interest Rate (%)", min_value=0.0, format="%.2f")
            time = st.number_input("⏳ Time Period (in years)", min_value=0.0, format="%.2f")
           
            calculate = st.form_submit_button("Calculate Interest")

        if calculate:
            if principal == 0 or rate == 0 or time == 0:
                st.warning("⚠️ Please fill out all the fields with non-zero values.")
            else:
                interest = simple_interest(principal, rate, time)
                total_amount = principal + interest

                st.success(f"💵 Simple Interest: ${interest:.2f}")
                st.info(f"📊 Total Amount Payable: ${total_amount:.2f}")

main()
