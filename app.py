import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # Need to import even if not used directly, for joblib

# --- Load All Artifacts ---
# Use @st.cache_resource to load models only once (new Streamlit standard)
@st.cache_resource
def load_artifacts():
    """
    Loads all the necessary .joblib files for the app.
    Returns:
        tuple: (preprocessor, classifier, regressor, label_encoder)
    """
    try:
        preprocessor = joblib.load('emi_preprocessor.joblib')
        classifier = joblib.load('emi_classifier_model.joblib')
        regressor = joblib.load('emi_regressor_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure all .joblib files are in the GitHub repository.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None
        
    return preprocessor, classifier, regressor, label_encoder

preprocessor, classifier, regressor, label_encoder = load_artifacts()

# --- Streamlit App UI ---
st.set_page_config(page_title="EMIPredict AI", layout="wide")
st.title("EMIPredict AI - Intelligent Financial Risk Assessment")
st.write("Enter the applicant's details to predict EMI eligibility and the maximum affordable EMI.")

# Check if models loaded
if preprocessor is None or classifier is None or regressor is None:
    st.stop()

# --- Input Form ---
# We need to get the feature names in the *exact* order the preprocessor was trained on.
try:
    # This retrieves the exact feature lists from the trained preprocessor
    original_num_std_features = preprocessor.transformers_[0][2]
    original_num_skewed_features = preprocessor.transformers_[1][2]
    original_cat_features = preprocessor.transformers_[2][2]
    
    # This is the final order of columns our DataFrame *must* have
    preprocessor_features = original_num_std_features + original_num_skewed_features + original_cat_features
    
except Exception as e:
    st.error(f"Error getting feature names from preprocessor: {e}. Using hardcoded list.")
    # Fallback to hardcoded list if the above fails
    # This MUST match the columns in the 'X' DataFrame from training
    preprocessor_features = [
        'age', 'years_of_employment', 'family_size', 'dependents',
        'credit_score', 'requested_tenure', 'monthly_salary', 'bank_balance',
        'emergency_fund', 'requested_amount', 'total_monthly_expenses',
        'disposable_income', 'dti_ratio', 'gender', 'marital_status',
        'education', 'employment_type', 'company_type', 'house_type',
        'existing_loans', 'emi_scenario'
    ]


st.header("Applicant Information")

# We'll use columns to organize the form
col1, col2, col3 = st.columns(3)

# Dictionary to hold user inputs
input_data = {}

# --- Define options for select boxes ---
gender_options = ['Male', 'Female']
marital_options = ['Single', 'Married']
education_options = ['High School', 'Graduate', 'Post Graduate', 'Professional']
employment_options = ['Private', 'Government', 'Self-employed']
company_options = ['Small', 'Medium', 'Large', 'MNC', 'Startup', 'Other'] # Added more options just in case
house_options = ['Rented', 'Own', 'Family']
loan_options = ['Yes', 'No'] # Use text for clarity, will convert 'existing_loans'
scenario_options = [
    'E-commerce Shopping EMI', 'Home Appliances EMI', 'Vehicle EMI',
    'Personal Loan EMI', 'Education EMI'
]


with col1:
    st.subheader("Personal & Employment")
    # We use .get(column_name) to avoid errors if a column was dropped (like 'age')
    if 'age' in preprocessor_features:
        input_data['age'] = st.number_input("Age", min_value=18, max_value=70, value=35)
    if 'gender' in preprocessor_features:
        input_data['gender'] = st.selectbox("Gender", options=gender_options)
    if 'marital_status' in preprocessor_features:
        input_data['marital_status'] = st.selectbox("Marital Status", options=marital_options)
    if 'education' in preprocessor_features:
        input_data['education'] = st.selectbox("Education", options=education_options)
    if 'employment_type' in preprocessor_features:
        input_data['employment_type'] = st.selectbox("Employment Type", options=employment_options)
    if 'years_of_employment' in preprocessor_features:
        input_data['years_of_employment'] = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    if 'company_type' in preprocessor_features:
        input_data['company_type'] = st.selectbox("Company Type", options=company_options)

with col2:
    st.subheader("Housing & Family")
    if 'house_type' in preprocessor_features:
        input_data['house_type'] = st.selectbox("House Type", options=house_options)
    # We need to ask for the raw expenses to calculate our engineered features
    input_data['monthly_rent'] = st.number_input("Monthly Rent", min_value=0.0, value=15000.0, step=100.0)
    if 'family_size' in preprocessor_features:
        input_data['family_size'] = st.number_input("Family Size", min_value=1, max_value=10, value=3)
    if 'dependents' in preprocessor_features:
        input_data['dependents'] = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    
    st.subheader("Monthly Expenses")
    input_data['school_fees'] = st.number_input("School Fees", min_value=0.0, value=0.0, step=100.0)
    input_data['college_fees'] = st.number_input("College Fees", min_value=0.0, value=0.0, step=100.0)
    input_data['travel_expenses'] = st.number_input("Travel Expenses", min_value=0.0, value=2000.0, step=100.0)
    input_data['groceries_utilities'] = st.number_input("Groceries/Utilities", min_value=0.0, value=5000.0, step=100.0)
    input_data['other_monthly_expenses'] = st.number_input("Other Expenses", min_value=0.0, value=1000.0, step=100.0)
    input_data['current_emi_amount'] = st.number_input("Current EMI Amount", min_value=0.0, value=0.0, step=100.0)

with col3:
    st.subheader("Financial Status")
    if 'monthly_salary' in preprocessor_features:
        input_data['monthly_salary'] = st.number_input("Monthly Salary (Gross)", min_value=0.0, value=50000.0, step=1000.0)
    if 'credit_score' in preprocessor_features:
        input_data['credit_score'] = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=750.0, step=1.0)
    if 'bank_balance' in preprocessor_features:
        input_data['bank_balance'] = st.number_input("Bank Balance", min_value=0.0, value=100000.0, step=1000.0)
    if 'emergency_fund' in preprocessor_features:
        input_data['emergency_fund'] = st.number_input("Emergency Fund", min_value=0.0, value=50000.0, step=1000.0)

    st.subheader("New Loan Details")
    if 'existing_loans' in preprocessor_features:
        # Convert 'Yes'/'No' to boolean for the model
        input_data['existing_loans'] = True if st.selectbox("Existing Loans?", options=loan_options) == 'Yes' else False
    if 'emi_scenario' in preprocessor_features:
        input_data['emi_scenario'] = st.selectbox("EMI Scenario", options=scenario_options)
    if 'requested_amount' in preprocessor_features:
        input_data['requested_amount'] = st.number_input("Requested Loan Amount", min_value=0.0, value=100000.0, step=1000.0)
    if 'requested_tenure' in preprocessor_features:
        input_data['requested_tenure'] = st.number_input("Requested Tenure (Months)", min_value=1, value=12)

# --- Prediction Logic ---
if st.button("Assess Financial Risk", type="primary"):
    
    # 1. Create derived features (same as in data wrangling)
    try:
        total_expenses = (
            input_data['monthly_rent'] +
            input_data['school_fees'] +
            input_data['college_fees'] +
            input_data['travel_expenses'] +
            input_data['groceries_utilities'] +
            input_data['other_monthly_expenses'] +
            input_data['current_emi_amount']
        )
        disposable_income = input_data['monthly_salary'] - total_expenses
        dti_ratio = total_expenses / (input_data['monthly_salary'] + 1e-6)
        
        # Add derived features to the input data
        if 'total_monthly_expenses' in preprocessor_features:
            input_data['total_monthly_expenses'] = total_expenses
        if 'disposable_income' in preprocessor_features:
            input_data['disposable_income'] = disposable_income
        if 'dti_ratio' in preprocessor_features:
            input_data['dti_ratio'] = dti_ratio

        # 2. Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 3. Reorder DataFrame columns to match the preprocessor's training order
        input_df_ordered = input_df[preprocessor_features]

        # 4. Preprocess the data
        input_preprocessed = preprocessor.transform(input_df_ordered)

        # 5. Make predictions
        # Classification
        class_pred_encoded = classifier.predict(input_preprocessed)
        class_pred_label = label_encoder.inverse_transform(class_pred_encoded)[0]
        class_pred_proba = classifier.predict_proba(input_preprocessed)[0]
        class_confidence = np.max(class_pred_proba)

        # Regression
        reg_pred_value = regressor.predict(input_preprocessed)[0]

        # 6. Display results
        st.header("Assessment Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("EMI Eligibility Assessment")
            if class_pred_label == "Eligible":
                st.success(f"**Status: {class_pred_label}** (Confidence: {class_confidence:.1%})")
                st.write("Applicant is a low risk and comfortably affordable.")
            elif class_pred_label == "High_Risk":
                st.warning(f"**Status: {class_pred_label}** (Confidence: {class_confidence:.1%})")
                st.write("Applicant is a marginal case. Recommend manual review or higher interest rates.")
            else: # Not_Eligible
                st.error(f"**Status: {class_pred_label}** (Confidence: {class_confidence:.1%})")
                st.write("Applicant is high risk. Loan is not recommended.")

        with res_col2:
            st.subheader("Affordability Assessment")
            st.metric(label="Maximum Recommended Safe EMI",
                      value=f"₹{reg_pred_value:,.2f} / month")
            st.write("This is the maximum monthly EMI the applicant can safely afford based on their complete financial profile.")

    except KeyError as e:
        st.error(f"A feature is missing from the input data: {e}")
        st.error("This can happen if the Streamlit form is out of sync with the 'preprocessor_features' list.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure all input values are correct.")
