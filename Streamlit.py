import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

# Load saved components
try:
    model = joblib.load("xgb_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    poly = joblib.load("poly_features.pkl")  # Pre-trained PolynomialFeatures transformer
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Define features
numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
time_features = ['hour', 'dayofweek', 'month']
# Final expected columns for bulk input: numerical_features + engineered time features + amount_bins

# Set up Streamlit page configuration and custom CSS for a modern look
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")
st.markdown("""
    <style>
        body {background-color: #f0f2f6;}
        .title {color: #e74c3c; font-size: 42px; font-weight: bold; text-align: center; margin-bottom: 20px;}
        .subheader {color: #3498db; font-size: 24px; font-weight: bold; margin-bottom: 10px;}
        .prediction-safe {color: green; font-size: 24px; padding: 10px; text-align: center;}
        .prediction-fraud {color: red; font-size: 24px; padding: 10px; text-align: center;}
        .sidebar .sidebar-content {background-color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💳 Fraud Detection System</p>', unsafe_allow_html=True)

# Preprocessing function to handle both single and bulk inputs
def preprocess_input(input_df):
    """
    Preprocess input data:
    - Convert datetime and extract time features.
    - Create 'amount_bins' feature.
    - Scale numerical features.
    - Generate polynomial features.
    """
    # Ensure transaction datetime is a datetime object
    if 'trans_date_trans_time' in input_df.columns:
        input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'])
        input_df['hour'] = input_df['trans_date_trans_time'].dt.hour
        input_df['dayofweek'] = input_df['trans_date_trans_time'].dt.dayofweek
        input_df['month'] = input_df['trans_date_trans_time'].dt.month
    else:
        # If datetime not provided, assume default values (could be enhanced)
        input_df['hour'] = 0
        input_df['dayofweek'] = 0
        input_df['month'] = 0

    # Create amount bins feature based on 'amt'
    input_df['amount_bins'] = pd.cut(input_df['amt'], bins=[0, 10, 100, 500, 5000], labels=[1, 2, 3, 4])
    # If any NaN occur in bins, fill with default value 0
    input_df['amount_bins'] = pd.to_numeric(input_df['amount_bins'], errors='coerce').fillna(0).astype(int)

    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Generate polynomial features
    poly_features = poly.transform(input_df[numerical_features])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_features))
    poly_df.index = input_df.index

    # Combine original features with engineered time and polynomial features
    processed_df = pd.concat([input_df, poly_df.add_prefix('poly_')], axis=1)
    # Optionally drop the original datetime column if it exists
    processed_df = processed_df.drop(columns=['trans_date_trans_time'], errors='ignore')
    
    return processed_df

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Single Transaction", "Bulk Upload", "Model Info"])

# ----------------------
# Single Transaction Page
# ----------------------
if page == "Single Transaction":
    st.subheader("🔍 Predict Fraud for a Single Transaction")
    
    with st.form("single_transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("💰 Transaction Amount ($)", min_value=0.01, value=50.0, step=0.01)
            trans_date = st.date_input("📅 Transaction Date", value=datetime.today())
            trans_time = st.time_input("⏰ Transaction Time", value=datetime.now().time())
            lat = st.number_input("📍 Customer Latitude", value=40.7128)
            long = st.number_input("📍 Customer Longitude", value=-74.0060)
        with col2:
            city_pop = st.number_input("🏙 City Population", min_value=1, value=100000)
            merch_lat = st.number_input("🏪 Merchant Latitude", value=40.7138)
            merch_long = st.number_input("🏪 Merchant Longitude", value=-74.0059)
        
        submitted = st.form_submit_button("🔍 Predict Fraud")
    
    if submitted:
        # Create a DataFrame for the single transaction
        input_dt = datetime.combine(trans_date, trans_time)
        single_input = pd.DataFrame([[amt, lat, long, city_pop, merch_lat, merch_long, input_dt]], 
                                    columns=numerical_features + ['trans_date_trans_time'])
        
        # Preprocess the input before prediction
        processed_input = preprocess_input(single_input)
        
        # Make prediction
        proba = model.predict_proba(processed_input)[:, 1][0]  # Get the probability for the fraud class
        prediction = proba > 0.5  # adjust threshold as needed
        
        if prediction:
            st.markdown(f'<div class="prediction-fraud">🚨 Fraud Detected (Probability: {proba:.2%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-safe">✅ Legitimate Transaction (Probability: {proba:.2%})</div>', unsafe_allow_html=True)

# ----------------------
# Bulk Upload Page
# ----------------------
elif page == "Bulk Upload":
    st.subheader("📂 Detect Fraud in Bulk Transactions")
    
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
        
        # Ensure required columns are present: numerical features and a transaction datetime column.
        missing_cols = set(numerical_features) - set(df.columns)
        if 'trans_date_trans_time' not in df.columns:
            st.error("CSV file is missing the required 'trans_date_trans_time' column for time feature extraction!")
        elif missing_cols:
            st.error(f"CSV file is missing required numerical columns: {missing_cols}")
        else:
            try:
                processed_df = preprocess_input(df)
                predictions = model.predict_proba(processed_df)[:, 1]
                df["Fraud_Probability"] = predictions
                df["Fraud_Prediction"] = (predictions > 0.5).astype(int)
                
                fraud_cases = df[df["Fraud_Prediction"] == 1]
                st.markdown(f'<p class="subheader" style="color:red;">🚨 {len(fraud_cases)} fraudulent transactions detected!</p>', unsafe_allow_html=True)
                st.dataframe(df[['amt', 'Fraud_Prediction', 'Fraud_Probability']].head(20))
                
                # Download predictions
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")

# ----------------------
# Model Info Page
# ----------------------
elif page == "Model Info":
    st.subheader("📊 Model Information")
    st.write("**Model Type:** XGBoost Classifier")
    st.write("**Features Used:**")
    st.write("- Numerical: " + ", ".join(numerical_features))
    st.write("- Time-based: " + ", ".join(time_features))
    st.write("- Engineered: amount_bins, polynomial features")
    st.write("**Model Performance Metrics:**")
    # You can update these metrics based on your evaluation
    st.metric("ROC-AUC Score", "0.985")
    st.metric("Recall", "0.92")
    st.metric("Precision", "0.89")
