import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load model and scaler with correct paths
model_path = os.path.join(project_root, 'models', 'disease_model.pkl')
scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"❌ Error: Could not load model files. Please ensure you have run training first.\n\nDetails: {e}")
    st.stop()

st.title("AI-Based Early Disease Detection System")

st.header("Enter Patient Symptoms")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=21, max_value=81, value=30)

if st.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], 
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High risk of diabetes. Probability: {probability:.2f}")
    else:
        st.success(f"Low risk of diabetes. Probability: {probability:.2f}")

st.header("Model Information")
st.write("This model predicts the risk of diabetes based on medical parameters.")
st.write("Accuracy metrics are displayed in the training script.")

st.write("---------------------------------------------------------------------------")
st.write("© 2026 AI-Based Early Disease Detection System.")
st.write("Created by IPSITA DAS")