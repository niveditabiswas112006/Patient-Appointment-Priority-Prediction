import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Patient Appointment Priority Prediction", page_icon="🏥", layout="centered")

# Title and description
st.title("🏥 Patient Appointment Priority Prediction")
st.markdown("""
This application classifies incoming patient appointments into priority categories: **Low**, **Medium**, or **High**.
It uses a Machine Learning model trained on historical appointment data.
""")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'models/best_priority_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

if model is None:
    st.error("Model not found! Please run the training script first.")
    st.stop()

st.sidebar.header("Patient & Appointment Details")

# Input fields
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30, step=1)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"])
medical_condition = st.sidebar.selectbox(
    "Medical Condition", 
    options=['Routine Checkup', 'Fever/Cold', 'Chronic Pain', 'Injury/Trauma', 'Cardiac Symptoms', 'Respiratory Issue']
)
condition_severity = st.sidebar.slider("Condition Severity (1: Very Mild - 5: Critical)", min_value=1, max_value=5, value=2)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75, step=1)
systolic_bp = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=120, step=1)
distance_km = st.sidebar.number_input("Distance from Hospital (km)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)

# Predict button
if st.sidebar.button("Predict Priority"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Medical_Condition': [medical_condition],
        'Condition_Severity_1to5': [condition_severity],
        'Heart_Rate': [heart_rate],
        'Systolic_BP': [systolic_bp],
        'Distance_km': [distance_km]
    })
    
    # Display the inputs
    st.subheader("Patient Profile")
    st.dataframe(input_data)
    
    # Predict
    with st.spinner("Analyzing..."):
        prediction = model.predict(input_data)[0]
        # prediction is 0: Low, 1: Medium, 2: High
        mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_class = mapping.get(prediction, "Unknown")
        
        # Display Result
        st.subheader("Predicted Priority Category")
        if predicted_class == 'High':
            st.error(f"🚨 {predicted_class} Priority")
        elif predicted_class == 'Medium':
            st.warning(f"⚠️ {predicted_class} Priority")
        else:
            st.success(f"✅ {predicted_class} Priority")
            
        st.markdown("---")
        st.info("**Note:** This is a Machine Learning prediction and should support, not replace, clinical judgment.")

# Additional Info
with st.expander("About the System"):
    st.write("""
    **Objective:** Help the hospital classify incoming appointments according to predefined priority categories.
    
    **Features Used:**
    - Demographics (Age, Gender, Distance)
    - Clinical factors (Severity, Condition, Heart Rate, BP)
    
    **Machine Learning:** The model utilizes the best performing classifier to evaluate patient conditions based on historical patterns.
    """)

with st.expander("Machine Learning Models Evaluated"):
    st.write("""
    During the development of this system, **6 different Machine Learning models** were trained and evaluated to ensure the highest accuracy:
    
    1. **Logistic Regression:** A statistical model that estimates the probability of each priority class using a linear equation.
    2. **K-Nearest Neighbors (KNN):** Predicts the priority by finding the most similar past patients (neighbors) based on their features.
    3. **Decision Tree:** A flowchart-like model that makes decisions based on features (like severity and age) to classify the appointment.
    4. **Random Forest:** An ensemble method that builds multiple decision trees and merges their results for more accurate and stable predictions.
    5. **Gradient Boosting:** An advanced ensemble technique that builds trees sequentially, where each new tree corrects the errors of previous ones.
    6. **Support Vector Machine (SVM):** Finds the optimal boundary (hyperplane) to separate the different priority classes in high-dimensional space.
    
    The system automatically selects and runs the **best performing model** to make these real-time predictions.
    """)
