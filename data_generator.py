import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Features
    age = np.random.randint(0, 100, num_samples)
    gender = np.random.choice(['Male', 'Female', 'Other'], num_samples, p=[0.48, 0.50, 0.02])
    
    # 1: Very Mild, 5: Critical
    condition_severity = np.random.randint(1, 6, num_samples) 
    
    # Conditions
    medical_condition = np.random.choice(
        ['Routine Checkup', 'Fever/Cold', 'Chronic Pain', 'Injury/Trauma', 'Cardiac Symptoms', 'Respiratory Issue'],
        num_samples
    )
    
    # Vitals
    heart_rate = np.random.normal(75, 15, num_samples).astype(int)
    systolic_bp = np.random.normal(120, 20, num_samples).astype(int)
    
    distance_from_hospital_km = np.random.uniform(0.5, 50.0, num_samples).round(1)
    
    # If a patient is older, has higher severity, or high heart rate, they are more likely to have higher priority
    # Let's create a logic to define the Priority (0: Low, 1: Medium, 2: High)
    priority_score = (
        (age * 0.05) + 
        (condition_severity * 2.5) + 
        (np.where(heart_rate > 100, 2, 0)) +
        (np.where(systolic_bp > 140, 2, 0)) +
        np.random.normal(0, 1.5, num_samples) # Add some noise
    )
    
    # Discretize the priority score into categories
    priority = pd.qcut(priority_score, q=[0, 0.4, 0.75, 1.0], labels=['Low', 'Medium', 'High'])
    
    # Introduce some missing values to simulate real-world data requiring cleaning
    missing_idx = np.random.choice(num_samples, size=int(num_samples*0.05), replace=False)
    systolic_bp = systolic_bp.astype(float)
    systolic_bp[missing_idx] = np.nan
    
    missing_idx_hr = np.random.choice(num_samples, size=int(num_samples*0.03), replace=False)
    heart_rate = heart_rate.astype(float)
    heart_rate[missing_idx_hr] = np.nan

    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Medical_Condition': medical_condition,
        'Condition_Severity_1to5': condition_severity,
        'Heart_Rate': heart_rate,
        'Systolic_BP': systolic_bp,
        'Distance_km': distance_from_hospital_km,
        'Priority': priority
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv('data/patient_appointments.csv', index=False)
    print("Synthetic dataset generated and saved to 'data/patient_appointments.csv'")
