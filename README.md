# Patient Appointment Priority Classification

## Overview
This repository contains a Machine Learning project aimed at classifying patient appointments into priority categories (Low, Medium, High). The project is built completely in Python, making it suitable for a standard Data Science/ML pipeline. 

## Features
- **Data Generator**: `data_generator.py` generates synthetic appointment data including age, gender, conditions, heart rate, blood pressure, and condition severity.
- **Model Training & Evaluation**: `model_training.py` performs EDA (saving plots to `eda_plots/`), handles data preprocessing, and trains 6 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM).
- **Deployment**: `app.py` serves the trained model interactively via Streamlit.
- **Project Report**: `Project_Report.md` directly answers all final analysis objectives.

## How to run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Generate synthetic data:
   ```
   python data_generator.py
   ```
3. Train models and generate plots/metrics:
   ```
   python model_training.py
   ```
4. Run the Streamlit web app:
   ```
   streamlit run app.py
   ```
