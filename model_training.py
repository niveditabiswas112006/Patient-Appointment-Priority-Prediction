import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def perform_eda(df):
    os.makedirs('eda_plots', exist_ok=True)
    
    # 1. Distribution of Priority
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Priority', order=['Low', 'Medium', 'High'], palette='viridis')
    plt.title('Distribution of Appointment Priority')
    plt.savefig('eda_plots/priority_distribution.png')
    plt.close()
    
    # 2. Age vs Priority
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Priority', y='Age', order=['Low', 'Medium', 'High'], palette='viridis')
    plt.title('Age Distribution by Priority')
    plt.savefig('eda_plots/age_vs_priority.png')
    plt.close()
    
    # 3. Severity vs Priority
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Condition_Severity_1to5', hue='Priority', palette='viridis')
    plt.title('Condition Severity vs Priority')
    plt.savefig('eda_plots/severity_vs_priority.png')
    plt.close()
    
    print("EDA completed and plots saved in 'eda_plots/' directory.")

def train_and_evaluate_models():
    # Load dataset
    df = pd.read_csv('data/patient_appointments.csv')
    
    # Data Cleaning and Preprocessing
    # Target encoding
    le = LabelEncoder()
    # Let's map explicitly to preserve ordinality if needed, though sklearn treats them as nominal
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Priority'] = df['Priority'].map(target_mapping)
    
    X = df.drop('Priority', axis=1)
    y = df['Priority']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing Pipeline
    numeric_features = ['Age', 'Condition_Severity_1to5', 'Heart_Rate', 'Systolic_BP', 'Distance_km']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Gender', 'Medical_Condition']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define models (6 models as required: added SVM to make it 6 classification models alongside the 5 requested algorithms)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = []
    best_model = None
    best_f1 = 0
    best_model_name = ""
    best_pipeline = None
    
    print("\n--- Model Training and Evaluation ---")
    
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # We use macro average for multiclass precision, recall, f1
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1-Score (Macro)': f1
        })
        
        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'eda_plots/cm_{name.replace(" ", "_")}.png')
        plt.close()
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf

    results_df = pd.DataFrame(results)
    print("\n--- Comparative Study ---")
    print(results_df.to_string(index=False))
    
    print(f"\nBest Model based on F1-Score: {best_model_name} with F1 = {best_f1:.4f}")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/best_priority_model.pkl')
    print("Best model saved to 'models/best_priority_model.pkl'")

    # Save feature importance if applicable (Random Forest)
    if 'Random Forest' in models:
        rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models['Random Forest'])])
        rf_pipeline.fit(X_train, y_train)
        
        # Get feature names after one-hot encoding
        cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        all_features = numeric_features + list(cat_features)
        
        importances = rf_pipeline.named_steps['classifier'].feature_importances_
        
        feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.tight_layout()
        plt.savefig('eda_plots/feature_importance.png')
        plt.close()
        print("Feature importance plot saved.")

if __name__ == "__main__":
    df = pd.read_csv('data/patient_appointments.csv')
    perform_eda(df)
    train_and_evaluate_models()
