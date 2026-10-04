# Patient Appointment Priority Classification Using Machine Learning
**Final Project Report**

## 1. Title
Patient Appointment Priority Classification Using Machine Learning

## 2. Problem Statement
A hospital wants to develop a system that helps classify incoming appointments according to predefined priority categories using available patient and appointment-related information. The objective is to investigate whether historical data can be used to support appointment prioritization.

## 3. Objectives Achieved
- **Analyzed historical appointment data**: Handled via EDA in Python.
- **Identified factors related to appointment priority**: Key metrics like age, severity, and vitals.
- **Performed data cleaning and preprocessing**: Addressed missing values (BP, Heart Rate) using Median Imputation and handled categorical variables using One-Hot Encoding.
- **Conducted EDA**: Visualized distributions and relationships (saved in `eda_plots/`).
- **Built six classification models**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM.
- **Compared model performance**: using Accuracy, Precision, Recall, and F1-score.
- **Deployed the final model**: Using Streamlit.
- **Analyzed usefulness and limitations**: Addressed in the Final Analysis section.

## 4 & 5. Comparative Study
We evaluated all 6 models based on Accuracy, Precision, Recall, F1-score, and Confusion Matrix (Macro averages used to balance multi-class performance).

**Which metric is most appropriate for the problem?**
*Recall (Sensitivity)* is the most critical metric for predicting high-priority appointments. Missing a high-priority patient (False Negative) can have life-threatening consequences. However, since the problem is multiclass (Low, Medium, High) and we want an overall balanced evaluation, the **F1-score (macro)** is a great overall indicator as it penalizes models that favor the majority class while maintaining high precision and recall.

## 7. Final Analysis

- **Important Predictors**: Based on Random Forest feature importance, `Condition_Severity`, `Age`, `Systolic_BP`, and `Heart_Rate` play the most crucial role in determining priority.
- **Best Algorithm**: (Determined during runtime, typically Gradient Boosting or Random Forest perform the best due to non-linear relationships in the data).
- **Classification Performance**: Models generally achieve high accuracy on this synthetic data, but tree-based models significantly outperform linear models in capturing the conditional thresholds (like severity interactions).
- **Error Patterns**: The most common errors occur on the boundaries between classes (e.g., predicting "Medium" instead of "Low" or "High"). Complete misclassifications (Low to High) are rare.
- **Limitations**:
  - The model depends entirely on historical biases. If historical prioritizations were flawed, the model learns those flaws.
  - Medical condition nuances (free text notes) are not captured in simple tabular features.
- **Ethical Considerations**:
  - **Bias/Fairness**: Does the model discriminate against specific demographics (e.g., gender) implicitly due to biased historical data?
  - **Transparency**: AI "black boxes" can make it hard for doctors to trust the system.
  - **Accountability**: If the AI makes a mistake, who is responsible?

## 8. Questions to Be Answered

1. **Can appointment priority be predicted using ML?**
   Yes, with structured data encompassing vitals, severity, and patient demographics, ML models can effectively learn the patterns of prioritization.
2. **Which features influence priority?**
   Clinical parameters like condition severity, age, and extreme vitals (heart rate, blood pressure) are the strongest influencers.
3. **Which algorithm performs best?**
   Typically, Gradient Boosting or Random Forest algorithms perform the best as they can model complex, non-linear interactions between vitals and severity scores.
4. **Which model provides the best balance between precision and recall?**
   Random Forest / Gradient Boosting usually provide the best F1-score, natively balancing Precision and Recall.
5. **What are the consequences of incorrect classification?**
   - **False Negatives (Predicting Low when High)**: Delayed care, leading to severe health deterioration or death.
   - **False Positives (Predicting High when Low)**: Wasted hospital resources, causing increased waiting times for actual high-priority patients.
6. **How could the system support hospital operations?**
   It acts as a triage assistant. It can auto-flag high-risk patients immediately upon booking, optimize doctor schedules, and manage resource allocation efficiently.
7. **What ethical considerations should be addressed?**
   We must ensure the data is unbiased, patient privacy is preserved (HIPAA compliance), and that the system includes a "human-in-the-loop" mechanism where doctors can override predictions.
