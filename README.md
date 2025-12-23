# Telecom-Customer-Churn-
📊 Telecom Customer Churn Prediction

 Project Overview

Customer churn is a critical problem in the telecom industry where customers discontinue their service. This project builds a machine learning model to predict whether a customer will churn (leave) or not, helping businesses take preventive actions.
🎯 Objective
To predict customer churn using historical telecom customer data and apply machine learning techniques to improve churn detection, especially for customers likely to leave.

Problem Type

Binary Classification

Target Variable: Churn

0 → Not Churn

1 → churn

Technologies Used

Python

Pandas, NumPy – Data manipulation

Scikit-learn – Machine learning models & evaluation

Matplotlib, Seaborn – Data visualization

Jupyter Notebook – Development environment

📁 Dataset

File: telecom_churn.csv

Rows: ~3300 customers

Features: Demographic, service usage, billing information

No missing values


Project Workflow

1️⃣ Data Loading

Loaded CSV data using Pandas

Verified shape, columns, and data types


2️⃣ Data Preprocessing

Separated features (X) and target (y)

Applied One-Hot Encoding to categorical variables

Checked and confirmed no missing values


3️⃣ Train-Test Split

80% training data

20% testing data


4️⃣ Feature Scaling

Applied StandardScaler to normalize feature values

Essential for Logistic Regression convergence


5️⃣ Model Building

Logistic Regression (baseline model)

Improved model using:

class_weight='balanced'

Feature scaling



6️⃣ Model Evaluation

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix visualization

 Results & Insights

🔹 Initial Model (Without Balancing)

Accuracy: 86%

Poor recall for churn customers (missed many churn cases)


 Improved Model (With Scaling & Class Balancing)

Accuracy: 79%

Churn Recall improved from 18% → 80%


📌 Although accuracy decreased slightly, the model became far more effective in identifying churn customers, which is crucial from a business perspective.


 Confusion Matrix Interpretation

True Negatives: Correctly identified non-churn customers

True Positives: Correctly identified churn customers

Reduced false negatives after class balancing

 Key Learnings

Accuracy alone is not sufficient for imbalanced datasets

Recall is critical in churn prediction

Feature scaling significantly improves model convergence

Class imbalance handling improves business usefulness

🚀 Future Improvements

Try advanced models (Random Forest, XGBoost)

Perform hyperparameter tuning

Add feature importance analysis

Deploy model using Flask or Streamlit

👤 Author

Iniyanila
Second-Year IT Student | Aspiring AI Engineer

📌 Conclusion

This project demonstrates an end-to-end machine learning pipeline for customer churn prediction and reflects real-world ML problem-solving skills suitable for internships and entry-level ro
