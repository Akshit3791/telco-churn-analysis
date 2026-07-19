# ==============================================================================
# PROJECT: Telco Customer Churn Analysis
# AUTHOR: Akshit
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. DATA LOADING & CLEANING
print("Loading and cleaning data...")
# Assuming the dataset is uploaded as 'telco_churn.csv'
df = pd.read_csv('telco_churn.csv')

# Handling missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Dropping unnecessary ID column
df.drop('customerID', axis=1, inplace=True)

# 2. FEATURE ENGINEERING
print("Engineering features...")
# Converting target variable to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Scaling numerical variables
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 3. TRAIN-TEST SPLIT
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. PREDICTIVE MODELING (Random Forest)
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 5. MODEL EVALUATION
print("Evaluating the model...")
y_pred = rf_model.predict(X_test)

print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================================================================
# End of Script
# ==============================================================================
