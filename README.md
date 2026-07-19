# Telco Customer Churn Analysis & Prediction

## 📌 Project Overview
This project is an end-to-end data analytics workflow focused on identifying the key drivers of customer attrition in the telecommunications industry and building a predictive machine learning model to assist retention efforts. 

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Statistical Testing:** SciPy, Statsmodels
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)

## 🚀 Project Workflow
1. **Exploratory Data Analysis (EDA):** Analyzed a dataset of 7,043 customers and 21 variables, uncovering a baseline churn rate of 26.5%.
2. **Data Cleaning & Pre-processing:** Handled missing values via median imputation, dropped redundant identifiers, and scaled continuous features using `StandardScaler`. Mapped categorical variables using One-Hot Encoding.
3. **Data Visualization:** Generated bar charts, scatter plots, and correlation heatmaps to identify visual trends between contract types, monthly charges, and churn.
4. **Statistical Hypothesis Testing:** Conducted Independent T-Tests, Chi-Square Tests, and Pearson Correlation Analysis to mathematically validate the exploratory findings.
5. **Predictive Modeling:** Built and fine-tuned a **Random Forest Classifier** using GridSearchCV. Evaluated the model using Accuracy, F1-Score, Confusion Matrix, and ROC-AUC curves.

## 💡 Key Business Insights & Recommendations
* **Contract Length Matters:** Month-to-month contracts suffer from the highest attrition rates. *Recommendation:* Incentivize customers to switch to 1-year or 2-year commitments.
* **Pricing Impact:** High initial monthly charges strongly correlate with early churn. *Recommendation:* Implement phased billing or promotional pricing for early-stage retention.
