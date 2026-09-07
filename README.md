# Telco Customer Churn Analysis

## Overview

This project analyzes customer churn using the Telco Customer Churn dataset. The main goal is to understand the factors associated with customer churn, build a baseline classification model, and investigate how model predictions can be translated into cost-sensitive retention decisions.

The project combines exploratory data analysis, SQL, Logistic Regression, feature importance techniques, model evaluation, threshold optimization, and a hypothetical retention campaign simulation.

## Project Structure

```text
telco-churn-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── telco_churn_analysis.ipynb
│
├── requirements.txt
└── README.md
```

## Analysis

The notebook covers:

1. Data Loading and Initial Exploration
2. Data Cleaning
3. Exploratory Data Analysis
4. SQL Analysis
5. Logistic Regression
6. Feature Importance
   - Logistic Regression coefficients
   - Permutation Importance
   - Mutual Information
7. Model Evaluation
   - Accuracy
   - Confusion Matrix
   - Precision
   - Recall
   - F1-score
8. ROC-AUC Analysis
9. Classification Threshold Analysis
10. Business Cost Analysis
11. Retention Campaign Simulation

## Key Findings

- Customer churn is strongly associated with contract type and tenure.
- Month-to-month customers have substantially higher churn rates than customers with longer contracts.
- Customers with shorter tenure are considerably more likely to churn.
- Service-related features such as OnlineSecurity and TechSupport show meaningful relationships with churn.
- Payment method and billing-related features also show differences in churn rates.
- Gender showed little predictive value compared with the other features.
- Logistic Regression achieved a ROC-AUC of approximately **0.83**.
- The optimal classification threshold depends on the business objective and the relative costs of false positives and false negatives.
- A more effective retention campaign can justify using a lower threshold and targeting a larger group of potentially churning customers.

## Business Perspective

A key part of the project is the transition from model evaluation to decision-making.

Instead of assuming that a probability threshold of 0.5 is always appropriate, different thresholds were evaluated under hypothetical business scenarios.

The analysis demonstrates that the best threshold depends on factors such as:

- Cost of contacting a customer
- Cost of losing a customer
- Expected value of retaining a customer
- Success rate of the retention campaign

Therefore, the model's probability predictions and the final business decision are treated as two separate steps.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQLite / SQL
- Jupyter Notebook

## Dataset

The project uses the Telco Customer Churn dataset containing customer demographic information, account information, subscribed services, billing information, and churn status.

The dataset contains **7,043 customers and 21 columns**.

## Future Work

Possible extensions include:

- Comparing Logistic Regression with tree-based models such as XGBoost
- Hyperparameter tuning
- Cross-validation
- Selecting the classification threshold using a validation set rather than the test set
- Comparing models using both predictive metrics and business-oriented metrics
- Investigating customer-level differences in retention campaign effectiveness