# Telco Customer Churn Analysis

A machine learning project for predicting customer churn using the **Telco Customer Churn dataset**. The project explores several classification approaches, compares their predictive performance, and connects model predictions to a simple business decision-making scenario.

The main goal is not only to predict *who is likely to churn*, but also to understand how different models and classification thresholds affect the potential business outcome.

---

## Project Overview

Customer churn is a common business problem in subscription-based services. Identifying customers who are likely to leave can help companies prioritize retention efforts.

In this project, the analysis follows a progressive modeling workflow:

**EDA + SQL → Logistic Regression → Decision Tree → XGBoost → Neural Network → Model Comparison**

Four different modeling approaches were evaluated, ranging from a simple interpretable linear model to nonlinear ensemble and neural network models.

---

## Dataset

The project uses the **Telco Customer Churn** dataset, containing:

- **7,043 customers**
- **21 original features**
- Demographic, service, contract, billing, and customer account information
- Binary target variable: `Churn`

The `customerID` column was removed because it does not contain predictive information.

### Data Cleaning

`TotalCharges` was stored as a string because of several blank values. These values were converted to numeric and missing values were filled with `0`. The affected observations corresponded to customers with zero tenure.

The target variable was encoded as:

- `No → 0`
- `Yes → 1`

---

## Project Structure

### 01 — EDA & Logistic Regression

`01_eda_logistic_regression.ipynb`

The first notebook establishes the baseline and explores the dataset.

Topics covered:

- Data cleaning
- Exploratory Data Analysis
- SQL-based analysis
- Feature preprocessing
- Logistic Regression
- Confusion matrix
- Precision, Recall, F1-score
- ROC-AUC
- Feature interpretation
- Classification threshold analysis
- Simple business profit simulation

Logistic Regression achieved a **ROC-AUC of approximately 0.83**.

A business-oriented threshold analysis was also performed to demonstrate that the default classification threshold of 0.5 is not necessarily optimal. Under a hypothetical retention-cost and churn-loss scenario, a lower threshold produced a substantially higher simulated profit.

This section highlights an important distinction between **predictive performance and business decision-making**.

---

### 02 — Decision Tree

`02_decision_tree_churn.ipynb`

A Decision Tree was evaluated as a nonlinear alternative to Logistic Regression.

The baseline model showed substantial overfitting, achieving nearly perfect training accuracy while performing considerably worse on the test set.

Hyperparameter tuning was performed using `GridSearchCV` with ROC-AUC as the optimization metric.

| Model | ROC-AUC |
|---|---:|
| Baseline Decision Tree | 0.652 |
| Tuned Decision Tree | 0.812 |

The tuned model generalized considerably better than the baseline, but still performed slightly below Logistic Regression.

This experiment demonstrates how controlling model complexity can substantially improve the generalization of tree-based models.

---

### 03 — XGBoost

`03_xgboost_churn.ipynb`

XGBoost was evaluated as a more powerful ensemble-based alternative.

The baseline model showed some overfitting, but substantially less than the default Decision Tree.

Hyperparameter search was performed using `RandomizedSearchCV`, optimizing cross-validation ROC-AUC.

| Model | ROC-AUC |
|---|---:|
| Baseline XGBoost | 0.812 |
| Tuned XGBoost | 0.831 |

The tuned XGBoost model achieved the highest test ROC-AUC among the models evaluated.

However, Logistic Regression still achieved slightly higher Accuracy, Precision, Recall, and F1-score at the default classification threshold of 0.5.

This illustrates that a higher ROC-AUC does not necessarily translate into better performance at one particular classification threshold.

---

### 04 — Neural Network

`04_neural_network_churn.ipynb`

A feed-forward neural network was evaluated using several architectures.

The models used:

- Standardized numerical features
- One-hot encoded categorical features
- ReLU hidden layers
- Dropout regularization
- Sigmoid output layer
- Adam optimizer
- Binary cross-entropy loss
- Early stopping based on validation ROC-AUC

Several architectures were compared rather than assuming that a larger network would perform better.

The selected architecture was:

**32 → 16 → 1**

Increasing the size and depth of the network did not provide a meaningful improvement in generalization. Some larger architectures achieved slightly higher ROC-AUC, but this did not translate into better Recall or F1-score at the default threshold.

The neural network was therefore retained as an additional model for comparison rather than the final best-performing model.

---

## Model Comparison

The final comparison uses the best/selected version of each modeling approach:

- Logistic Regression
- Tuned Decision Tree
- Tuned XGBoost
- Selected Neural Network (`32 → 16`)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.797 | 0.635 | 0.524 | 0.574 | 0.827 |
| Tuned Decision Tree | 0.787 | 0.608 | 0.519 | 0.560 | 0.812 |
| Tuned XGBoost | 0.795 | 0.634 | 0.508 | 0.564 | **0.831** |
| Neural Network (32→16) | 0.786 | 0.596 | **0.554** | **0.575** | 0.819 |

### Key observations

- **XGBoost achieved the highest ROC-AUC (0.831)**, indicating the strongest overall ranking ability among the evaluated models.
- **Logistic Regression remained highly competitive**, despite being considerably simpler and more interpretable.
- The **Decision Tree benefited substantially from hyperparameter tuning**, improving its ROC-AUC from 0.65 to 0.81.
- The **Neural Network did not provide a meaningful advantage** over the simpler models.
- Performance depends on the evaluation metric and classification threshold; no single model was best across every metric.

These results demonstrate that increasing model complexity does not automatically lead to better predictive performance.

---

## Business Perspective

The project goes beyond comparing classification metrics.

For churn prediction, the cost of missing a potential churner may be different from the cost of contacting a customer who would have stayed anyway.

A simple hypothetical profit model was therefore used to investigate how the classification threshold affects:

- Number of customers targeted
- False positives
- False negatives
- Retention cost
- Potential churn loss
- Expected profit

This highlights an important practical point:

> **The best classification threshold depends on the business objective, not only on the model's default probability cutoff.**

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SQL
- Jupyter Notebook

---

## Machine Learning Techniques

The project covers:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- SQL analysis
- One-hot encoding
- Feature scaling
- Logistic Regression
- Decision Trees
- XGBoost
- Neural Networks
- Cross-validation
- Grid Search
- Randomized Search
- Early Stopping
- Classification metrics
- ROC-AUC analysis
- Threshold optimization
- Basic business/profit analysis

---

## Main Takeaways

This project demonstrates a progression from an interpretable statistical baseline to more complex machine learning approaches.

The main findings were:

1. Logistic Regression provided a strong and competitive baseline.
2. A single Decision Tree initially overfit heavily, but tuning substantially improved its generalization.
3. XGBoost achieved the highest ROC-AUC, although its advantage over Logistic Regression was relatively small.
4. Increasing neural network complexity did not produce a meaningful improvement.
5. Model evaluation should consider multiple metrics rather than relying on Accuracy alone.
6. For a real churn application, the classification threshold should ultimately be selected according to the business cost of false positives and false negatives.

Overall, the project shows that **model complexity should be justified by measurable improvement**, rather than assumed to produce better results.