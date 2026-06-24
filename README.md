# Audit Anomaly & Fraud Detection Pipeline

## 📌 Executive Summary
This project builds an end-to-end machine learning pipeline to detect fraudulent credit card transactions within a highly imbalanced dataset (0.17% actual fraud). Designed for a high-stakes financial audit environment, the system prioritizes **Recall** and **Precision** over raw accuracy to balance the interception of fraudulent behavior with the mitigation of false alarms for normal customers.

## 🏗️ Architecture & Tech Stack
* **Data Extraction:** SQL (SQLite database mimicking enterprise data warehouses)
* **Exploratory Data Analysis (EDA):** Python, Pandas, Seaborn
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Models Evaluated:** Logistic Regression, Artificial Neural Network (MLP)

## 📊 Methodology
### 1. Feature Engineering & Correlation
To maintain interpretability and identify direct linear relationships, **Pearson Correlation** was utilized over dimensionality reduction techniques like factor analysis. 
* **Key Finding:** Transaction `Amount` showed the highest linear correlation with fraud, while geographic features exhibited severe multicollinearity, informing our feature selection.

### 2. Handling the Class Imbalance
The initial dataset contained 1.03 million training rows, but only a **0.58%** fraud ratio. Training a model on this distribution leads to a 99% accuracy rate but a 0% fraud catch rate. 
* **Solution:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly to the training data to synthesize realistic fraudulent transactions, achieving a 50/50 balanced training environment while keeping the test data untouched to prevent data leakage.

## 🚀 Model Evaluation & Business Impact
Two distinct architectures were evaluated on 259,335 unseen test transactions. The Artificial Neural Network was configured with an efficient 8 hidden units to balance learning capacity and prevent overfitting on tabular data.

| Model | Recall (Catch Rate) | Precision (False Alarms) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Artificial Neural Network** | 0.76 | 0.06 | 0.11 |
| **Logistic Regression** | 0.76 | 0.09 | 0.15 |

### 🏆 Conclusion: The Case for Simplicity
While both models successfully intercepted **76% of actual fraud**, the **Logistic Regression** model is the clear winner for deployment in an audit context. 

It achieved a higher precision score (0.09 vs. 0.06), meaning it generated fewer false positives. In a live banking environment, this translates to catching the exact same amount of financial fraud while significantly reducing the number of normal customers who have their credit cards unnecessarily blocked.