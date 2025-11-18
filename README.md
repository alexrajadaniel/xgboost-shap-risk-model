# Interpretable AI: SHAP Analysis of a High-Dimensional Classification Model  
### XGBoost + SHAP + Realistic Financial Risk Dataset

This project builds an **interpretable machine learning model** that predicts customer credit risk using a realistic, multi-dimensional dataset.  
The key objective is to combine **high predictive performance** with **transparent explanations** using **SHAP (SHapley Additive exPlanations)**.

---

## 📌 Project Objectives
- Build a classification model using **XGBoost**
- Apply a **Scikit-Learn preprocessing pipeline**
- Compute **global and local SHAP explanations**
- Understand the influence of each feature on predictions
- Analyze **false positives & false negatives** with SHAP force plots
- Provide insights for real-world financial risk assessment

---

## 📂 Dataset Overview
The dataset contains **1200 samples** and **16 features**:

### **Numeric Features**
- age  
- income  
- credit_score  
- loan_amount  
- loan_tenure_months  
- num_previous_loans  
- default_history  
- monthly_spend  
- transaction_count  
- late_payments  
- account_age_months  

### **Categorical Features**
- gender  
- employment_type  
- region  

### **Target**
- `0` → Low Risk  
- `1` → High Risk  

Dataset is slightly imbalanced (70% vs 30%).

---

## 🔧 Tech Stack / Libraries Used
- Python 3  
- pandas  
- numpy  
- scikit-learn  
- XGBoost  
- SHAP  
- Matplotlib  

---

## 🏗️ Model Pipeline

Preprocessing and modeling are done using a unified Scikit-Learn `Pipeline`:

1. **Numeric features** → Median Imputation  
2. **Categorical features** → Most-frequent Imputation + OneHotEncoder  
3. **Classifier** → `XGBClassifier`  

This ensures clean and repeatable training.

---

## 📊 Model Performance

### **Classification Report**
- Precision (Class 0): **0.89**  
- Recall (Class 0): **0.86**  
- Precision (Class 1): **0.70**  
- Recall (Class 1): **0.75**  
- Overall Accuracy: **0.83**

### **ROC-AUC**
**0.904** → Excellent performance.

---

## 🔍 Global SHAP Analysis

SHAP summary and bar plots show the most influential features.

### **Top Feature (Most Important)**
✅ **loan_amount**

### **Other high-impact features include:**
- credit_score  
- monthly_spend  
- late_payments  
- default_history  
- income  
- num_previous_loans  

The model aligns well with real financial logic.

---

## 🎯 Local SHAP Explanations (Edge Cases)

Five edge cases were analyzed:

- **3 False Negatives**  
- **2 False Positives**

Each case was visualized using SHAP **force plots**, and reasons for misclassification were documented.  
Examples:

- FN: low loan amount + high income → model underestimates risk  
- FP: high loan amount + many late payments → model overestimates risk  

All plots are saved inside:


