# 💰 EMIPredict AI

### Intelligent Financial Risk Assessment

EMIPredict AI is a machine learning application designed to support loan assessment by predicting an applicant's EMI eligibility category and estimating the maximum affordable monthly EMI.

The project uses historical financial data, feature engineering, and two tuned XGBoost models in a Streamlit web application.

## 🌐 Project Links

- **GitHub Repository:** https://github.com/PiyushChavda595/EMI-Predict-AI

## 📌 Features

### ✅ EMI Eligibility Classification

Classifies applicants into three categories:

- `Eligible`
- `High_Risk`
- `Not_Eligible`

### 💵 Maximum Affordable EMI Prediction

Estimates the maximum affordable monthly EMI in Indian rupees using a regression model.

### 🧹 Data Cleaning and Feature Engineering

Cleans inconsistent data types, handles missing values, and creates features such as `total_monthly_expenses`, `disposable_income`, and `dti_ratio`.

### 📊 Interactive Streamlit Interface

Provides an interactive Streamlit application for entering applicant information and viewing both model predictions. The report describes a three-column input layout.

### 🔍 Model Explainability

Uses SHAP with `TreeExplainer` to inspect the classifier's predictions and identify influential features.

### 🧪 Experiment Tracking

Uses MLflow to record model parameters, metrics, and artifacts.

## 🏗️ System Architecture

```text
Applicant Financial Information
             │
             ▼
   Feature Engineering
             │
             ▼
    Data Preprocessor
             │
       ┌─────┴─────┐
       ▼           ▼
XGBoost         XGBoost
Classifier      Regressor
       │           │
       ▼           ▼
Eligibility     Maximum
Category        Affordable EMI
       │           │
       └─────┬─────┘
             ▼
      Streamlit App
             │
             ▼
       Display Results
```

## 🔄 Model Workflow

```text
Financial Dataset
       ↓
Data Cleaning
       ↓
Missing-Value Imputation
       ↓
Feature Engineering
       ↓
Numerical and Categorical Preprocessing
       ↓
Model Training and Evaluation
       ↓
MLflow Experiment Tracking
       ↓
Save Models with joblib
       ↓
Streamlit Application
       ↓
Eligibility and EMI Predictions
```

## 🤖 Machine Learning Models

The project solves two related prediction tasks using the same applicant input.

### 1. EMI Eligibility Classification

**Problem Type:** Multi-class classification

**Selected Model:** Tuned XGBoost Classifier

The classifier predicts one of three eligibility categories: `Eligible`, `High_Risk`, or `Not_Eligible`.

The dataset contains a strong class imbalance, with approximately 77.3% of records labeled `Not_Eligible`. The project uses class weighting to help address this imbalance and evaluates classification performance using Macro F1-Score.

### 2. Maximum Affordable EMI Prediction

**Problem Type:** Regression

**Selected Model:** Tuned XGBoost Regressor

The regressor estimates an applicant's maximum affordable monthly EMI in rupees.

## 📊 Model Performance

The project report records the following final model results.

### Classification

| Metric | Result |
|---|---:|
| Macro F1-Score | 0.81 |
| `High_Risk` F1-Score | 0.53 |
| `Eligible` F1-Score | 0.93 |
| `High_Risk` recall | 64% (reported in the business-impact summary) |

### Regression

| Metric | Result |
|---|---:|
| RMSE | ₹1,053.65 |
| R² | 0.9817 |

The report compares Decision Tree, Random Forest, and tuned XGBoost models. Tuned XGBoost achieved the best reported result for both classification and regression.

## 🧹 Data Processing and Feature Engineering

The data preparation workflow includes:

- Converting incorrectly typed fields, such as `monthly_salary`, into numeric values.
- Removing text fragments such as currency labels and units where necessary.
- Imputing numerical missing values using the median.
- Imputing categorical missing values using the mode.
- Creating `total_monthly_expenses`.
- Creating `disposable_income`.
- Creating `dti_ratio`.
- Dropping the corrupted `age` feature, which contained implausible values.

### Preprocessing Pipeline

The project uses a `ColumnTransformer` to keep preprocessing consistent between training and inference.

**Numerical features:**

- Median imputation
- Log transformation for skewed features
- Standard scaling

**Categorical features:**

- Mode imputation
- One-hot encoding with `handle_unknown="ignore"`

## 🔍 Model Explainability

The report describes using SHAP `TreeExplainer` on the final classifier.

The analysis identified these influential features:

1. `disposable_income`
2. `requested_amount`
3. `dti_ratio`

This provides insight into which engineered financial features influence the classifier's predictions.

## 🧪 MLflow Experiment Tracking

MLflow was used to record both final models, including parameters, evaluation metrics, and model artifacts. The report describes running an MLflow server through ngrok during experimentation.

Reported metrics include:

- Classification Macro F1-Score: `0.81`
- Regression RMSE: `1053.65`

## 🖥️ Application

The Streamlit application is implemented in `app.py`.

According to the report, the application:

- Loads the saved preprocessing pipeline, classifier, regressor, and label encoder.
- Collects applicant information through a three-column input interface.
- Performs the same feature engineering used during training.
- Passes the processed input to both models.
- Displays the eligibility category and estimated maximum affordable EMI.

## 📁 Key Project Artifacts

The report identifies these deployment files:

```text
app.py
emi_preprocessor.joblib
emi_classifier_model.joblib
emi_regressor_model.joblib
label_encoder.joblib
requirements.txt
```

The report specifies a `scikit-learn==1.6.1` version lock in `requirements.txt`.

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Classification | XGBoost |
| Regression | XGBoost |
| Data Processing | Pandas |
| Preprocessing | Scikit-learn |
| Model Explainability | SHAP |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

## 🚀 Setup and Deployment

The project report documents the deployment artifacts and hosting platform, but does not provide a complete list of installation commands or the full `requirements.txt` contents.

For an accurate local setup:

1. Clone the repository.
2. Install the dependencies specified in `requirements.txt`.
3. Ensure the required `.joblib` artifacts are available.
4. Run the Streamlit application using the repository's entry point.

The report identifies `app.py` as the application entry point. A typical launch command is:

```bash
streamlit run app.py
```

For hosted deployment, the report describes uploading `app.py`, the four model artifacts, and `requirements.txt` to GitHub and deploying through Streamlit Community Cloud.

## ⚠️ Limitations and Considerations

- The classifier's reported `High_Risk` F1-Score is `0.53`, so high-risk predictions should be reviewed carefully.
- The training data has substantial class imbalance.
- The report identifies further tuning on the full training set as a potential improvement.
- The application provides model-based estimates and should support, not replace, a lender's broader review and decision process.

## 🔮 Future Scope

The report identifies these potential improvements:

- **Further model tuning:** Run Bayesian optimization on the full 323,000-row training set rather than a 50,000-row sample.
- **What-if analysis:** Allow users to explore how paying off an existing loan or changing salary could affect predicted eligibility.
- **Additional financial data:** Incorporate sources such as CIBIL transaction history to enrich the model's predictive features.

## 👨‍💻 Project Information

- **Project:** EMIPredict AI
- **Author:** Piyush Chavda
- **Report Date:** November 12, 2025
- **Purpose:** EMI eligibility classification and maximum affordable EMI estimation

## 🔗 Project Links

- **GitHub:** https://github.com/PiyushChavda595/EMI-Predict-AI
