# Telecom Customer Churn Prediction

### End-to-End Machine Learning Project with FastAPI Deployment

---

## 1. Project Overview

Customer churn is a major challenge in the telecom industry. Retaining existing customers is significantly cheaper than acquiring new ones.

This project builds an "end-to-end churn prediction system" that not only predicts whether a customer is likely to churn, but also recommends business actions based on churn risk.

The project covers the "entire machine learning lifecycle":

* Data preprocessing
* Model training and evaluation
* Cost-sensitive decision making
* Model explainability
* Production-ready API deployment using "FastAPI"

---

## 2. Business Problem

Goal:
Predict customer churn and help the business take proactive retention actions.

Challenges:

* Churn is influenced by complex, non-linear factors
* False negatives (missing a churner) are more costly than false positives
* Predictions must be explainable and actionable

---

## 3. Dataset Description

The dataset contains customer demographic, service usage, and billing information.

### Key Features

* Demographics: gender, senior citizen, partner, dependents
* Account information: tenure, contract type, payment method
* Services: internet service, tech support, streaming services
* Billing: monthly charges, total charges
* Target variable:`Churn` (Yes / No)

---

## 4. Machine Learning Approach

### 4.1 Data Preprocessing

* Missing values handled
* Numerical features scaled using `StandardScaler`
* Categorical features encoded using `OneHotEncoder`
* Preprocessing implemented using `ColumnTransformer`
* Data leakage avoided using pipelines

---

### 4.2 Models Used

#### Baseline Model

* Logistic Regression
* Used for interpretability and baseline comparison

#### Advanced Model

* Random Forest Classifier
* Captures non-linear feature interactions
* Handles complex churn behavior better than linear models

---

### 4.3 Model Evaluation

Evaluation metrics:

* Precision
* Recall
* F1-Score
* ROC-AUC

Key Insight:
Accuracy alone is not sufficient for churn prediction. Recall and business cost are more important.

---

## 5. Cost-Sensitive Threshold Optimization

Instead of using the default 0.5 probability threshold, a business-aware threshold was optimized.

### Business Costs

* **False Negative (missed churn):** High revenue loss
* **False Positive (unnecessary offer):** Lower cost

### Result

* Threshold optimized to minimize expected business loss
* Model intentionally prioritizes capturing churn customers
* Improves real-world usefulness of predictions

---

## 6. Explainability (SHAP)

To ensure transparency and trust:

* **SHAP (SHapley Additive exPlanations)** used
* Global explanations identify key churn drivers
* Local explanations explain individual customer predictions

This ensures the model is:

* Explainable
* Auditable
* Business-friendly

---

## 7. Decision Intelligence Layer

Predictions are translated into actionable business decisions.

| Churn Probability | Risk Level | Recommended Action            |
| ----------------- | ---------- | ----------------------------- |
| < 0.30            | Low        | No Action                     |
| 0.30 – 0.60       | Medium     | Email / SMS Discount          |
| > 0.60            | High       | Call Center + Retention Offer |

This converts raw ML outputs into operational decisions.

---

## 8. Application Layer (FastAPI)

The model is deployed as a FastAPI inference service.

### Features

* REST API for predictions
* Input validation using Pydantic
* JSON request / response
* Swagger UI for easy testing

### API Endpoints

* `GET /` → Health check
* `POST /predict` → Churn prediction and action recommendation

---


## 9. How to Run the Application

### 9.1 Install Dependencies

```bash
python -m pip install fastapi uvicorn pandas scikit-learn joblib
```

---

### 9.2 Run the FastAPI Server

From the project root directory:

```bash
python -m uvicorn app.main:app --reload
```

---

### 9.3 Access the API

Open your browser:

```
http://127.0.0.1:8000/docs
```

Swagger UI will open, allowing you to test the API interactively.

---

## 10. Sample Prediction Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5
}
```

### Sample Response

```json
{
  "churn_probability": 0.742,
  "risk_level": "High",
  "recommended_action": "Call Center + Retention Offer"
}
```

---

## 11. Key Outcomes

* Built a complete ML pipeline from data to deployment
* Implemented cost-sensitive decision making
* Ensured explainability using SHAP
* Deployed a production-ready API using FastAPI
* Created a business-aligned churn prediction system

---

## 12. Conclusion

This project demonstrates the ability to:

* Solve real-world business problems using machine learning
* Go beyond accuracy and focus on business impact
* Deploy models in a production-ready manner
* Communicate results clearly to both technical and non-technical stakeholders

---

