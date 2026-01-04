from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Decision Logic
# -----------------------------
def assign_risk_and_action(prob):
    if prob < 0.30:
        return "Low", "No Action"
    elif prob < 0.60:
        return "Medium", "Email / SMS Discount"
    else:
        return "High", "Call Center + Retention Offer"

# -----------------------------
# FastAPI app (DO NOT RENAME)
# -----------------------------
app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict_churn(data: CustomerInput):

    df = pd.DataFrame([{
        **data.dict(),
        "TotalCharges": data.MonthlyCharges * data.tenure
    }])

    churn_prob = model.predict_proba(df)[0][1]
    risk, action = assign_risk_and_action(churn_prob)

    return {
        "churn_probability": round(churn_prob, 3),
        "risk_level": risk,
        "recommended_action": action
    }
