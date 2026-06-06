
# api/main.py - FastAPI Application for Fraud Detection

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import json
from xgboost import XGBClassifier

# Initialize app
app = FastAPI(
    title='Fraud Detection API',
    description='Real-time credit card fraud detection using XGBoost',
    version='1.0.0'
)

# Load models at startup
model = XGBClassifier()
model.load_model('models/xgb_model.json')

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Define input schema
class Transaction(BaseModel):
    TransactionAmt: float = Field(..., example=125.50, description='Amount in USD')
    ProductCD: str = Field(..., example='W', description='Product code')
    card4: str = Field(..., example='visa', description='Card network')
    hour: int = Field(..., example=14, description='Hour of day 0-23')
    C1: float = Field(default=1.0, description='Count feature')
    C2: float = Field(default=1.0, description='Count feature')
    addr1: float = Field(default=299.0, description='Billing address code')

# Define output schema
class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold_used: float

# Health check
@app.get('/')
def root():
    return {'status': 'ok', 'model': 'XGBoost Fraud Detector v1.0'}

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': model is not None}

# Main prediction endpoint
@app.post('/predict', response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        # Build feature array
        features = np.array([[
            transaction.TransactionAmt,
            np.log1p(transaction.TransactionAmt),
            transaction.hour,
            transaction.C1,
            transaction.C2,
            transaction.addr1,
        ]])
        
        # Get fraud probability
        prob = float(model.predict_proba(features)[0, 1])
        
        # Determine risk level
        if prob < 0.3:
            risk = 'LOW'
        elif prob < 0.7:
            risk = 'MEDIUM'
        else:
            risk = 'HIGH'
        
        return PredictionResponse(
            fraud_probability=prob,
            is_fraud=prob >= 0.5,
            risk_level=risk,
            threshold_used=0.5
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
