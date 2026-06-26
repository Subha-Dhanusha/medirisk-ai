# src/api/main.py

import joblib
import numpy as np
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PatientInput, PredictionOutput
from src.features.engineer import create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MediRisk AI",
    description="Hospital Readmission Risk Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = "models/xgboost_readmission.joblib"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/")
def root():
    return {
        "name": "MediRisk AI",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to dataframe
        input_dict = patient.model_dump()
        df = pd.DataFrame([input_dict])

        # Run feature engineering
        df = create_features(df)

        # Align columns with training data
        expected_cols = model.get_booster().feature_names
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Predict
        prob = model.predict_proba(df)[0][1]
        risk_score = round(prob * 100, 1)

        # Risk level
        if prob >= 0.4:
            risk_level = "HIGH"
            recommendation = "Immediate care coordination required. Schedule follow-up within 48 hours."
        elif prob >= 0.2:
            risk_level = "MEDIUM"
            recommendation = "Monitor closely. Schedule follow-up within 7 days."
        else:
            risk_level = "LOW"
            recommendation = "Standard discharge protocol. Routine follow-up in 30 days."

        # Top risk factors from model
        feature_names = model.get_booster().feature_names
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        top_factors = [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
            for i in top_indices
        ]

        return PredictionOutput(
            risk_score=risk_score,
            risk_level=risk_level,
            readmission_probability=round(float(prob), 4),
            recommendation=recommendation,
            top_risk_factors=top_factors
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))