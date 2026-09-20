import os
import json
import torch
import joblib
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import xgboost as xgb
import sys

# Ensure src modules can be imported
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from module2_features.reduction import parse_temporal_features
from module3_execution.quantum_circuit import HybridQuantumClassicalModel
from module3_execution.qsvm_fallback import extract_mps_embeddings

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AMR-UTI Quantum Pipeline API")

# Allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
standard_scaler = None
inference_model = None
hybrid_model = None
qsvm_model = None
qsvm_embedder = None

def fetch_s3_artifacts():
    """Downloads model artifacts from S3 on startup."""
    bucket_name = "amr-uti-models" # TODO: Update with your exact AWS S3 bucket name
    s3_prefix = "models/"
    required_files = ["standard_scaler.pkl", "hybrid_mps_qnn.pt", "xgb_baseline.json", "qsvm_weights.pkl"]
    
    s3 = boto3.client('s3')
    for file in required_files:
        local_path = MODEL_DIR / file
        if not local_path.exists():
            try:
                print(f"Downloading {file} from S3...")
                s3.download_file(bucket_name, f"{s3_prefix}{file}", str(local_path))
            except Exception as e:
                print(f"Skipping {file}: Not found in S3 or access denied.")

@app.on_event("startup")
def load_models():
    global standard_scaler, inference_model, hybrid_model, qsvm_model, qsvm_embedder
    
    fetch_s3_artifacts()
    
    scaler_path = MODEL_DIR / "standard_scaler.pkl"
    if not scaler_path.exists():
        print("Warning: standard_scaler.pkl not found. API will fail.")
        return

    standard_scaler = joblib.load(scaler_path)

    # 1. Load XGBoost (Optional)
    xgb_path = MODEL_DIR / "xgb_baseline.json"
    if xgb_path.exists():
        try:
            temp_xgb = xgb.XGBClassifier()
            temp_xgb.load_model(xgb_path)
            if set(temp_xgb.get_booster().feature_names or []) == set(standard_scaler.feature_names_in_):
                inference_model = temp_xgb
        except Exception:
            pass

    # 2. Load Hybrid MPS-QNN
    hybrid_path = MODEL_DIR / "hybrid_mps_qnn.pt"
    if hybrid_path.exists():
        feature_names = list(standard_scaler.feature_names_in_)
        dummy_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        feature_groups, static_features = parse_temporal_features(dummy_df)
        active_steps = [t for t in ['ALL', '180', '90', '30', '14', '7'] if len(feature_groups[t]) > 0]
        
        hybrid_model = HybridQuantumClassicalModel(
            mps_input_dims=[len(feature_groups[t]) for t in active_steps], 
            static_dim=len(static_features), 
            n_layers=3
        )
        hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=torch.device('cpu')))
        hybrid_model.eval()
        hybrid_model.static_feature_names = static_features

class PatientData(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
async def predict(data: PatientData):
    if standard_scaler is None:
        raise HTTPException(status_code=500, detail="Core artifacts missing.")

    feature_names = list(standard_scaler.feature_names_in_)
    padded_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    for col, val in data.features.items():
        if col in feature_names:
            padded_df.at[0, col] = val

    X_scaled = standard_scaler.transform(padded_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    resistance_prob = 0.5
    engine_used = "Unknown"

    if hybrid_model is not None:
        feature_groups, _ = parse_temporal_features(X_scaled_df)
        active_steps = [t for t in ['ALL', '180', '90', '30', '14', '7'] if len(feature_groups[t]) > 0]
        X_seq = [torch.tensor(X_scaled_df[feature_groups[t]].values, dtype=torch.float32) for t in active_steps]
        
        static_names = getattr(hybrid_model, "static_feature_names", [])
        X_static = torch.tensor(X_scaled_df[static_names].values, dtype=torch.float32) if static_names else None

        with torch.no_grad():
            resistance_prob = float(hybrid_model(X_seq, X_static).item())
        engine_used = "Hybrid MPS-QNN"
    elif inference_model is not None:
        resistance_prob = float(inference_model.predict_proba(X_scaled_df)[0, 1])
        engine_used = "Penalized XGBoost Baseline"

    return {
        "susceptibility_score": 1.0 - resistance_prob,
        "resistance_probability": resistance_prob,
        "uncertainty_entropy": 1.0 - abs(resistance_prob - 0.5) * 2,
        "engine_used": engine_used
    }