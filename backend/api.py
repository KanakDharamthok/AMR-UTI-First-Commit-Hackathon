import logging
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import joblib

# Import your ML modules
from src.module2_features.reduction import parse_temporal_features
from src.module3_execution.quantum_circuit import HybridQuantumClassicalModel
from src.module3_execution.qsvm_fallback import extract_mps_embeddings

# Import Hackathon Track Modules (Cedar Auth & OpenSearch)
from src.module4_auth.cedar_policy import CedarPolicyEngine
from src.module0_opensearch.opensearch_client import get_opensearch_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AMR-UTI Quantum API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models in memory
scaler = None
hybrid_model = None
xgb_baseline = None
feature_names = []

@app.on_event("startup")
def load_artifacts():
    global scaler, hybrid_model, xgb_baseline, feature_names
    logger.info("Loading ML Artifacts into memory...")
    
    try:
        scaler = joblib.load("models/standard_scaler.pkl")
        feature_names = list(scaler.feature_names_in_)
        
        # Try loading PyTorch Hybrid Model
        try:
            dummy_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
            feature_groups, static_features = parse_temporal_features(dummy_df)
            time_order = ['ALL', '180', '90', '30', '14', '7']
            active_steps = [t for t in time_order if len(feature_groups[t]) > 0]
            
            input_dims = [len(feature_groups[t]) for t in active_steps]
            static_dim = len(static_features)
            
            hybrid_model = HybridQuantumClassicalModel(mps_input_dims=input_dims, static_dim=static_dim, n_layers=3)
            hybrid_model.load_state_dict(torch.load("models/hybrid_mps_qnn.pt", map_location='cpu'))
            hybrid_model.eval()
            logger.info(f"✅ Hybrid Model Loaded Successfully (Dims: {input_dims}, Static: {static_dim})")
        except Exception as model_err:
            logger.warning(f"Hybrid model not loaded: {model_err}")

    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")

# Dynamic Pydantic Model (Accepts flexible key-value clinical data)
class ClinicalPayload(BaseModel):
    data: dict

@app.post("/predict")
def predict_susceptibility(
    payload: ClinicalPayload,
    x_user_role: str = Header("AttendingPhysician", description="User role for Cedar policy evaluation")
):
    # 1. Evaluate access using Cedar Policy Engine
    authorized = CedarPolicyEngine.evaluate_access(
        user_role=x_user_role,
        action="run_quantum_pipeline",
        resource="AMR_Quantum_Model"
    )
    
    if not authorized:
        raise HTTPException(status_code=403, detail="Access denied by Cedar security policy.")

    # 2. Optional: Log query telemetry to OpenSearch (non-blocking)
    try:
        os_client = get_opensearch_client()
        os_client.index(
            index="amr-query-logs",
            body={"role": x_user_role, "status": "authorized_and_executing"}
        )
    except Exception:
        pass 

    if not scaler:
        raise HTTPException(status_code=500, detail="Scaler not loaded.")

    try:
        # 3. Pad missing features with 0 to match scaler expectations
        padded_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        for col, val in payload.data.items():
            if col in feature_names:
                padded_df.at[0, col] = val

        # 4. Scale features
        X_scaled = scaler.transform(padded_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        resistance_prob = 0.5
        engine_used = "Unknown"

        # 5. Route Inference (REAL INFERENCE LOGIC)
        if hybrid_model:
            feature_groups, static_features = parse_temporal_features(X_scaled_df)
            time_order = ['ALL', '180', '90', '30', '14', '7']
            active_steps = [t for t in time_order if len(feature_groups[t]) > 0]
            
            X_seq = [torch.tensor(X_scaled_df[feature_groups[t]].values, dtype=torch.float32) for t in active_steps]
            X_static = torch.tensor(X_scaled_df[static_features].values, dtype=torch.float32) if static_features else None

            with torch.no_grad():
                resistance_prob = float(hybrid_model(X_seq, X_static).item())
            engine_used = "Hybrid MPS-QNN"
            
        else:
            engine_used = "Classical Fallback / Unknown"
            resistance_prob = 0.50

        # 6. Calculate final metrics
        susceptibility_score = round((1.0 - resistance_prob) * 100, 2)
        entropy = round(1.0 - abs(resistance_prob - 0.5) * 2, 3)

        return {
            "susceptibility_score": susceptibility_score,
            "entropy_uncertainty": entropy,
            "engine_used": engine_used,
            "is_resistant": resistance_prob >= 0.30
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))