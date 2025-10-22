import ctypes
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# --- 1. Constants ---
# This MUST match the #define in the C code
MAX_FEATURES = 3 

# --- 2. Load C Library & Define Signatures ---
try:
    lib_path = os.path.abspath('./reco_lib.so')
    lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded library from: {lib_path}")
except OSError as e:
    print(f"FATAL ERROR: Could not load reco_lib.so.")
    print(f"Details: {e}")
    print("Make sure you compiled 'reco_lib.c' first:")
    print("  gcc -shared -o reco_lib.so -fPIC reco_lib.c -lm")
    exit()

# C: float predictor(float* new_item)
lib.predictor.argtypes = [
    # We must use float32 to match C's 'float'
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS") 
]
lib.predictor.restype = ctypes.c_float # C's 'float' maps to c_float

# --- 3. FastAPI App & Pydantic Model ---

# No lifespan manager is needed, the C lib has no "state" to set up.
app = FastAPI()

class ItemFeatures(BaseModel):
    # We expect a JSON body like: {"features": [1.0, 0.0, 1.0]}
    features: List[float]

# --- 4. API Endpoint ---
@app.post("/predict")
async def predict_rating(item: ItemFeatures):
    """
    Predicts the user rating for a new item based on its features.
    """
    # --- Input Validation ---
    if len(item.features) != MAX_FEATURES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feature vector. Expected {MAX_FEATURES} features, but got {len(item.features)}."
        )
    
    # --- Data Conversion ---
    # Convert Python list[float] (which are 64-bit) 
    # to a NumPy array of float32 (which matches C's 'float')
    features_array = np.array(item.features, dtype=np.float32)

    # --- Call C Function ---
    predicted_rating = lib.predictor(features_array)

    # --- Return Result ---
    return {
        "input_features": item.features,
        "predicted_rating": f"{predicted_rating:.4f}"
    }

@app.get("/")
async def root():
    return {"message": "Content-based filtering API is running. POST to /predict"}