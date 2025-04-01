from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import sys
import pickle
import pandas as pd
from typing import List, Optional, Dict, Any

# Add src directory to Python path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules
from src.utils.config import load_config
from src.models.model import BaseModel as MimicBaseModel

# --- Global Variables for Loaded Model Components ---
# These will be populated at startup
model_data: Dict[str, Any] = {}
model_instance: Optional[MimicBaseModel] = None
expected_features: List[str] = []


app = FastAPI(
    title="MIMIC Readmission Predictor API",
    description="API to predict 30-day hospital readmission risk using the MIMIC demo dataset.",
    version="0.1.0"
)

# --- Configuration and Model Loading ---
config = None
MODEL_PATH = "models/readmission_model.pkl"  # Path relative to project root

# --- Startup Event Handler ---
@app.on_event("startup")
async def load_model_on_startup():
    """Load the model, scaler, and feature names when the API starts."""
    global model_data, model_instance, expected_features, config
    try:
        # Try to load config
        try:
            config = load_config()
            print("Configuration loaded successfully.")
        except Exception as config_e:
            print(f"Warning: Configuration file not found or invalid: {config_e}. Using defaults.")
            config = {} # Use empty dict if config fails

        # Construct absolute path for the model
        model_abs_path = os.path.join(project_root, MODEL_PATH)

        if not os.path.exists(model_abs_path):
            print(f"CRITICAL ERROR: Model file not found at: {model_abs_path}")
            # Keep globals empty/None, health check will fail
            return

        # Load the entire model object (which includes model, scaler, features)
        model_instance = MimicBaseModel.load(model_abs_path)
        expected_features = model_instance.feature_names
        # Store components in model_data for potential direct access if needed
        model_data['model'] = model_instance.model
        model_data['scaler'] = model_instance.scaler
        model_data['feature_names'] = model_instance.feature_names

        print(f"Model loaded successfully from {model_abs_path}")
        print(f"Model Type: {model_instance.model_type}")
        print(f"Expected Features ({len(expected_features)}): {expected_features[:5]}...") # Print first 5

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Model file or dependency not found during startup: {e}")
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing expected key in model file ('model', 'scaler', or 'feature_names'): {e}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model during startup: {e}")
        # Reset globals to ensure health check fails clearly
        model_data = {}
        model_instance = None
        expected_features = []

@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the MIMIC Readmission Predictor API"}

@app.get("/health", summary="Check API Health")
async def health_check():
    """Check if the API is running and the model is loaded."""
    """Check if the API is running and the model components are loaded."""
    if model_instance and expected_features and model_data.get('scaler') and model_data.get('model'):
        return {
            "status": "ok",
            "message": "API is running and model components are loaded.",
            "model_type": model_instance.model_type,
            "expected_features_count": len(expected_features)
        }
    else:
        error_details = []
        if not model_instance: error_details.append("Model instance not loaded.")
        if not expected_features: error_details.append("Expected features list not loaded.")
        if not model_data.get('scaler'): error_details.append("Scaler not loaded.")
        if not model_data.get('model'): error_details.append("Base model not loaded.")
        raise HTTPException(status_code=503,
                           detail=f"API is unhealthy. Failed to load model components: {'; '.join(error_details)}")

# --- Prediction Endpoint ---
@app.post("/predict", summary="Predict 30-Day Readmission Risk")
async def predict_readmission(patient_data: Dict[str, Any]):
    """
    Predicts the probability of 30-day hospital readmission based on patient features provided as a JSON object.

    - **Input**: A JSON object where keys are feature names and values are the corresponding feature values.
                 The API will attempt to match these features against the model's expected features. Missing features will be imputed with 0.
    - **Output**: Predicted probability of readmission (float between 0 and 1).
    """
    if not model_instance or not expected_features:
        raise HTTPException(status_code=503, detail="Model not loaded properly. Cannot make predictions. Check API health.")

    try:
        # 1. Validate Input Data Structure
        if not isinstance(patient_data, dict):
             raise HTTPException(status_code=400, detail="Invalid input format. Expected a JSON object (dictionary).")

        # 2. Prepare DataFrame with expected features
        # Create a DataFrame from the input dict
        input_df_raw = pd.DataFrame([patient_data])

        # Check for unexpected features provided by the user
        provided_features = set(input_df_raw.columns)
        model_feature_set = set(expected_features)
        unexpected_features = provided_features - model_feature_set
        if unexpected_features:
            print(f"Warning: Received unexpected features, ignoring: {list(unexpected_features)}")
            # Optionally, raise an error if strict matching is required:
            # raise HTTPException(status_code=400, detail=f"Received unexpected features: {list(unexpected_features)}")

        # Reindex to ensure all expected columns are present, in the correct order,
        # filling missing features with 0 (consistent with previous logic).
        try:
            input_df = input_df_raw.reindex(columns=expected_features, fill_value=0)
        except Exception as reindex_e:
             # This might happen with very unusual column names, though unlikely
             print(f"Error during DataFrame reindexing: {reindex_e}")
             raise HTTPException(status_code=500, detail=f"Internal server error during data preparation (reindexing).")


        # 3. Perform Scaling
        try:
            scaled_features = model_instance.scaler.transform(input_df)
        except ValueError as scale_e:
             # This might happen if data types are wrong after reindexing/filling
             print(f"Error during scaling: {scale_e}")
             raise HTTPException(status_code=400, detail=f"Data scaling error. Check input data types. Details: {scale_e}")
        except Exception as scale_e:
             print(f"Unexpected error during scaling: {scale_e}")
             raise HTTPException(status_code=500, detail=f"Internal server error during data scaling.")

        # 4. Make Prediction
        try:
            prediction_proba = model_instance.model.predict_proba(scaled_features)
            # Assuming binary classification, get probability of the positive class (index 1)
            readmission_probability = prediction_proba[0, 1]
        except AttributeError as predict_e:
             print(f"Error calling predict_proba (model structure issue?): {predict_e}")
             raise HTTPException(status_code=500, detail="Internal server error: Model prediction method not found.")
        except Exception as predict_e:
             print(f"Unexpected error during prediction: {predict_e}")
             raise HTTPException(status_code=500, detail="Internal server error during prediction.")

        # 5. Format and return the prediction
        return {"predicted_readmission_probability": float(readmission_probability)}

    except FileNotFoundError as e:
        # Catch specific exceptions from above, plus general ones
        raise HTTPException(status_code=500, detail=f"Model file or essential component not found: {e}")
    except KeyError as e:
        # This might occur if the input dict processing fails unexpectedly
        print(f"Internal key error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error processing input features: {e}")
    except ValueError as e:
        # More specific ValueErrors are caught above, this is a fallback
        print(f"Generic ValueError during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input data value or type: {e}")
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected prediction error: {type(e).__name__} - {e}")
        # Consider logging traceback here: import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred.")


if __name__ == "__main__":
    # Run the API using Uvicorn - Get host/port from config if available
    # Use loaded config (or default if loading failed)
    api_host = config.get('api', {}).get('host', '127.0.0.1') # Default to localhost for safety
    api_port = config.get('api', {}).get('port', 8001) # Default to 8001
    print(f"Starting API server on {api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port)