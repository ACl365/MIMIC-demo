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

# --- Pydantic Model for Input Data ---
# IMPORTANT: This is a placeholder! Update with the actual features your model expects.
# Field descriptions should clarify units or expected values.
# --- Pydantic Model for Input Data ---
# IMPORTANT: This model MUST be updated to reflect the *exact* features
#            found in the 'feature_names' list within the loaded readmission_model.pkl file.
#            All features should likely be Optional[float] or Optional[int] to handle
#            potential missing inputs, which will be imputed to 0.
class PatientFeatures(BaseModel):
    # --- Features based on readmission_model.pkl ---
    age: Optional[float] = Field(None, example=65.0, description="Age in years")
    gender_f: Optional[int] = Field(None, example=1, description="Gender: Female (1 if true, 0 otherwise)")
    gender_m: Optional[int] = Field(None, example=0, description="Gender: Male (1 if true, 0 otherwise)")
    gender_nan: Optional[int] = Field(None, example=0, description="Gender: Unknown/NaN (1 if true, 0 otherwise)")
    admission_type_ambulatory_observation: Optional[int] = Field(None, alias="admission_type_ambulatory observation", example=0, description="Admission Type: Ambulatory Observation (1 if true, 0 otherwise)")
    admission_type_direct_emer: Optional[int] = Field(None, alias="admission_type_direct emer.", example=0, description="Admission Type: Direct Emergency (1 if true, 0 otherwise)")
    admission_type_direct_observation: Optional[int] = Field(None, alias="admission_type_direct observation", example=0, description="Admission Type: Direct Observation (1 if true, 0 otherwise)")
    admission_type_elective: Optional[int] = Field(None, example=0, description="Admission Type: Elective (1 if true, 0 otherwise)")
    admission_type_emergency: Optional[int] = Field(None, example=1, description="Admission Type: Emergency (1 if true, 0 otherwise)")
    admission_type_eu_observation: Optional[int] = Field(None, alias="admission_type_eu observation", example=0, description="Admission Type: EU Observation (1 if true, 0 otherwise)")
    admission_type_ew_emer: Optional[int] = Field(None, alias="admission_type_ew emer.", example=0, description="Admission Type: EW Emergency (1 if true, 0 otherwise)")
    admission_type_observation_admit: Optional[int] = Field(None, alias="admission_type_observation admit", example=0, description="Admission Type: Observation Admit (1 if true, 0 otherwise)")
    admission_type_surgical_same_day_admission: Optional[int] = Field(None, alias="admission_type_surgical same day admission", example=0, description="Admission Type: Surgical Same Day (1 if true, 0 otherwise)")
    admission_type_urgent: Optional[int] = Field(None, example=0, description="Admission Type: Urgent (1 if true, 0 otherwise)")
    admission_type_nan: Optional[int] = Field(None, example=0, description="Admission Type: Unknown/NaN (1 if true, 0 otherwise)")
    insurance_government: Optional[int] = Field(None, example=0, description="Insurance: Government (1 if true, 0 otherwise)")
    insurance_medicaid: Optional[int] = Field(None, example=0, description="Insurance: Medicaid (1 if true, 0 otherwise)")
    insurance_medicare: Optional[int] = Field(None, example=1, description="Insurance: Medicare (1 if true, 0 otherwise)")
    insurance_other: Optional[int] = Field(None, example=0, description="Insurance: Other (1 if true, 0 otherwise)")
    insurance_private: Optional[int] = Field(None, example=0, description="Insurance: Private (1 if true, 0 otherwise)")
    insurance_nan: Optional[int] = Field(None, example=0, description="Insurance: Unknown/NaN (1 if true, 0 otherwise)")
    marital_status_divorced: Optional[int] = Field(None, example=0, description="Marital Status: Divorced (1 if true, 0 otherwise)")
    marital_status_married: Optional[int] = Field(None, example=1, description="Marital Status: Married (1 if true, 0 otherwise)")
    marital_status_separated: Optional[int] = Field(None, example=0, description="Marital Status: Separated (1 if true, 0 otherwise)")
    marital_status_single: Optional[int] = Field(None, example=0, description="Marital Status: Single (1 if true, 0 otherwise)")
    marital_status_unknown_default: Optional[int] = Field(None, alias="marital_status_unknown (default)", example=0, description="Marital Status: Unknown (1 if true, 0 otherwise)")
    marital_status_widowed: Optional[int] = Field(None, example=0, description="Marital Status: Widowed (1 if true, 0 otherwise)")
    marital_status_nan: Optional[int] = Field(None, example=0, description="Marital Status: Unknown/NaN (1 if true, 0 otherwise)")
    feature_0: Optional[int] = Field(None, alias='0', example=0, description="Unnamed feature '0' (check origin)") # Handle the '0' feature name
    diagnosis_other: Optional[int] = Field(None, example=0, description="Diagnosis Group: Other (1 if true, 0 otherwise)")
    diagnosis_e_external_causes: Optional[int] = Field(None, example=0, description="Diagnosis Group: External Causes (1 if true, 0 otherwise)")
    diagnosis_v_supplementary: Optional[int] = Field(None, example=0, description="Diagnosis Group: Supplementary (1 if true, 0 otherwise)")
    diagnosis_digestive: Optional[int] = Field(None, example=0, description="Diagnosis Group: Digestive (1 if true, 0 otherwise)")
    diagnosis_nervous: Optional[int] = Field(None, example=0, description="Diagnosis Group: Nervous (1 if true, 0 otherwise)")
    diagnosis_respiratory: Optional[int] = Field(None, example=0, description="Diagnosis Group: Respiratory (1 if true, 0 otherwise)")
    diagnosis_circulatory: Optional[int] = Field(None, example=1, description="Diagnosis Group: Circulatory (1 if true, 0 otherwise)")
    diagnosis_mental: Optional[int] = Field(None, example=0, description="Diagnosis Group: Mental (1 if true, 0 otherwise)")
    diagnosis_musculoskeletal: Optional[int] = Field(None, example=0, description="Diagnosis Group: Musculoskeletal (1 if true, 0 otherwise)")
    diagnosis_injury: Optional[int] = Field(None, example=0, description="Diagnosis Group: Injury (1 if true, 0 otherwise)")
    diagnosis_endocrine: Optional[int] = Field(None, example=0, description="Diagnosis Group: Endocrine (1 if true, 0 otherwise)")
    diagnosis_genitourinary: Optional[int] = Field(None, example=0, description="Diagnosis Group: Genitourinary (1 if true, 0 otherwise)")
    diagnosis_neoplasms: Optional[int] = Field(None, example=0, description="Diagnosis Group: Neoplasms (1 if true, 0 otherwise)")

    class Config:
        schema_extra = {
            "example": {
                "age": 72.0,
                "gender_f": 1,
                "gender_m": 0,
                "gender_nan": 0,
                "admission_type_ambulatory observation": 0,
                "admission_type_direct emer.": 0,
                "admission_type_direct observation": 0,
                "admission_type_elective": 0,
                "admission_type_emergency": 1,
                "admission_type_eu observation": 0,
                "admission_type_ew emer.": 0,
                "admission_type_observation admit": 0,
                "admission_type_surgical same day admission": 0,
                "admission_type_urgent": 0,
                "admission_type_nan": 0,
                "insurance_government": 0,
                "insurance_medicaid": 0,
                "insurance_medicare": 1,
                "insurance_other": 0,
                "insurance_private": 0,
                "insurance_nan": 0,
                "marital_status_divorced": 0,
                "marital_status_married": 1,
                "marital_status_separated": 0,
                "marital_status_single": 0,
                "marital_status_unknown (default)": 0,
                "marital_status_widowed": 0,
                "marital_status_nan": 0,
                "0": 0, # Example for the '0' feature
                "diagnosis_other": 0,
                "diagnosis_e_external_causes": 0,
                "diagnosis_v_supplementary": 0,
                "diagnosis_digestive": 0,
                "diagnosis_nervous": 0,
                "diagnosis_respiratory": 0,
                "diagnosis_circulatory": 1,
                "diagnosis_mental": 0,
                "diagnosis_musculoskeletal": 0,
                "diagnosis_injury": 0,
                "diagnosis_endocrine": 0,
                "diagnosis_genitourinary": 0,
                "diagnosis_neoplasms": 0
            }
        }


app = FastAPI(
    title="MIMIC Readmission Predictor API",
    description="API to predict 30-day hospital readmission risk using the MIMIC demo dataset.",
    version="0.1.0"
)

# --- Configuration and Model Loading ---
config = None
model: Optional[Any] = None
MODEL_PATH = "models/readmission_model.pkl"  # Path relative to project root

try:
    # Try to load config, but continue even if it fails
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except Exception as config_e:
        print(f"Warning: Configuration file not found: {config_e}")
        print("Using default configuration")
        config = None
    
    # Construct absolute path for the model
    model_abs_path = os.path.join(project_root, MODEL_PATH)

    if not os.path.exists(model_abs_path):
        raise FileNotFoundError(f"Model file not found at: {model_abs_path}")
    
    # Verify the model file can be loaded
    with open(model_abs_path, "rb") as f:
        model_data = pickle.load(f)
        
    # Just store the raw model for health check purposes
    # We'll create a proper model instance during prediction
    model = model_data["model"]
    feature_count = len(model_data["feature_names"])
    print(f"Model verified with {feature_count} features from {model_abs_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Allow API to start but prediction will fail
    model = None

@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the MIMIC Readmission Predictor API"}

@app.get("/health", summary="Check API Health")
async def health_check():
    """Check if the API is running and the model is loaded."""
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. API is unhealthy.")
    
    try:
        # Try to load the full model to verify all components
        model_instance = MimicBaseModel.load(os.path.join(project_root, MODEL_PATH))
        return {
            "status": "ok",
            "model_loaded": True,
            "feature_count": len(model_instance.feature_names),
            "model_type": model_instance.model_type
        }
    except Exception as e:
        raise HTTPException(status_code=503,
                           detail=f"Model verification failed: {str(e)}")

# --- Prediction Endpoint ---
@app.post("/predict", summary="Predict 30-Day Readmission Risk")
async def predict_readmission(patient_data: PatientFeatures):
    """
    Predicts the probability of 30-day hospital readmission based on patient features.

    - **Input**: Patient features matching the `PatientFeatures` schema.
    - **Output**: Predicted probability of readmission.

    *Note: Ensure input features match those used during model training.*
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Cannot make predictions.")

    try:
        # 1. Convert input Pydantic model to DataFrame
        input_data = patient_data.dict()
        input_df = pd.DataFrame([input_data])

        # 2. Create a model instance from the loaded model data
        model_instance = MimicBaseModel.load(os.path.join(project_root, MODEL_PATH))
        
        # 3. Use the model's built-in predict method which handles preprocessing
        # This ensures we use the exact same preprocessing logic as during training
        raw_predictions = model_instance.predict(input_df)
        
        # 4. Get prediction probabilities for readmission
        prediction_proba = model_instance.model.predict_proba(
            model_instance.scaler.transform(
                input_df[model_instance.feature_names].fillna(0)
            )
        )
        readmission_probability = prediction_proba[0, 1]  # Probability of the positive class (readmission)

        # 5. Format and return the prediction
        return {"predicted_readmission_probability": float(readmission_probability)}

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file or preprocessing asset not found: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or incorrect feature in input data: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
    except Exception as e:
        # Log the exception details for debugging
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")


if __name__ == "__main__":
    # Run the API using Uvicorn - Get host/port from config if available
    api_host = config.get('api', {}).get('host', '0.0.0.0') if config else '0.0.0.0'
    api_port = 8001  # Use port 8001 to avoid conflict with existing server
    print(f"Starting API server on {api_host}:{api_port}")
    uvicorn.run(app, host=api_host, port=api_port)