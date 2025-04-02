import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np  # Added for SHAP
import pandas as pd
import shap  # Added for SHAP
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src directory to Python path BEFORE importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Project Module Imports ---
# Moved imports here to be after sys.path modification
from src.models.model import BaseModel as MimicBaseModel
from src.utils import get_data_path
from src.utils.config import load_config
from src.utils.llm_utils import explain_shap_with_llm
from src.utils.logger import get_logger

# --- Global Variables ---
logger = get_logger(__name__)  # Initialize logger for API
model_data: Dict[str, Any] = {}
model_instance: Optional[MimicBaseModel] = None
expected_features: Optional[Union[List[str], Dict[str, List[str]]]] = None
shap_background_data: Optional[pd.DataFrame] = None
config = {}  # Initialize as empty dict

# --- FastAPI App ---
app = FastAPI(
    title="MIMIC Readmission Predictor API",
    description=(
        "API to predict 30-day hospital readmission risk using the MIMIC demo dataset. "
        "Optionally provides LLM-generated explanations based on SHAP values."
    ),
    version="0.1.1",
)


# --- Startup Event Handler ---
@app.on_event("startup")
async def load_model_on_startup() -> None:
    """Load model, scaler, features, and SHAP background data."""
    global model_data, model_instance, expected_features, config, shap_background_data
    logger.info("API startup: Loading resources...")
    try:
        # Load config
        try:
            config = load_config()
            logger.info("Configuration loaded successfully.")
        except Exception as config_e:
            logger.warning(
                f"Config file not found/invalid: {config_e}. Using defaults."
            )
            config = {}

        # Get model path from config
        model_rel_path = config.get("api", {}).get(
            "model_path", "models/readmission_model.pkl"
        )
        model_abs_path = os.path.join(project_root, model_rel_path)
        logger.info(f"Attempting to load model from: {model_abs_path}")

        if not os.path.exists(model_abs_path):
            logger.critical(f"Model file not found at: {model_abs_path}")
            raise FileNotFoundError(f"Model file not found: {model_abs_path}")

        # Load model instance using the class method
        model_instance = MimicBaseModel.load(model_abs_path)

        # --- Validate loaded model ---
        if not hasattr(model_instance, "model") or model_instance.model is None:
            raise ValueError("Loaded instance missing 'model' attribute or it's None.")

        # Validate attributes based on model type
        if model_instance.model_type == "temporal_readmission":
            if not hasattr(model_instance, "feature_names") or not isinstance(
                model_instance.feature_names, dict
            ):
                raise TypeError("Temporal model 'feature_names' missing or not dict.")
            if (
                not hasattr(model_instance, "static_scaler")
                or model_instance.static_scaler is None
            ):
                raise ValueError("Temporal model missing 'static_scaler' or it's None.")
        else:  # Standard models
            if not hasattr(model_instance, "feature_names") or not isinstance(
                model_instance.feature_names, list
            ):
                raise TypeError("Standard model 'feature_names' missing or not list.")
            if not hasattr(model_instance, "scaler") or model_instance.scaler is None:
                raise ValueError("Standard model missing 'scaler' or it's None.")

        # Store expected features
        expected_features = model_instance.feature_names

        logger.info(f"Model loaded successfully from {model_abs_path}")
        logger.info(f"Model Type: {model_instance.model_type}")

        # Log feature details
        if isinstance(expected_features, dict):
            f_static = expected_features.get("static", [])
            f_seq = expected_features.get("sequence", [])
            f_names_to_print = f_static[:5]
            f_count = len(f_static) + len(f_seq)
            f_type = "Static/Sequence"
        elif isinstance(expected_features, list):
            f_names_to_print = expected_features[:5]
            f_count = len(expected_features)
            f_type = "Flat"
        else:
            f_names_to_print = []
            f_count = 0
            f_type = "Unknown"
        logger.info(
            f"Expected Features ({f_count}, Type: {f_type}): {f_names_to_print}..."
        )

        # --- Prepare SHAP background data (for standard models) ---
        if model_instance.model_type != "temporal_readmission" and isinstance(
            expected_features, list
        ):
            logger.info("Preparing SHAP background data...")
            try:
                data_path = get_data_path("processed", "combined_features", config)
                if not os.path.exists(data_path):
                    logger.warning(
                        f"Combined features file not found at {data_path}. "
                        "Cannot create SHAP background data."
                    )
                else:
                    full_data = pd.read_csv(data_path)
                    X_processed, _ = model_instance.preprocess(
                        full_data, for_prediction=True
                    )
                    X_processed = X_processed.reindex(
                        columns=expected_features, fill_value=0
                    )
                    X_scaled = pd.DataFrame(
                        model_instance.scaler.transform(X_processed),
                        columns=expected_features,
                    )
                    sample_size = 100
                    shap_background_data = shap.sample(
                        X_scaled, min(sample_size, len(X_scaled)), random_state=42
                    )
                    logger.info(
                        f"SHAP background data prepared with {len(shap_background_data)} samples."
                    )
            except Exception as shap_data_e:
                logger.error(
                    f"Failed to prepare SHAP background data: {shap_data_e}",
                    exc_info=True,
                )
                shap_background_data = None
        else:
            logger.info(
                "Skipping SHAP background data preparation for temporal model."
            )

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during API startup: {e}", exc_info=True)
        # Reset globals to ensure health check fails clearly
        model_data = {}
        model_instance = None
        expected_features = None
        shap_background_data = None


@app.get("/")
async def root() -> dict:
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the MIMIC Readmission Predictor API"}


@app.get("/health", summary="Check API Health")
async def health_check() -> dict:
    """Check if the API is running and the model components are loaded."""
    scaler_ok = (
        hasattr(model_instance, "scaler") and model_instance.scaler is not None
    ) or (
        hasattr(model_instance, "static_scaler")
        and model_instance.static_scaler is not None
    )

    if (
        model_instance
        and hasattr(model_instance, "model")
        and model_instance.model is not None
        and scaler_ok
        and expected_features is not None
    ):
        shap_status = "Not Applicable (Temporal Model)"
        if model_instance.model_type != "temporal_readmission":
            shap_status = "Loaded" if shap_background_data is not None else "Not Loaded"

        feat_count = (
            len(expected_features.get("static", []))
            + len(expected_features.get("sequence", []))
            if isinstance(expected_features, dict)
            else len(expected_features)
        )

        return {
            "status": "ok",
            "message": "API is running and model components are loaded.",
            "model_type": getattr(model_instance, "model_type", "Unknown"),
            "expected_features_count": feat_count,
            "shap_background_data_status": shap_status,
        }
    else:
        error_details = []
        if not model_instance:
            error_details.append("Model instance not loaded.")
        elif not hasattr(model_instance, "model") or model_instance.model is None:
            error_details.append("Model attribute missing or None.")
        elif not scaler_ok:
            error_details.append(
                "Scaler attribute (scaler/static_scaler) missing or None."
            )
        if not expected_features:
            error_details.append("Expected features not loaded (None or empty).")
        raise HTTPException(
            status_code=503,
            detail=f"API is unhealthy. Failed to load model components: {'; '.join(error_details)}",
        )


# --- Prediction Endpoint ---
class PatientFeatures(BaseModel):
    # Placeholder for potential future input validation using Pydantic
    pass


@app.post("/predict", summary="Predict 30-Day Readmission Risk")
async def predict_readmission(
    patient_data: Dict[str, Any],
    explain: bool = False,  # Optional query parameter for explanation
) -> dict:
    """
    Predicts the probability of 30-day hospital readmission.

    - **Input**: JSON object with patient features.
    - **Query Parameter**: `explain=true` for LLM explanation (adds latency).
    - **Output**: JSON with probability and optional explanation.
    """
    # Re-check model load status
    scaler_ok = (
        hasattr(model_instance, "scaler") and model_instance.scaler is not None
    ) or (
        hasattr(model_instance, "static_scaler")
        and model_instance.static_scaler is not None
    )
    if (
        not model_instance
        or not hasattr(model_instance, "model")
        or model_instance.model is None
        or not scaler_ok
        or expected_features is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded properly. Check API health.",
        )

    explanation_text: Optional[str] = None

    try:
        # Validate Input
        if not isinstance(patient_data, dict):
            raise HTTPException(
                status_code=400, detail="Invalid input: Expected JSON object."
            )

        input_df_raw = pd.DataFrame([patient_data])

        # Prepare features based on model type
        scaled_features = None
        if model_instance.model_type == "temporal_readmission":
            input_df = input_df_raw
            logger.debug("Using Temporal model predict (internal preprocessing).")
        else:
            logger.debug("Using standard model predict (manual preprocessing/scaling).")
            standard_features = expected_features
            try:
                if input_df_raw.empty and not standard_features:
                    input_df = pd.DataFrame(columns=standard_features)
                elif input_df_raw.empty:
                    input_df = pd.DataFrame(
                        [[0] * len(standard_features)], columns=standard_features
                    )
                else:
                    input_df = input_df_raw.reindex(
                        columns=standard_features, fill_value=0
                    )
            except Exception as reindex_e:
                logger.error(f"Reindexing error: {reindex_e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Data preparation error.")

            # Scale standard features
            try:
                if not hasattr(model_instance, "scaler") or model_instance.scaler is None:
                    raise ValueError("Standard scaler not found/fitted.")
                scaled_features = model_instance.scaler.transform(input_df)
            except Exception as scale_e:
                logger.error(f"Scaling error: {scale_e}", exc_info=True)
                raise HTTPException(status_code=400, detail="Data scaling error.")

        # Make Prediction
        try:
            if model_instance.model_type == "temporal_readmission":
                prediction_result = model_instance.predict(input_df)
                if isinstance(prediction_result, np.ndarray) and prediction_result.size > 0:
                    readmission_probability = prediction_result.item(0)
                else:
                    raise ValueError("Temporal prediction gave unexpected result.")
            else:
                if not hasattr(model_instance.model, "predict_proba"):
                    raise AttributeError("Model missing 'predict_proba'.")
                prediction_proba = model_instance.model.predict_proba(scaled_features)
                readmission_probability = prediction_proba[0, 1]
        except Exception as predict_e:
            logger.error(f"Prediction error: {predict_e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Prediction execution error.")

        # Generate Explanation (Optional)
        if explain:
            logger.info("Explanation requested.")
            if model_instance.model_type == "temporal_readmission":
                explanation_text = "Explanation not available for temporal models."
                logger.warning(explanation_text)
            elif shap_background_data is None:
                explanation_text = "Explanation unavailable (missing background data)."
                logger.warning(explanation_text)
            elif scaled_features is None:
                explanation_text = "Explanation unavailable (scaling error)."
                logger.error(explanation_text)
            else:
                try:
                    # Define predict function for SHAP
                    def predict_fn_shap(X_np):
                        if hasattr(model_instance.model, "predict_proba"):
                            proba = model_instance.model.predict_proba(X_np)
                            return proba[:, 1]
                        else:
                            return model_instance.model.predict(X_np)

                    explainer = shap.KernelExplainer(
                        predict_fn_shap, shap_background_data.values
                    )
                    shap_values_single = explainer.shap_values(
                        scaled_features[0, :].reshape(1, -1)
                    )
                    if isinstance(shap_values_single, list): # Handle multi-output
                        shap_values_single = shap_values_single[1]
                    shap_values_single = shap_values_single.flatten()

                    # Call LLM helper
                    explanation_text = explain_shap_with_llm(
                        shap_values_single=shap_values_single,
                        feature_names=expected_features, # Standard models use list
                        prediction_prob=readmission_probability,
                        top_n=3,
                    )
                    if explanation_text is None:
                        explanation_text = "Failed to generate LLM explanation."

                except Exception as shap_err:
                    logger.error(f"SHAP explanation error: {shap_err}", exc_info=True)
                    explanation_text = "Error generating explanation."

        # Format and return response
        response_data = {
            "predicted_readmission_probability": float(readmission_probability)
        }
        if explain:
            response_data["explanation"] = explanation_text

        return response_data

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logger.error(f"Unexpected prediction endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


if __name__ == "__main__":
    # Run the API using Uvicorn
    if not config: # Load config if startup didn't run (e.g., direct execution)
        try:
            config = load_config()
        except Exception:
            config = {}

    api_host = config.get("api", {}).get("host", "127.0.0.1")
    api_port = config.get("api", {}).get("port", 8001)
    logger.info(f"Starting API server on http://{api_host}:{api_port}")
    # Use reload=True for development convenience
    uvicorn.run("main:app", host=api_host, port=api_port, reload=True)
