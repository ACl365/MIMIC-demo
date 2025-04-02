import os
import sys
import time  # Added for latency
from collections import deque  # Added for feature stats
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
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

# --- Monitoring Variables ---
# Store recent values for key features (e.g., age) for summary stats
# Store counts and maybe simple stats in memory for demo purposes
MAX_RECENT_VALUES = 100  # Configurable: How many recent values to store
request_stats = {
    "total_requests": 0,
    "recent_ages": deque(maxlen=MAX_RECENT_VALUES),
    # Add other features to track here, e.g.:
    # "recent_feature_x": deque(maxlen=MAX_RECENT_VALUES),
}

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
                    # Ensure columns match exactly, filling missing with 0
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
            logger.info("Skipping SHAP background data preparation for temporal model.")

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
    global request_stats  # Allow modification of global stats - Must be before assignment
    start_time = time.time()

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
        request_stats["total_requests"] += 1
        logger.info(f"Received prediction request #{request_stats['total_requests']}")

        # --- Basic Input Validation & Feature Logging ---
        if not isinstance(patient_data, dict):
            raise HTTPException(
                status_code=400, detail="Invalid input: Expected JSON object."
            )

        # Log summary stats for key features (e.g., age)
        try:
            age = patient_data.get("age")  # Assuming 'age' is a key feature
            if isinstance(age, (int, float)):
                request_stats["recent_ages"].append(float(age))

            # Log stats periodically (e.g., every 10 requests)
            if request_stats["total_requests"] % 10 == 0:
                if request_stats["recent_ages"]:
                    mean_age = np.mean(list(request_stats["recent_ages"]))
                    median_age = np.median(list(request_stats["recent_ages"]))
                    logger.info(
                        f"Recent {len(request_stats['recent_ages'])} requests - Age Stats: Mean={mean_age:.2f}, Median={median_age:.2f}"
                    )
                # Log stats for other tracked features here

        except Exception as stats_err:
            logger.warning(f"Could not log feature stats: {stats_err}")

        # --- Data Preparation ---
        input_df_raw = pd.DataFrame([patient_data])

        # Prepare features based on model type
        scaled_features = None
        if model_instance.model_type == "temporal_readmission":
            input_df = input_df_raw
            logger.debug("Using Temporal model predict (internal preprocessing).")
        else:
            logger.debug("Using standard model predict (manual preprocessing/scaling).")
            # Ensure expected_features is a list for standard models here
            if not isinstance(expected_features, list):
                raise TypeError(
                    f"Expected features for standard model should be a list, got {type(expected_features)}"
                )
            standard_features = expected_features
            try:
                # Handle empty input DataFrame correctly
                if input_df_raw.empty:
                    input_df = pd.DataFrame(
                        columns=standard_features
                    )  # Create empty DF with correct columns
                else:
                    input_df = input_df_raw.reindex(
                        columns=standard_features,
                        fill_value=0,  # Use 0 or appropriate default
                    )
            except Exception as reindex_e:
                logger.error(f"Reindexing error: {reindex_e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Data preparation error.")

            # Scale standard features
            try:
                if (
                    not hasattr(model_instance, "scaler")
                    or model_instance.scaler is None
                ):
                    raise ValueError("Standard scaler not found/fitted.")
                # Ensure input_df has the correct columns before scaling
                input_df_for_scaling = input_df[standard_features]
                scaled_features = model_instance.scaler.transform(input_df_for_scaling)
            except Exception as scale_e:
                logger.error(f"Scaling error: {scale_e}", exc_info=True)
                raise HTTPException(status_code=400, detail="Data scaling error.")

        # Make Prediction
        try:
            if model_instance.model_type == "temporal_readmission":
                prediction_result = model_instance.predict(
                    input_df
                )  # Pass raw input for temporal
                if (
                    isinstance(prediction_result, np.ndarray)
                    and prediction_result.size > 0
                ):
                    readmission_probability = prediction_result.item(0)
                else:
                    raise ValueError("Temporal prediction gave unexpected result.")
            else:  # Standard model
                if scaled_features is None:
                    raise ValueError(
                        "Scaled features are missing for standard model prediction."
                    )
                if not hasattr(model_instance.model, "predict_proba"):
                    raise AttributeError("Model missing 'predict_proba'.")
                prediction_proba = model_instance.model.predict_proba(scaled_features)
                # Log probability distribution (in this case, just the positive class prob)
                readmission_probability = prediction_proba[0, 1]
                logger.info(f"Predicted probability: {readmission_probability:.4f}")
                # If multi-class, log prediction_proba[0]

                # --- Conceptual Data Drift Detection ---
                # 1. Load baseline statistics (e.g., mean, std, distribution percentiles)
                #    for key features (like 'age') saved during training or from a reference dataset.
                #    These could be stored alongside the model artifact or in a config file.
                #    baseline_age_mean = config.get("monitoring", {}).get("baseline_age_mean", 55.0)
                #    baseline_age_std = config.get("monitoring", {}).get("baseline_age_std", 15.0)

                # 2. Compare current input distribution (using `request_stats['recent_ages']`)
                #    to the baseline using statistical tests (e.g., KS test, Z-score).
                #    if request_stats["recent_ages"]: # Check if deque is not empty
                #        current_mean_age = np.mean(list(request_stats["recent_ages"]))
                #        if abs(current_mean_age - baseline_age_mean) > 3 * baseline_age_std: # Example Z-score check
                #            logger.warning(f"Potential data drift detected in 'age'. Mean: {current_mean_age:.2f}, Baseline Mean: {baseline_age_mean:.2f}")
                #            # Trigger alert or further investigation

                # 3. For categorical features, compare frequency distributions.

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
                            return proba[:, 1]  # Return probability of positive class
                        else:
                            # Fallback if predict_proba not available (less ideal for SHAP)
                            return model_instance.model.predict(X_np)

                    # Ensure background data columns match scaled_features columns
                    # This assumes expected_features is the list of columns for standard models
                    background_df = pd.DataFrame(
                        shap_background_data, columns=expected_features
                    )

                    explainer = shap.KernelExplainer(
                        predict_fn_shap, background_df.values  # Use .values
                    )
                    # Ensure scaled_features is 2D numpy array
                    scaled_features_np = np.array(scaled_features)
                    if scaled_features_np.ndim == 1:
                        scaled_features_np = scaled_features_np.reshape(1, -1)

                    shap_values_single = explainer.shap_values(
                        scaled_features_np[0, :]  # Explain the first (only) instance
                    )
                    # KernelExplainer might return a list for binary classification, take the second element
                    if (
                        isinstance(shap_values_single, list)
                        and len(shap_values_single) == 2
                    ):
                        shap_values_single = shap_values_single[1]
                    shap_values_single = shap_values_single.flatten()

                    # Call LLM helper
                    explanation_text = explain_shap_with_llm(
                        shap_values_single=shap_values_single,
                        feature_names=expected_features,  # Standard models use list
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

        # --- Latency Logging ---
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"Prediction request completed in {latency:.4f} seconds.")

        return response_data  # Correctly indented return

    except HTTPException as http_e:
        # Log latency even for handled HTTP exceptions
        end_time = time.time()
        latency = end_time - start_time
        logger.warning(
            f"HTTPException in prediction after {latency:.4f} seconds: {http_e.detail}"
        )
        raise http_e
    except Exception as e:
        # Log latency for unexpected errors
        end_time = time.time()
        latency = end_time - start_time
        logger.error(
            f"Unexpected prediction endpoint error after {latency:.4f} seconds: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error.")


# Correctly unindented __main__ block
if __name__ == "__main__":
    # Run the API using Uvicorn
    if not config:  # Load config if startup didn't run (e.g., direct execution)
        try:
            config = load_config()
        except Exception:
            config = {}

    api_host = config.get("api", {}).get("host", "127.0.0.1")
    api_port = config.get("api", {}).get("port", 8001)
    logger.info(f"Starting API server on http://{api_host}:{api_port}")
    # Use reload=True for development convenience
    uvicorn.run("main:app", host=api_host, port=api_port, reload=True)
