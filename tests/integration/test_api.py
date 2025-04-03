import os
import sys
from unittest.mock import MagicMock, patch  # Import patch for mocking

import numpy as np  # Import numpy for isnan check
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add the api directory to the Python path to import the app
api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../api"))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Import the app *after* potentially patching dependencies
# We need to import 'main' here so the patch targets below are valid
try:
    import main
except ImportError as e:
    print(f"Could not import 'main' from api directory: {e}")
    main = None  # Set to None if import fails

# Define expected features for the mock model
MOCK_EXPECTED_FEATURES = ["age", "heart_rate", "systolic_bp", "glucose", "feature5"]


# --- Mock Model Setup ---
@pytest.fixture(scope="function")  # Use function scope for isolation
def mock_model_load(mocker):
    """Mocks the model loading process referenced in main.py."""
    if not main:  # Skip if main couldn't be imported
        pytest.skip("Skipping tests because api/main.py could not be imported.")

    mock_scaler = MagicMock()

    # Configure the mock scaler's transform method
    def mock_transform(df):
        # 1. Check if columns match (should always pass after API reindex)
        if list(df.columns) != MOCK_EXPECTED_FEATURES:
            raise ValueError(
                f"Mock scaler received unexpected columns: {list(df.columns)}"
            )

        # 2. Attempt numeric conversion and check for errors/NaNs
        #    Simulate scaler failing on non-numeric input
        try:
            # Select only the columns expected by the scaler
            df_to_scale = df[MOCK_EXPECTED_FEATURES]
            # Attempt conversion, coercing errors to NaN
            numeric_df = df_to_scale.apply(pd.to_numeric, errors="coerce")
            # If any NaNs were produced by coercion, it means invalid data was present
            if numeric_df.isnull().any().any():
                # Find the first column with NaN to include in the error message
                first_nan_col = numeric_df.columns[numeric_df.isnull().any()][0]
                first_bad_value = df[first_nan_col].iloc[
                    numeric_df[first_nan_col].isnull().idxmax()
                ]
                raise ValueError(
                    f"could not convert string to float: '{first_bad_value}' in column '{first_nan_col}'"
                )
            # If conversion is successful, return the numpy array
            return numeric_df.to_numpy()
        except ValueError as ve:
            # Re-raise the ValueError caught from the numeric conversion check
            raise ve
        except Exception as e:
            # Catch any other unexpected errors during the mock transformation
            raise RuntimeError(f"Unexpected error in mock scaler transform: {e}")

    mock_scaler.transform = mock_transform

    mock_predictor = MagicMock()

    # Configure the mock model's predict_proba method
    def mock_predict_proba(scaled_data):
        # Basic check: ensure input shape matches expected features
        if not isinstance(scaled_data, np.ndarray):
            raise TypeError(f"Mock model expected numpy array, got {type(scaled_data)}")
        if scaled_data.shape[1] != len(MOCK_EXPECTED_FEATURES):
            raise ValueError("Mock model received data with wrong number of features")
        # Simulate prediction: return a fixed probability array
        # [[prob_class_0, prob_class_1]]
        return np.array([[0.3, 0.7]])  # Example prediction as numpy array

    mock_predictor.predict_proba = mock_predict_proba

    mock_loaded_model = MagicMock()
    mock_loaded_model.model = mock_predictor
    mock_loaded_model.scaler = mock_scaler
    mock_loaded_model.feature_names = MOCK_EXPECTED_FEATURES
    mock_loaded_model.model_type = "MockLogisticRegression"  # Example type

    # Add a mock preprocess method needed for SHAP background data prep
    def mock_preprocess_for_shap(data, for_prediction=False):
        # Return a dummy DataFrame matching expected features and None for target
        dummy_X = pd.DataFrame(
            [[0] * len(MOCK_EXPECTED_FEATURES)], columns=MOCK_EXPECTED_FEATURES
        )
        return dummy_X, None

    mock_loaded_model.preprocess = mock_preprocess_for_shap

    # Patch targets within the imported 'main' module
    mocker.patch("main.MimicBaseModel.load", return_value=mock_loaded_model)

    # Provide a minimal config including the 'data' section needed for SHAP background data path
    shap_data_path = "mock/path/does/not/matter/for/mock/shap"
    mock_config_for_api = {
        "data": {"processed": {"combined_features": shap_data_path}},
        "logging": {"level": "INFO"},  # Keep logging if needed
        # Add other minimal sections if startup requires them
    }
    mocker.patch("main.load_config", return_value=mock_config_for_api)

    # Patch os.path.exists used within main.py's startup handler
    mocker.patch(
        "main.os.path.exists", return_value=True
    )  # Assume model file exists for loading

    # Patch pd.read_csv specifically for the SHAP background data loading call
    def mock_read_csv_for_shap(*args, **kwargs):
        # Check if the requested path ends with the expected mock path suffix
        # Use os.path.normpath to handle potential separator differences (e.g., / vs \)
        normalized_arg_path = os.path.normpath(args[0])
        normalized_shap_path = os.path.normpath(shap_data_path)
        if normalized_arg_path.endswith(normalized_shap_path):
            # Return a minimal DataFrame matching expected features for SHAP
            return pd.DataFrame(
                [[0] * len(MOCK_EXPECTED_FEATURES)], columns=MOCK_EXPECTED_FEATURES
            )
        # For other calls, raise an error or return default (though none expected here)
        raise ValueError(f"Unexpected pd.read_csv call in mock_model_load: {args[0]}")

    mocker.patch("main.pd.read_csv", side_effect=mock_read_csv_for_shap)

    # Return the mock object if needed, though patching is the main goal
    return mock_loaded_model


# --- Test Client Fixture ---
@pytest.fixture(
    scope="function"
)  # Use function scope to benefit from mock_model_load fixture
def client(mock_model_load):  # Depend on the mock fixture
    """Provides a TestClient instance with mocked model loading."""
    if not main:  # Skip if main couldn't be imported
        pytest.skip("Skipping tests because api/main.py could not be imported.")

    # The app instance is created in main.py, use it directly
    app_instance = main.app
    # The startup event runs when TestClient is initialized
    with TestClient(app_instance) as test_client:
        yield test_client
    # Cleanup happens automatically when exiting 'with' block


# --- Health Check Tests ---


def test_health_check_ok(client):
    """Test /health endpoint when model is loaded (mocked)."""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert json_response["message"] == "API is running and model components are loaded."
    assert json_response["model_type"] == "MockLogisticRegression"
    assert json_response["expected_features_count"] == len(MOCK_EXPECTED_FEATURES)


def test_health_check_fail_model_not_loaded(mocker):
    """Test /health endpoint when model loading fails (mocked)."""
    if not main:  # Skip if main couldn't be imported
        pytest.skip("Skipping tests because api/main.py could not be imported.")

    # Patch load to raise an exception *before* creating the client for this specific test
    mocker.patch(
        "main.MimicBaseModel.load", side_effect=FileNotFoundError("Mock load error")
    )
    mocker.patch("main.load_config", return_value={})  # Ensure config mock is active
    mocker.patch(
        "main.os.path.exists", return_value=True
    )  # Still assume file exists initially

    # Need to re-initialize client *after* re-patching to trigger startup failure
    app_instance = main.app
    # Reset globals in the app instance as startup handler might have failed partially
    # This simulates the state where model_instance is None after startup fails
    main.model_instance = None
    main.expected_features = []
    main.model_data = {}

    with TestClient(app_instance) as failed_client:
        response = failed_client.get("/health")
        assert response.status_code == 503
        assert "API is unhealthy" in response.json()["detail"]
        # Check specific failure reason based on api.main logic
        assert "Model instance not loaded" in response.json()["detail"]


# --- Root Endpoint Test ---


def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the MIMIC Readmission Predictor API"
    }


# --- Prediction Endpoint Tests (/predict) ---


def test_predict_valid_input(client):
    """Test /predict with valid input matching expected features."""
    valid_payload = {
        "age": 65,
        "heart_rate": 88.0,
        "systolic_bp": 120.0,
        "glucose": 110.0,
        "feature5": 1,
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_readmission_probability" in json_response
    # Check if the probability is the one returned by the mock model
    assert json_response["predicted_readmission_probability"] == 0.7
    # Check type
    assert isinstance(json_response["predicted_readmission_probability"], float)


def test_predict_missing_features(client):
    """Test /predict with some expected features missing (should be filled with 0)."""
    payload_missing_features = {
        "age": 70,
        "systolic_bp": 140.0,
        # 'heart_rate', 'glucose', 'feature5' are missing
    }
    response = client.post("/predict", json=payload_missing_features)
    # Should still succeed as the API fills missing with 0
    assert response.status_code == 200
    assert "predicted_readmission_probability" in response.json()
    assert (
        response.json()["predicted_readmission_probability"] == 0.7
    )  # Mock returns fixed value


def test_predict_extra_features(client):
    """Test /predict with extra features not expected by the model (should be ignored)."""
    payload_extra_features = {
        "age": 55,
        "heart_rate": 75.0,
        "systolic_bp": 115.0,
        "glucose": 95.0,
        "feature5": 0,
        "extra_info": "some_value",  # This should be ignored
        "another_field": 123,
    }
    response = client.post("/predict", json=payload_extra_features)
    # Should succeed as the API ignores extra features
    assert response.status_code == 200
    assert "predicted_readmission_probability" in response.json()
    assert response.json()["predicted_readmission_probability"] == 0.7


def test_predict_invalid_data_type_causes_scaling_error(client):
    """Test /predict with data type that causes scaling error after reindexing."""
    # The API fills missing with 0, but if a provided feature intended for scaling
    # is not numeric, the scaler.transform call should fail.
    invalid_payload = {
        "age": 60,
        "heart_rate": "high",  # Invalid type for a numeric feature
        "systolic_bp": 130.0,
        # Other features missing, will be filled with 0
    }
    response = client.post("/predict", json=invalid_payload)
    # This should cause an error during the scaler.transform step inside the endpoint
    assert (
        response.status_code == 400
    )  # As defined in the API's error handling for scaling ValueError
    assert "Data scaling error" in response.json()["detail"]
    # Check for specific error detail propagated from the exception
    # Check for the general error message returned by the API's exception handler
    assert "data scaling error" in response.json()["detail"].lower()


def test_predict_invalid_input_format_not_dict(client):
    """Test /predict with input that is not a JSON object."""
    invalid_payload = [{"key": "value"}]  # Send a list instead of a dict
    response = client.post("/predict", json=invalid_payload)
    # FastAPI/Pydantic intercepts this before our manual check if input type is Dict
    # It returns 422 Unprocessable Entity for invalid input structure against the endpoint signature
    assert response.status_code == 422
    # Optionally check the detail message structure from FastAPI/Pydantic
    # assert "value is not a valid dict" in str(response.json()).lower() # Example detail


def test_predict_empty_payload(client):
    """Test /predict with an empty JSON object."""
    empty_payload = {}
    response = client.post("/predict", json=empty_payload)
    # Should succeed, filling all expected features with 0
    assert response.status_code == 200
    assert "predicted_readmission_probability" in response.json()
    assert response.json()["predicted_readmission_probability"] == 0.7


def test_predict_model_not_loaded(mocker):
    """Test /predict when the model failed to load."""
    if not main:  # Skip if main couldn't be imported
        pytest.skip("Skipping tests because api/main.py could not be imported.")

    # Patch load to raise an exception *before* creating the client
    mocker.patch(
        "main.MimicBaseModel.load", side_effect=FileNotFoundError("Mock load error")
    )
    mocker.patch("main.load_config", return_value={})
    mocker.patch("main.os.path.exists", return_value=True)

    # Re-initialize client to trigger startup with failed load
    app_instance = main.app
    # Reset globals in the app instance as startup handler might have failed partially
    main.model_instance = None
    main.expected_features = []
    main.model_data = {}

    with TestClient(app_instance) as failed_client:
        # The check happens at the start of the /predict endpoint
        valid_payload = {
            "age": 65,
            "heart_rate": 88.0,
            "systolic_bp": 120.0,
            "glucose": 110.0,
            "feature5": 1,
        }
        response = failed_client.post("/predict", json=valid_payload)

        assert response.status_code == 503
        assert "Model not loaded properly" in response.json()["detail"]
