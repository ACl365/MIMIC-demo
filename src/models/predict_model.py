"""
Script to make predictions using trained models.

This script provides functions to load previously trained models (readmission,
mortality, length of stay) and use them to generate predictions on new data.
It includes functionality to predict using a single specified model or all
available models, saving the predictions to CSV files.
"""

import argparse
import os
from typing import Any, Dict, Optional, Union  # Added Union, Any

import numpy as np
import pandas as pd

# Import specific model classes and the base class for type hinting
from src.models.model import (
    BaseModel,
    LengthOfStayModel,
    MortalityModel,
    ReadmissionModel,
)
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


def load_model(model_type: str, model_path: Optional[str] = None) -> BaseModel:
    """
    Load a trained model based on its type.

    Uses the appropriate model class's `load` method. If `model_path` is not
    provided, it constructs a default path based on the `model_type` within
    the project's 'models/' directory.

    Args:
        model_type (str): Type of model to load ('readmission', 'mortality', or 'los').
                          This determines which model class's load method is called.
        model_path (Optional[str], optional): Explicit path to the model file (.pkl or .pt).
                                             If None, constructs a default path. Defaults to None.

    Returns:
        BaseModel: The loaded model object (an instance of ReadmissionModel,
                   MortalityModel, or LengthOfStayModel).

    Raises:
        FileNotFoundError: If the model file (either specified or default) does not exist.
        ValueError: If an unknown `model_type` is provided.
        Exception: Re-raises exceptions from the underlying `model.load()` method.
    """
    if model_path is None:
        # Construct default path (assuming .pkl for standard models)
        # Note: Temporal models might use .pt and require different loading logic handled by their load method
        default_filename = f"{model_type}_model.pkl"
        model_path = os.path.join(get_project_root(), "models", default_filename)
        logger.info(f"Model path not specified, using default: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading {model_type} model from {model_path}...")
    # Use the classmethod load from the appropriate subclass
    if model_type == "readmission":
        model = ReadmissionModel.load(model_path)
    elif model_type == "mortality":
        model = MortalityModel.load(model_path)
    elif model_type == "los":
        model = LengthOfStayModel.load(model_path)
    # Add elif for TemporalReadmissionModel if needed, calling its specific load
    # elif model_type == "temporal_readmission":
    #     model = TemporalReadmissionModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def predict(
    data: pd.DataFrame,
    model_type: str,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a model and generate predictions on the input data.

    Loads the specified model type, calls its `predict` method, and formats
    the output into a DataFrame containing identifiers and predictions.
    Optionally saves the predictions to a CSV file.

    Args:
        data (pd.DataFrame): Input DataFrame containing features required by the model's
                             preprocessing step. Must include 'subject_id', 'hadm_id'.
        model_type (str): Type of model to use ('readmission', 'mortality', or 'los').
        model_path (Optional[str], optional): Path to the trained model file. Defaults to None (uses default path).
        output_path (Optional[str], optional): Path to save the predictions CSV file.
                                             If None, predictions are not saved to disk. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing 'subject_id', 'hadm_id', 'stay_id' (if present),
                      and a column named '{model_type}_prediction' with the model's output.

    Raises:
        FileNotFoundError: If the model file cannot be found.
        ValueError: If the input data is missing required ID columns or if prediction fails.
        Exception: Re-raises exceptions from model loading or prediction.
    """
    logger.info(f"Making predictions with {model_type} model")

    # Ensure necessary ID columns are present
    required_ids = ["subject_id", "hadm_id"]
    if not all(col in data.columns for col in required_ids):
        missing_ids = set(required_ids) - set(data.columns)
        raise ValueError(f"Input data is missing required ID columns: {missing_ids}")

    # Load model
    model = load_model(model_type, model_path)

    # Make predictions
    try:
        predictions = model.predict(
            data.copy()
        )  # Pass a copy to avoid modifying original data in predict
    except Exception as e:
        logger.error(
            f"Error during prediction with {model_type} model: {e}", exc_info=True
        )
        raise ValueError(f"Prediction failed for model type {model_type}") from e

    # Create output dataframe
    output_data: Dict[str, Any] = {
        "subject_id": data["subject_id"],
        "hadm_id": data["hadm_id"],
    }
    # Include stay_id if it exists in the input data
    if "stay_id" in data.columns:
        output_data["stay_id"] = data["stay_id"]

    output_data[f"{model_type}_prediction"] = predictions
    output = pd.DataFrame(output_data)

    # Save predictions if output path is provided
    if output_path is not None:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save predictions
            output.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save predictions to {output_path}: {e}", exc_info=True
            )

    return output


def predict_all(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all available models (readmission, mortality, los) and generate predictions.

    Iterates through the model types, loads each model, makes predictions,
    saves individual prediction files, and finally saves a combined prediction file.

    Args:
        data (pd.DataFrame): Input DataFrame containing features.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        output_dir (Optional[str], optional): Directory to save prediction files.
                                             Defaults to 'predictions/' in the project root.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are model types and values
                                 are the corresponding prediction DataFrames.
    """
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "predictions")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving predictions to directory: {output_dir}")

    # Make predictions with each model
    predictions: Dict[str, pd.DataFrame] = {}
    model_types_to_predict = [
        "readmission",
        "mortality",
        "los",
    ]  # Add other types if needed

    for model_type in model_types_to_predict:
        try:
            output_path = os.path.join(output_dir, f"{model_type}_predictions.csv")
            # Assuming default model paths are used if specific paths not in config
            model_path = os.path.join(
                get_project_root(), "models", f"{model_type}_model.pkl"
            )
            if os.path.exists(model_path):
                predictions[model_type] = predict(
                    data, model_type, model_path=model_path, output_path=output_path
                )
            else:
                logger.warning(
                    f"{model_type.capitalize()} model not found at default path {model_path}, skipping predictions."
                )

        except FileNotFoundError:
            logger.warning(
                f"{model_type.capitalize()} model not found, skipping predictions."
            )
        except Exception as e:
            logger.error(
                f"Error predicting with {model_type} model: {e}", exc_info=True
            )

    # Combine predictions into a single file
    if predictions:
        # Use the first available prediction df as base and ensure IDs are present
        first_key = list(predictions.keys())[0]
        base_ids = ["subject_id", "hadm_id"]
        if "stay_id" in predictions[first_key].columns:
            base_ids.append("stay_id")
        combined = predictions[first_key][base_ids + [f"{first_key}_prediction"]].copy()

        for model_type, preds_df in predictions.items():
            if model_type != first_key:
                pred_col_name = f"{model_type}_prediction"
                if pred_col_name in preds_df.columns:
                    # Merge predictions based on available IDs
                    merge_on = ["subject_id", "hadm_id"]
                    if "stay_id" in base_ids and "stay_id" in preds_df.columns:
                        merge_on.append("stay_id")
                    combined = pd.merge(
                        combined,
                        preds_df[merge_on + [pred_col_name]],
                        on=merge_on,
                        how="left",  # Keep all rows from the first prediction set
                    )

        try:
            combined_output_path = os.path.join(output_dir, "combined_predictions.csv")
            combined.to_csv(combined_output_path, index=False)
            logger.info(f"Saved combined predictions to {combined_output_path}")
        except Exception as e:
            logger.error(f"Failed to save combined predictions: {e}", exc_info=True)

    else:
        logger.warning("No predictions were generated to combine.")

    return predictions


def main() -> None:
    """
    Main execution function for the prediction script.

    Parses command-line arguments for config path, model type ('all' or specific),
    input data path, and output directory. Loads data and runs predictions.
    Prints a summary of the generated predictions.
    """
    parser = argparse.ArgumentParser(description="Make predictions with trained models")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "readmission",
            "mortality",
            "los",
            "all",
        ],  # Add other valid model types here
        default="all",
        help="Type of model to use for prediction (default: all)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input data CSV file (default: combined_features.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save prediction CSV files (default: ./predictions)",
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Load data
    logger.info("Loading data for prediction...")
    if args.input is not None:
        input_path = args.input
    else:
        input_path = get_data_path("processed", "combined_features", config)

    try:
        data = pd.read_csv(input_path)
        logger.info(f"Data loaded successfully from {input_path}. Shape: {data.shape}")
    except FileNotFoundError:
        logger.error(
            f"Input data file not found at {input_path}. Cannot make predictions."
        )
        return
    except Exception as e:
        logger.error(f"Error loading input data from {input_path}: {e}", exc_info=True)
        return

    # Make predictions
    if args.model == "all":
        predictions = predict_all(data, config, args.output)
    else:
        output_path = None
        if args.output is not None:
            output_path = os.path.join(args.output, f"{args.model}_predictions.csv")

        try:
            predictions = {
                args.model: predict(data, args.model, output_path=output_path)
            }
        except (FileNotFoundError, ValueError) as e:
            logger.error(
                f"Could not generate predictions for model '{args.model}': {e}"
            )
            predictions = {}  # Ensure predictions is defined even on error
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during prediction for model '{args.model}': {e}",
                exc_info=True,
            )
            predictions = {}

    # Print prediction summary
    if predictions:
        logger.info("--- Prediction Summary ---")
        for model_type, preds in predictions.items():
            logger.info(f"{model_type.capitalize()} model predictions:")
            if preds is not None and not preds.empty:
                logger.info(f"  Number of predictions: {len(preds)}")
                pred_col = f"{model_type}_prediction"
                if pred_col in preds.columns:
                    if model_type in ["readmission", "mortality"]:
                        positive_count = preds[pred_col].sum()
                        logger.info(
                            f"  Positive predictions: {positive_count} ({positive_count / len(preds) * 100:.2f}%)"
                        )
                    elif model_type == "los":
                        mean_los = preds[pred_col].mean()
                        median_los = preds[pred_col].median()
                        logger.info(
                            f"  Mean predicted length of stay: {mean_los:.2f} days"
                        )
                        logger.info(
                            f"  Median predicted length of stay: {median_los:.2f} days"
                        )
                else:
                    logger.warning(
                        f"  Prediction column '{pred_col}' not found in results."
                    )
            else:
                logger.warning("  Prediction DataFrame is empty or None.")
        logger.info("--- End Summary ---")
    else:
        logger.warning("No predictions were generated.")


if __name__ == "__main__":
    main()
