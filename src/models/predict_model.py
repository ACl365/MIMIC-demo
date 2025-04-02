"""
Script to make predictions with trained models.
"""

import argparse
import os
from typing import Dict, Optional  # Removed List, Union

import numpy as np
import pandas as pd

from src.models.model import LengthOfStayModel, MortalityModel, ReadmissionModel
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


def load_model(model_type: str, model_path: Optional[str] = None):
    """
    Load a trained model.

    Args:
        model_type (str): Type of model to load ('readmission', 'mortality', or 'los')
        model_path (Optional[str], optional): Path to the model file.
            If None, uses the default path. Defaults to None.

    Returns:
        Union[ReadmissionModel, MortalityModel, LengthOfStayModel]: Loaded model
    """
    if model_path is None:
        model_path = os.path.join(
            get_project_root(), "models", f"{model_type}_model.pkl"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model based on type
    if model_type == "readmission":
        model = ReadmissionModel.load(model_path)
    elif model_type == "mortality":
        model = MortalityModel.load(model_path)
    elif model_type == "los":
        model = LengthOfStayModel.load(model_path)
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
    Make predictions with a trained model.

    Args:
        data (pd.DataFrame): Input data
        model_type (str): Type of model to use ('readmission', 'mortality', or 'los')
        model_path (Optional[str], optional): Path to the model file.
            If None, uses the default path. Defaults to None.
        output_path (Optional[str], optional): Path to save the predictions to.
            If None, doesn't save the predictions. Defaults to None.

    Returns:
        pd.DataFrame: Predictions
    """
    logger.info(f"Making predictions with {model_type} model")

    # Load model
    model = load_model(model_type, model_path)

    # Make predictions
    predictions = model.predict(data)

    # Create output dataframe
    output = pd.DataFrame(
        {
            "subject_id": (
                data["subject_id"] if "subject_id" in data.columns else np.nan
            ),
            "hadm_id": data["hadm_id"] if "hadm_id" in data.columns else np.nan,
            "stay_id": data["stay_id"] if "stay_id" in data.columns else np.nan,
            f"{model_type}_prediction": predictions,
        }
    )

    # Save predictions if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save predictions
        output.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")

    return output


def predict_all(
    data: pd.DataFrame, config: Optional[Dict] = None, output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Make predictions with all trained models.

    Args:
        data (pd.DataFrame): Input data
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
        output_dir (Optional[str], optional): Directory to save the predictions to.
            If None, uses the default directory. Defaults to None.

    Returns:
        Dict[str, pd.DataFrame]: Predictions for each model
    """
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "predictions")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make predictions with each model
    predictions = {}

    # Readmission model
    try:
        readmission_output_path = os.path.join(
            output_dir, "readmission_predictions.csv"
        )
        predictions["readmission"] = predict(
            data, "readmission", output_path=readmission_output_path
        )
    except FileNotFoundError:
        logger.warning("Readmission model not found, skipping predictions")

    # Mortality model
    try:
        mortality_output_path = os.path.join(output_dir, "mortality_predictions.csv")
        predictions["mortality"] = predict(
            data, "mortality", output_path=mortality_output_path
        )
    except FileNotFoundError:
        logger.warning("Mortality model not found, skipping predictions")

    # Length of stay model
    try:
        los_output_path = os.path.join(output_dir, "los_predictions.csv")
        predictions["los"] = predict(data, "los", output_path=los_output_path)
    except FileNotFoundError:
        logger.warning("Length of stay model not found, skipping predictions")

    # Combine predictions
    if predictions:
        combined = predictions[list(predictions.keys())[0]].copy()

        for model_type, preds in predictions.items():
            if model_type != list(predictions.keys())[0]:
                combined[f"{model_type}_prediction"] = preds[f"{model_type}_prediction"]

        combined_output_path = os.path.join(output_dir, "combined_predictions.csv")
        combined.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined predictions to {combined_output_path}")

    return predictions


def main() -> None:
    """
    Main function to make predictions.
    """
    parser = argparse.ArgumentParser(description="Make predictions with trained models")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["readmission", "mortality", "los", "all"],
        default="all",
        help="Type of model to use",
    )
    parser.add_argument(
        "--input", type=str, default=None, help="Path to input data file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to output directory"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Load data
    logger.info("Loading data")
    if args.input is not None:
        data = pd.read_csv(args.input)
    else:
        data_path = get_data_path("processed", "combined_features", config)
        data = pd.read_csv(data_path)

    # Make predictions
    if args.model == "all":
        predictions = predict_all(data, config, args.output)
    else:
        output_path = None
        if args.output is not None:
            output_path = os.path.join(args.output, f"{args.model}_predictions.csv")

        predictions = {args.model: predict(data, args.model, output_path=output_path)}

    # Print prediction summary
    for model_type, preds in predictions.items():
        logger.info(f"{model_type.capitalize()} model predictions:")
        logger.info(f"  Number of predictions: {len(preds)}")

        if model_type in ["readmission", "mortality"]:
            positive_count = preds[f"{model_type}_prediction"].sum()
            logger.info(
                f"  Positive predictions: {positive_count} ({positive_count / len(preds) * 100:.2f}%)"
            )
        elif model_type == "los":
            mean_los = preds[f"{model_type}_prediction"].mean()
            median_los = preds[f"{model_type}_prediction"].median()
            logger.info(f"  Mean predicted length of stay: {mean_los:.2f} days")
            logger.info(f"  Median predicted length of stay: {median_los:.2f} days")


if __name__ == "__main__":
    main()
