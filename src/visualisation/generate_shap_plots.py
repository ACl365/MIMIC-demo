"""
Generates SHAP summary plots for a saved model pipeline.
"""

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional  # Import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Need to import ReadmissionModel to allow pickle loading if it's part of the saved object
# Adjust the import based on your actual project structure if needed
try:
    from src.models.model import (  # Needed for loading pickled model class
        ReadmissionModel,
    )
    from src.utils import get_data_path, get_logger, get_project_root, load_config
except ModuleNotFoundError:
    # Handle cases where script might be run differently or utils are elsewhere
    print(
        "Warning: Could not import src.utils or src.models.model. Ensure PYTHONPATH is set correctly or adjust imports."
    )

    # Define dummy functions if needed for basic execution without full utils
    def get_project_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def get_logger(name):
        import logging

        return logging.getLogger(name)  # Basic logger


logger = get_logger(__name__)


def load_pipeline_data(pipeline_path: str) -> Optional[Dict[str, Any]]:
    """Loads the pickled pipeline dictionary."""
    if not os.path.exists(pipeline_path):
        logger.error(f"Pipeline file not found: {pipeline_path}")
        return None
    try:
        with open(pipeline_path, "rb") as f:
            pipeline_data = pickle.load(f)
        logger.info(f"Pipeline data loaded successfully from {pipeline_path}")
        # Basic validation
        if (
            not isinstance(pipeline_data, dict)
            or "pipeline" not in pipeline_data
            or "features" not in pipeline_data
        ):
            logger.error(
                "Loaded pipeline data is missing required keys ('pipeline', 'features')."
            )
            return None
        return pipeline_data
    except Exception as e:
        logger.error(f"Error loading pipeline from {pipeline_path}: {e}")
        return None


def load_and_preprocess_data_for_shap(
    config: Dict[str, Any], features: List[str]
) -> Optional[pd.DataFrame]:
    """Loads and preprocesses data, returning only the features needed for SHAP."""
    try:
        # Load full processed data
        data_path = get_data_path("processed", "combined_features", config)
        logger.info(f"Loading data for SHAP analysis from {data_path}")
        data = pd.read_csv(data_path)

        # Preprocess using the ReadmissionModel's logic to get X
        # We only need X here, not y
        model_instance = ReadmissionModel(config=config)
        X, _ = model_instance.preprocess(data)

        # Ensure columns match the features the pipeline was trained on
        missing_cols = set(features) - set(X.columns)
        extra_cols = set(X.columns) - set(features)

        if missing_cols:
            logger.warning(
                f"Missing expected feature columns after preprocessing: {missing_cols}. Filling with 0."
            )
            for col in missing_cols:
                X[col] = 0
        if extra_cols:
            logger.warning(
                f"Extra columns found after preprocessing: {extra_cols}. Dropping them."
            )
            X = X.drop(columns=list(extra_cols))

        # Reorder columns to match the training order
        X = X[features]
        logger.info(f"Data loaded and preprocessed for SHAP. Shape: {X.shape}")
        return X

    except FileNotFoundError:
        logger.error(f"Processed data file not found at {data_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading or preprocessing data for SHAP: {e}")
        return None


def generate_shap_summary_plot(
    pipeline: Any,
    X_processed: pd.DataFrame,
    feature_names: List[str],
    output_path: str,
    sample_size: int = 100,
    plot_type: str = "dot",  # "dot" (beeswarm) or "bar"
) -> None:
    """Generates and saves a SHAP summary plot."""
    try:
        # Sample data for SHAP explanation (can be slow on full dataset)
        if len(X_processed) > sample_size:
            X_sample = shap.sample(X_processed, sample_size, random_state=42)
            logger.info(
                f"Using sample of {sample_size} instances for SHAP calculation."
            )
        else:
            X_sample = X_processed
            logger.info(
                f"Using full dataset ({len(X_processed)} instances) for SHAP calculation."
            )

        # --- Choose the right SHAP explainer ---
        # Assuming the last step of the pipeline is the model
        model = pipeline.steps[-1][1]

        # Use predict_proba for classifiers if available
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
            # For KernelExplainer, we need a function that takes a numpy array
            # and returns probabilities for the positive class.
            # Need to handle potential pipeline steps before the model.

            # If there are preprocessing steps in the pipeline before the model:
            if len(pipeline.steps) > 1:
                # Create a function that applies preprocessing then predicts
                def pipeline_predict_proba(data_np):
                    data_df = pd.DataFrame(data_np, columns=feature_names)
                    # Apply all steps *except* the final classifier
                    transformed_data = pipeline[:-1].transform(data_df)
                    # Predict using only the final classifier step
                    proba = pipeline.steps[-1][1].predict_proba(transformed_data)
                    # Return probability of the positive class (usually index 1)
                    return proba[:, 1] if proba.ndim > 1 else proba

                # KernelExplainer is model-agnostic but can be slow
                logger.info("Using shap.KernelExplainer (may be slow)...")
                explainer = shap.KernelExplainer(pipeline_predict_proba, X_sample)

            else:  # Pipeline only contains the model
                # Simpler predict function
                def model_predict_proba(data_np):
                    proba = model.predict_proba(data_np)
                    return proba[:, 1] if proba.ndim > 1 else proba

                logger.info("Using shap.KernelExplainer (may be slow)...")
                explainer = shap.KernelExplainer(model_predict_proba, X_sample)

        else:  # Fallback for models without predict_proba (e.g., regressors)
            logger.warning(
                "Model does not have predict_proba. Using predict function for SHAP."
            )
            predict_fn = model.predict
            # Adjust pipeline function if needed
            if len(pipeline.steps) > 1:

                def pipeline_predict(data_np):
                    data_df = pd.DataFrame(data_np, columns=feature_names)
                    transformed_data = pipeline[:-1].transform(data_df)
                    return pipeline.steps[-1][1].predict(transformed_data)

                logger.info("Using shap.KernelExplainer with predict (may be slow)...")
                explainer = shap.KernelExplainer(pipeline_predict, X_sample)
            else:
                logger.info("Using shap.KernelExplainer with predict (may be slow)...")
                explainer = shap.KernelExplainer(model.predict, X_sample)

        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        logger.info("SHAP values calculated.")

        # For classifiers with predict_proba, KernelExplainer might return a single array
        # or a list (for multi-class). We typically want the values for the positive class.
        # If shap_values is a list, assume index 1 corresponds to the positive class.
        if isinstance(shap_values, list) and len(shap_values) > 1:
            logger.info(
                "SHAP values appear to be multi-output. Selecting values for the positive class (index 1)."
            )
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values

        # Generate plot
        logger.info(f"Generating SHAP summary plot (type: {plot_type})...")
        plt.figure()
        shap.summary_plot(shap_values_plot, X_sample, plot_type=plot_type, show=False)

        # Enhance plot
        plt.title(f"SHAP Summary Plot ({plot_type.capitalize()} View)")
        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved to {output_path}")
        plt.close()

    except ImportError:
        logger.error(
            "SHAP library not installed. Cannot generate SHAP plots. Install with: pip install shap"
        )
    except Exception as e:
        logger.error(f"Error generating SHAP plot: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP summary plots for a saved model pipeline."
    )
    parser.add_argument(
        "--pipeline-path",
        type=str,
        required=True,
        help="Path to the saved .pkl file containing the pipeline dictionary.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Directory to save the output plot (default: assets).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="shap_summary_plot.png",
        help="Filename for the output plot (default: shap_summary_plot.png).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to use for SHAP calculation (default: 100).",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="dot",
        choices=["dot", "bar"],
        help="Type of SHAP summary plot ('dot' for beeswarm, 'bar') (default: dot).",
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline_data = load_pipeline_data(args.pipeline_path)
    if pipeline_data is None:
        return  # Exit if pipeline loading failed

    pipeline = pipeline_data["pipeline"]
    features = pipeline_data["features"]

    # Load and preprocess data
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load project config: {e}. Cannot proceed.")
        return

    X_processed = load_and_preprocess_data_for_shap(config, features)
    if X_processed is None:
        logger.error(
            "Failed to load or preprocess data for SHAP analysis. Cannot proceed."
        )
        return

    # Define output path
    os.makedirs(args.output_dir, exist_ok=True)
    output_plot_path = os.path.join(args.output_dir, args.output_filename)

    # Generate plot
    generate_shap_summary_plot(
        pipeline=pipeline,
        X_processed=X_processed,
        feature_names=features,
        output_path=output_plot_path,
        sample_size=args.sample_size,
        plot_type=args.plot_type,
    )


if __name__ == "__main__":
    main()
