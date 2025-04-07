"""
Generates fairness analysis plots for a saved model pipeline.
Compares model performance across demographic groups.
"""

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

# Need to import ReadmissionModel to allow pickle loading
try:
    from src.models.model import (  # Needed for loading pickled model class
        ReadmissionModel,
    )
    from src.utils import get_data_path, get_logger, get_project_root, load_config
except ModuleNotFoundError:
    print(
        "Warning: Could not import src.utils or src.models.model. Ensure PYTHONPATH is set correctly or adjust imports."
    )

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


def load_and_preprocess_data_for_fairness(
    config: Dict[str, Any],
    pipeline_features: List[str],
    sensitive_attribute_prefix: str,
) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """
    Loads and preprocesses data, returning features (including sensitive attributes),
    target, and the names of the sensitive attribute columns found in the preprocessed data.
    Ensures the final DataFrame contains both pipeline features and sensitive attribute columns.
    """
    try:
        # Load full processed data
        data_path = get_data_path("processed", "combined_features", config)
        logger.info(f"Loading data for fairness analysis from {data_path}")
        data = pd.read_csv(data_path)

        # Preprocess using the ReadmissionModel's logic to get X and y
        model_instance = ReadmissionModel(config=config)
        X_preprocessed, y = model_instance.preprocess(data)  # y is the target variable

        # --- Identify Sensitive Attribute Columns FROM PREPROCESSED DATA ---
        sensitive_cols = [
            col
            for col in X_preprocessed.columns
            if col.startswith(f"{sensitive_attribute_prefix}_")
        ]
        if not sensitive_cols:
            logger.error(
                f"Could not find one-hot encoded columns starting with '{sensitive_attribute_prefix}_' in the *preprocessed* feature columns: {list(X_preprocessed.columns)}"
            )
            return None
        logger.info(
            f"Found sensitive attribute columns in preprocessed data: {sensitive_cols}"
        )

        # --- Align with Pipeline Features BUT Keep Sensitive Columns ---
        final_feature_columns = list(
            pipeline_features
        )  # Start with features needed by pipeline
        sensitive_cols_to_keep = [
            col for col in sensitive_cols if col not in final_feature_columns
        ]
        final_feature_columns.extend(
            sensitive_cols_to_keep
        )  # Add sensitive cols if not already present

        missing_cols = set(final_feature_columns) - set(X_preprocessed.columns)
        extra_cols = set(X_preprocessed.columns) - set(final_feature_columns)

        X_final = X_preprocessed.copy()  # Work on a copy

        if missing_cols:
            logger.warning(
                f"Missing expected columns after preprocessing: {missing_cols}. Filling with 0."
            )
            for col in missing_cols:
                X_final[col] = 0  # Add missing columns (pipeline features or sensitive)

        if extra_cols:
            logger.warning(
                f"Extra columns found after preprocessing vs expected final columns: {extra_cols}. Dropping them."
            )
            X_final = X_final.drop(columns=list(extra_cols))

        # Reorder columns: pipeline features first, then sensitive columns
        ordered_columns = pipeline_features + sensitive_cols_to_keep
        X_final = X_final[ordered_columns]

        logger.info(
            f"Data loaded and preprocessed for fairness. Final X shape: {X_final.shape}, y shape: {y.shape}"
        )

        # X_final now contains both the features for the model AND the sensitive attribute columns
        return (
            X_final,
            y,
            sensitive_cols,
        )  # Return the names of the sensitive columns found

    except FileNotFoundError:
        logger.error(f"Processed data file not found at {data_path}")
        return None
    except Exception as e:
        logger.error(
            f"Error loading or preprocessing data for fairness: {e}", exc_info=True
        )
        return None


def calculate_fairness_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    X_processed: pd.DataFrame,  # Contains sensitive columns
    sensitive_cols: List[str],
    metric_func=recall_score,  # Function to calculate (e.g., recall_score, precision_score)
) -> Dict[str, float]:
    """Calculates the specified metric for each group defined by sensitive attributes."""
    metrics = {}

    # Ensure y_true and y_pred have the same index as X_processed for alignment
    if not y_true.index.equals(X_processed.index):
        logger.error(
            "Index mismatch between y_true and X_processed. Cannot calculate fairness metrics accurately."
        )
        return {}
    if len(y_pred) != len(X_processed):
        logger.error(
            f"Length mismatch between y_pred ({len(y_pred)}) and X_processed ({len(X_processed)}). Cannot calculate fairness metrics accurately."
        )
        return {}
    # Align y_pred with the index if it's a numpy array
    y_pred_series = pd.Series(y_pred, index=X_processed.index)

    for col in sensitive_cols:
        if col not in X_processed.columns:
            logger.error(
                f"Sensitive column '{col}' not found in final X_processed dataframe. Skipping."
            )
            continue

        # Identify the group members (where the one-hot encoded column is 1)
        group_mask = X_processed[col] == 1  # Calculate mask from X_processed
        if group_mask.sum() == 0:
            logger.warning(
                f"No samples found for group '{col}'. Skipping metric calculation."
            )
            continue

        y_true_group = y_true[group_mask]
        y_pred_group = y_pred_series[group_mask]  # Use aligned y_pred

        # Calculate metric, handle potential division by zero
        try:
            # Ensure there are positive samples in the true group for recall/precision
            if (
                metric_func in [recall_score, precision_score, f1_score]
                and y_true_group.sum() == 0
                and metric_func != f1_score
            ):  # F1 handles zero TP/FP/FN
                metric_value = 0.0  # Or np.nan, depending on desired handling
                logger.warning(
                    f"No positive samples found for group '{col}' in y_true. Setting {metric_func.__name__} to 0."
                )
            elif metric_func == precision_score and y_pred_group.sum() == 0:
                metric_value = 0.0  # Or np.nan
                logger.warning(
                    f"No positive predictions found for group '{col}'. Setting precision to 0."
                )
            else:
                metric_value = metric_func(y_true_group, y_pred_group, zero_division=0)

            # Extract group name from column name (e.g., 'gender_f' -> 'F')
            # Handle potential '_nan' suffix as well
            group_name = col.replace(
                f"{sensitive_cols[0].split('_')[0]}_", ""
            )  # More robust extraction
            metrics[group_name] = metric_value
            logger.info(
                f"Metric ({metric_func.__name__}) for group '{group_name}': {metric_value:.4f}"
            )
        except Exception as e:
            logger.error(f"Error calculating metric for group '{col}': {e}")
            metrics[col.replace(f"{sensitive_cols[0].split('_')[0]}_", "")] = (
                np.nan
            )  # Use extracted group name

    return metrics


def plot_fairness_comparison(
    metrics: Dict[str, float],
    metric_name: str,
    sensitive_attribute: str,
    output_path: str,
) -> None:
    """Generates and saves a bar plot comparing fairness metrics."""
    if not metrics or all(
        np.isnan(v) for v in metrics.values()
    ):  # Check if metrics dict is empty or all NaN
        logger.warning(
            "No valid metrics calculated, skipping fairness plot generation."
        )
        return

    # Filter out NaN values before plotting
    valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
    if not valid_metrics:
        logger.warning(
            "All calculated metrics are NaN, skipping fairness plot generation."
        )
        return

    groups = list(valid_metrics.keys())
    values = list(valid_metrics.values())

    plt.figure(figsize=(8, 6))
    sns.barplot(x=groups, y=values)
    plt.title(
        f"{metric_name.replace('_', ' ').title()} Comparison by {sensitive_attribute.title()}"
    )
    plt.xlabel(sensitive_attribute.title())
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.ylim(0, max(1.0, max(values) * 1.1))  # Adjust ylim dynamically

    # Add value labels to bars
    for index, value in enumerate(values):
        plt.text(index, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Fairness comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate fairness analysis plots for a saved model pipeline."
    )
    parser.add_argument(
        "--pipeline-path",
        type=str,
        required=True,
        help="Path to the saved .pkl file containing the pipeline dictionary.",
    )
    parser.add_argument(
        "--sensitive-attribute",
        type=str,
        default="gender",
        help="Name of the sensitive attribute column prefix (e.g., 'gender', 'race'). Assumes one-hot encoding like 'prefix_value'.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        choices=["recall", "precision", "f1"],
        help="Metric to compare across groups (default: recall).",
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
        default=None,  # Default filename based on metric and attribute
        help="Filename for the output plot (default: fairness_[metric]_[attribute].png).",
    )

    args = parser.parse_args()

    # --- Load Pipeline ---
    pipeline_data = load_pipeline_data(args.pipeline_path)
    if pipeline_data is None:
        return
    pipeline = pipeline_data["pipeline"]
    pipeline_features = pipeline_data["features"]  # Features used by the pipeline

    # --- Load and Prepare Data ---
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load project config: {e}. Cannot proceed.")
        return

    preprocess_result = load_and_preprocess_data_for_fairness(
        config, pipeline_features, args.sensitive_attribute
    )
    if preprocess_result is None:
        logger.error(
            "Failed to load or preprocess data for fairness analysis. Cannot proceed."
        )
        return
    # X_processed now contains sensitive columns AND the features needed for the model
    X_processed, y_true, sensitive_cols = preprocess_result

    # --- Make Predictions ---
    try:
        # Extract only the features the pipeline expects for prediction
        X_to_predict = X_processed[
            pipeline_features
        ]  # Use the list of features the pipeline was trained on
        logger.info(f"Making predictions on {len(X_to_predict)} samples...")
        y_pred = pipeline.predict(X_to_predict)
        logger.info("Predictions completed.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return

    # --- Calculate Fairness Metrics ---
    metric_functions = {
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score,
    }
    metric_func = metric_functions.get(args.metric)
    if metric_func is None:  # Should not happen due to choices in argparse
        logger.error(f"Invalid metric specified: {args.metric}")
        return

    logger.info(
        f"Calculating fairness metrics for attribute '{args.sensitive_attribute}' using metric '{args.metric}'..."
    )
    fairness_metrics = calculate_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        X_processed=X_processed,  # Pass the dataframe containing sensitive columns
        sensitive_cols=sensitive_cols,
        metric_func=metric_func,
    )

    # --- Generate Plot ---
    if not args.output_filename:
        output_filename = f"fairness_{args.metric}_{args.sensitive_attribute}.png"
    else:
        output_filename = args.output_filename

    os.makedirs(args.output_dir, exist_ok=True)
    output_plot_path = os.path.join(args.output_dir, output_filename)

    plot_fairness_comparison(
        metrics=fairness_metrics,
        metric_name=args.metric,
        sensitive_attribute=args.sensitive_attribute,
        output_path=output_plot_path,
    )


if __name__ == "__main__":
    main()
