"""
Script to analyse and compare different imbalance handling techniques for the readmission model.

This script implements and compares the following techniques:
1. Baseline (no imbalance handling)
2. Class weights (class_weight='balanced')
3. Random oversampling
4. SMOTE (Synthetic Minority Over-sampling Technique)
5. Random undersampling

For each technique, it generates:
- PR curves
- F1 scores
- Precision
- Recall
- PR AUC
- Confusion Matrix (for a selected technique)
- Calibration Curve (for a selected technique)
- Feature Coefficients (for a selected technique)
- Feature Distribution plot (for a selected feature, if found)

It also saves the fitted pipeline for the selected technique for later use (e.g., SHAP analysis).

All metrics are computed using cross-validation to ensure robust evaluation.
Plots are saved to the 'assets' directory.
MLflow is used to log parameters, metrics, and artifacts.
"""

import argparse
import os
import pickle
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models.model import ReadmissionModel
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


# Helper function to get git hash (same as in train_model.py)
def get_git_revision_hash() -> str:
    """Gets the current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception as e:
        logger.warning(f"Could not get git hash: {e}")
        return "unknown"


def load_data() -> pd.DataFrame:
    """
    Load the processed data for analysis.

    Returns:
        pd.DataFrame: The processed data
    """
    config = load_config()
    data_path = get_data_path("processed", "combined_features", config)
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    return data


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data for modelling.

    Args:
        data (pd.DataFrame): The input data

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The preprocessed features and target
    """
    # Initialize a readmission model to use its preprocessing logic
    model = ReadmissionModel()
    X, y = model.preprocess(data)

    # Log class distribution
    if y is not None:  # Check if y is not None before using value_counts
        class_counts = y.value_counts()
        total = len(y)
        logger.info(f"Class distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count} ({count/total:.2%})")
    else:
        logger.warning("Target variable 'y' is None after preprocessing.")
        # Return empty DataFrame/Series if preprocessing failed to produce target
        return pd.DataFrame(), pd.Series(dtype="float64")

    return X, y


def create_imbalance_pipelines(random_state: int = 42) -> Dict[str, Any]:
    """
    Create pipelines for different imbalance handling techniques.

    Args:
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Dict[str, Any]: Dictionary of pipelines
    """
    # Base classifier
    # Increase max_iter to help with convergence warnings
    # base_clf = LogisticRegression(max_iter=2000, random_state=random_state) # Unused variable

    # Create pipelines
    pipelines = {
        "Baseline": Pipeline(
            [
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                )
            ]
        ),
        "Class Weights": Pipeline(
            [
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "Random Oversampling": Pipeline(
            [
                ("sampler", RandomOverSampler(random_state=random_state)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "SMOTE": Pipeline(
            [
                ("sampler", SMOTE(random_state=random_state)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "Random Undersampling": Pipeline(
            [
                ("sampler", RandomUnderSampler(random_state=random_state)),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
    }

    return pipelines


def evaluate_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Dict[str, Any],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate the pipelines using cross-validation.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        pipelines (Dict[str, Any]): Dictionary of pipelines
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
            - Dictionary of evaluation results
            - Dictionary of fitted pipelines for each technique (trained on full data)
    """
    results = {}
    fitted_pipelines = {}  # Store pipelines fitted on the full data
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Flag to track if any pipeline succeeded
    any_success = False

    for name, pipeline in pipelines.items():
        logger.info(f"Evaluating {name}...")

        # Initialize results dictionary for this pipeline
        results[name] = {
            "y_true": y,
            "y_pred": None,
            "y_prob": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "pr_auc": None,
            "precision_curve": None,
            "recall_curve": None,
            "thresholds": None,
            "success": False,  # Track success for each pipeline
            "pipeline": None,  # Store the fitted pipeline later
        }

        # Get cross-validated predictions and probabilities
        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=cv)
            y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[
                :, 1
            ]

            # Calculate metrics
            results[name]["y_pred"] = y_pred
            results[name]["y_prob"] = y_prob
            results[name]["precision"] = precision_score(
                y, y_pred, zero_division=0
            )  # Added zero_division
            results[name]["recall"] = recall_score(
                y, y_pred, zero_division=0
            )  # Added zero_division
            results[name]["f1"] = f1_score(
                y, y_pred, zero_division=0
            )  # Added zero_division
            results[name]["pr_auc"] = average_precision_score(y, y_prob)

            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y, y_prob)
            results[name]["precision_curve"] = precision
            results[name]["recall_curve"] = recall
            results[name]["thresholds"] = thresholds
            results[name]["success"] = True  # Mark as successful

            # Update the success flag
            any_success = True

            logger.info(f"  Precision: {results[name]['precision']:.4f}")
            logger.info(f"  Recall: {results[name]['recall']:.4f}")
            logger.info(f"  F1: {results[name]['f1']:.4f}")
            logger.info(f"  PR AUC: {results[name]['pr_auc']:.4f}")

            # Fit the pipeline on the full data for coefficient extraction etc.
            try:
                logger.info(f"  Fitting {name} on full data...")
                pipeline.fit(X, y)
                fitted_pipelines[name] = pipeline
                results[name][
                    "pipeline"
                ] = pipeline  # Store fitted pipeline in results too
                logger.info(f"  Fitting {name} on full data completed.")
            except Exception as fit_e:
                logger.error(f"  Error fitting {name} on full data: {str(fit_e)}")
                # Mark as failed if fitting on full data fails, even if CV worked
                results[name]["success"] = False
                any_success = any(
                    res.get("success", False)
                    for res_name, res in results.items()
                    if res_name != "_meta"
                )

        except Exception as e:
            logger.error(f"Error evaluating {name} during cross-validation: {str(e)}")
            # Ensure success is False if CV fails
            results[name]["success"] = False
            any_success = any(
                res.get("success", False)
                for res_name, res in results.items()
                if res_name != "_meta"
            )
            continue

    # Update the overall success status after all pipelines are processed
    results["_meta"] = {"any_success": any_success}

    if not any_success:
        logger.warning(
            "All pipelines failed evaluation or fitting. Check for data issues or model compatibility."
        )

    return results, fitted_pipelines


def plot_pr_curves(
    results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
) -> None:
    """
    Plot precision-recall curves for all pipelines.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    # Check if any pipeline succeeded
    if "_meta" in results and not results["_meta"]["any_success"]:
        logger.warning("No successful pipelines to plot PR curves for.")

        # Create a simple plot with a message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "No successful pipelines to compare.\nCheck logs for error details.",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty PR curve plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)
        return

    plt.figure(figsize=(10, 8))

    # Flag to track if any curves were plotted
    any_curves_plotted = False

    for name, result in results.items():
        # Skip the _meta entry
        if name == "_meta":
            continue

        if (
            result.get("success", False)
            and result["precision_curve"] is not None
            and result["recall_curve"] is not None
        ):
            plt.plot(
                result["recall_curve"],
                result["precision_curve"],
                label=f"{name} (PR AUC = {result['pr_auc']:.3f})",
            )
            any_curves_plotted = True

    if not any_curves_plotted:
        logger.warning(
            "No PR curves to plot. All pipelines failed or had missing data."
        )
        plt.clf()  # Clear the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "No PR curves to plot.\nAll pipelines failed or had missing data.",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()
    else:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves for Different Imbalance Handling Techniques")
        plt.legend(loc="best")
        plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PR curve plot to {save_path}")
    else:
        plt.show()
    plt.close()  # Close the figure


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
) -> None:
    """
    Plot a comparison of metrics for all pipelines.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    # Check if any pipeline succeeded
    if "_meta" in results and not results["_meta"]["any_success"]:
        logger.warning("No successful pipelines to plot metrics for.")

        # Create a simple plot with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No successful pipelines to compare.\nCheck logs for error details.",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty metrics comparison plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)
        return

    # Filter out the _meta entry and any failed pipelines
    pipeline_names = [
        name
        for name in results.keys()
        if name != "_meta" and results[name].get("success", False)
    ]

    if not pipeline_names:
        logger.warning("No successful pipelines to plot metrics for.")

        # Create a simple plot with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No successful pipelines to compare.\nCheck logs for error details.",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty metrics comparison plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)
        return

    metrics = ["precision", "recall", "f1", "pr_auc"]

    # Extract metrics for each pipeline
    metric_values = {
        metric: [results[name][metric] for name in pipeline_names] for metric in metrics
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(pipeline_names))
    width = 0.2
    multiplier = 0

    for metric, values in metric_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric.upper())
        ax.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    ax.set_ylabel("Score")
    ax.set_title("Comparison of Metrics Across Imbalance Handling Techniques")
    ax.set_xticks(x + width * 1.5, pipeline_names)  # Adjust x-ticks position
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics comparison plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_results_to_csv(results: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """
    Save the evaluation results to a CSV file.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results
        save_path (str): Path to save the CSV file
    """
    # Check if any pipeline succeeded
    if "_meta" in results and not results["_meta"]["any_success"]:
        logger.warning("No successful pipelines to save results for.")

        # Create a simple CSV with a message
        df = pd.DataFrame(
            [
                {
                    "Technique": "No successful pipelines",
                    "Precision": None,
                    "Recall": None,
                    "F1": None,
                    "PR_AUC": None,
                }
            ]
        )
    else:
        # Filter out the _meta entry and any failed pipelines
        valid_results = {
            name: res
            for name, res in results.items()
            if name != "_meta" and res.get("success", False)
        }

        if not valid_results:
            logger.warning("No successful pipelines to save results for.")
            df = pd.DataFrame(
                [
                    {
                        "Technique": "No successful pipelines",
                        "Precision": None,
                        "Recall": None,
                        "F1": None,
                        "PR_AUC": None,
                    }
                ]
            )
        else:
            # Prepare data for DataFrame
            data_for_df = []
            for name, result in valid_results.items():
                data_for_df.append(
                    {
                        "Technique": name,
                        "Precision": result.get("precision"),
                        "Recall": result.get("recall"),
                        "F1": result.get("f1"),
                        "PR_AUC": result.get("pr_auc"),
                    }
                )
            df = pd.DataFrame(data_for_df)

    # Save to CSV
    try:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved evaluation results to {save_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV {save_path}: {e}")


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot the confusion matrix.

    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_calibration_curve(
    pipeline: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    technique_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the calibration curve for a fitted pipeline.

    Args:
        pipeline (Any): The fitted pipeline (must have predict_proba method)
        X (pd.DataFrame): Features
        y_true (pd.Series): True labels
        technique_name (str): Name of the technique being plotted
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    try:
        # Check if the final step (classifier) has predict_proba
        if not hasattr(pipeline.steps[-1][1], "predict_proba"):
            logger.warning(
                f"Classifier in pipeline '{technique_name}' does not support predict_proba. Skipping calibration curve."
            )
            return

        y_prob = pipeline.predict_proba(X)[:, 1]

        plt.figure(figsize=(10, 8))
        disp = CalibrationDisplay.from_predictions(
            y_true, y_prob, n_bins=10, name=technique_name
        )
        disp.plot()  # Use the plot method of the display object

        plt.title(f"Calibration Curve for {technique_name}")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved calibration curve plot to {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting calibration curve for {technique_name}: {e}")


def plot_feature_coefficients(
    classifier: Any, feature_names: List[str], save_path: Optional[str] = None
) -> None:
    """
    Plot feature coefficients for a linear model.

    Args:
        classifier (Any): The fitted linear classifier (must have .coef_ attribute)
        feature_names (List[str]): List of feature names
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    if not hasattr(classifier, "coef_"):
        logger.warning(
            "Classifier does not have 'coef_' attribute. Skipping feature coefficients plot."
        )
        return

    try:
        coefficients = classifier.coef_[0]
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
        coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

        # Plot top N and bottom N features for clarity
        top_n = 15
        bottom_n = 15
        if len(coef_df) > top_n + bottom_n:
            plot_df = pd.concat([coef_df.head(top_n), coef_df.tail(bottom_n)])
            title = f"Top {top_n} and Bottom {bottom_n} Feature Coefficients"
        else:
            plot_df = coef_df
            title = "Feature Coefficients"

        plt.figure(
            figsize=(12, max(8, len(plot_df) * 0.3))
        )  # Adjust height based on number of features
        sns.barplot(x="Coefficient", y="Feature", data=plot_df, palette="viridis")
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature coefficients plot to {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting feature coefficients: {e}")


def plot_feature_distribution(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    technique_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of a specific feature for each class.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        feature_name (str): Name of the feature to plot
        technique_name (str): Name of the technique (used for title)
        save_path (Optional[str], optional): Path to save the plot. If None, the plot is displayed.
            Defaults to None.
    """
    if feature_name not in X.columns:
        logger.warning(
            f"Feature '{feature_name}' not found in data. Skipping distribution plot."
        )
        return

    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=X.assign(target=y), x=feature_name, hue="target", kde=True)
        plt.title(
            f"Distribution of '{feature_name}' by Target Class ({technique_name})"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(
                f"Saved feature distribution plot for '{feature_name}' to {save_path}"
            )
        else:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting feature distribution for '{feature_name}': {e}")


def analyze_imbalance_techniques(
    cv_folds: int = 5,
    random_state: int = 42,
    selected_technique: str = "Class Weights",
    feature_for_distribution: Optional[str] = "age",
    config: Optional[Dict] = None,  # Add config for MLflow experiment name
) -> None:
    """
    Main function to run the imbalance analysis, generate plots, save results,
    and log everything to MLflow.

    Args:
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        selected_technique (str, optional): The technique for detailed plots. Defaults to "Class Weights".
        feature_for_distribution (Optional[str], optional): Feature for distribution plot. Defaults to "age".
        config (Optional[Dict], optional): Configuration dictionary. Defaults to None.
    """
    if config is None:
        config = load_config()  # Load config if not provided

    # --- MLflow Setup ---
    git_hash = get_git_revision_hash()
    experiment_name = config.get("mlflow", {}).get(
        "experiment_name", "MIMIC_Imbalance_Analysis"
    )
    mlflow.set_experiment(experiment_name)
    run_name = f"imbalance_analysis_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Starting MLflow Run: {run.info.run_name} ({run.info.run_id})")
        logger.info(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        # Log parameters
        mlflow.log_params(
            {
                "cv_folds": cv_folds,
                "random_state": random_state,
                "selected_technique_detailed_plots": selected_technique,
                "feature_for_distribution_plot": feature_for_distribution or "None",
                "git_hash": git_hash,
            }
        )

        # --- Original Logic ---
        # Load and preprocess data
        data = load_data()
        X, y = preprocess_data(data)

        if X.empty or y.empty:
            logger.error("Preprocessing failed. Exiting analysis.")
            mlflow.log_param("status", "failed_preprocessing")
            return

        # Create pipelines
        pipelines = create_imbalance_pipelines(random_state=random_state)

        # Evaluate pipelines
        results, fitted_pipelines = evaluate_pipelines(
            X, y, pipelines, cv_folds=cv_folds, random_state=random_state
        )

        # Log metrics for each technique
        logger.info("Logging metrics to MLflow...")
        for name, result in results.items():
            if name != "_meta" and result.get("success", False):
                metrics_to_log = {
                    f"{name}_precision": result.get("precision"),
                    f"{name}_recall": result.get("recall"),
                    f"{name}_f1": result.get("f1"),
                    f"{name}_pr_auc": result.get("pr_auc"),
                }
                # Filter out None values before logging
                metrics_to_log = {
                    k: v for k, v in metrics_to_log.items() if v is not None
                }
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)
                    logger.info(f"  Logged metrics for {name}")
                else:
                    logger.warning(f"  No metrics to log for {name}")
            elif name != "_meta":
                logger.warning(
                    f"  Skipping logging metrics for failed/incomplete pipeline: {name}"
                )

        # Create directories for assets and results if they don't exist
        assets_dir = os.path.join(get_project_root(), "assets")
        results_dir = os.path.join(get_project_root(), "results", "imbalance_analysis")
        os.makedirs(assets_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Plot PR curves
        pr_curve_path = os.path.join(assets_dir, "imbalance_pr_curves.png")
        plot_pr_curves(results, save_path=pr_curve_path)
        if os.path.exists(pr_curve_path):
            mlflow.log_artifact(pr_curve_path, artifact_path="plots")
            logger.info(f"Logged PR curve plot: {pr_curve_path}")
        else:
            logger.warning(
                f"PR curve plot not found at {pr_curve_path}, skipping MLflow logging."
            )

        # Plot metrics comparison
        metrics_plot_path = os.path.join(assets_dir, "imbalance_metrics_comparison.png")
        plot_metrics_comparison(results, save_path=metrics_plot_path)
        if os.path.exists(metrics_plot_path):
            mlflow.log_artifact(metrics_plot_path, artifact_path="plots")
            logger.info(f"Logged metrics comparison plot: {metrics_plot_path}")
        else:
            logger.warning(
                f"Metrics comparison plot not found at {metrics_plot_path}, skipping MLflow logging."
            )

        # Save results to CSV
        csv_path = os.path.join(results_dir, "imbalance_metrics.csv")
        save_results_to_csv(results, save_path=csv_path)
        if os.path.exists(csv_path):
            mlflow.log_artifact(csv_path, artifact_path="results")
            logger.info(f"Logged results CSV: {csv_path}")
        else:
            logger.warning(
                f"Results CSV not found at {csv_path}, skipping MLflow logging."
            )

        # Generate detailed plots and save pipeline for the selected technique
        if selected_technique in results and results[selected_technique].get(
            "success", False
        ):
            logger.info(f"Generating detailed plots for '{selected_technique}'...")
            result = results[selected_technique]
            pipeline_to_log = fitted_pipelines.get(
                selected_technique
            )  # Use fitted pipeline

            # Confusion Matrix
            cm_path = os.path.join(
                assets_dir,
                f"confusion_matrix_{selected_technique.lower().replace(' ', '_')}.png",
            )
            if result["y_true"] is not None and result["y_pred"] is not None:
                plot_confusion_matrix(
                    result["y_true"], result["y_pred"], save_path=cm_path
                )
                if os.path.exists(cm_path):
                    mlflow.log_artifact(cm_path, artifact_path="plots")
                    logger.info(f"Logged confusion matrix plot: {cm_path}")
                else:
                    logger.warning(
                        f"Confusion matrix plot not found at {cm_path}, skipping MLflow logging."
                    )

            # Calibration Curve
            cal_curve_path = os.path.join(
                assets_dir,
                f"calibration_curve_{selected_technique.lower().replace(' ', '_')}.png",
            )
            if (
                pipeline_to_log
                and result["y_true"] is not None
                and result["y_prob"] is not None
            ):
                plot_calibration_curve(
                    pipeline_to_log,
                    X,
                    result["y_true"],
                    selected_technique,
                    save_path=cal_curve_path,
                )
                if os.path.exists(cal_curve_path):
                    mlflow.log_artifact(cal_curve_path, artifact_path="plots")
                    logger.info(f"Logged calibration curve plot: {cal_curve_path}")
                else:
                    logger.warning(
                        f"Calibration curve plot not found at {cal_curve_path}, skipping MLflow logging."
                    )

            # Feature Coefficients (only for Logistic Regression)
            coeffs_path = os.path.join(
                assets_dir,
                f"feature_coefficients_{selected_technique.lower().replace(' ', '_')}.png",
            )
            if pipeline_to_log:
                classifier = pipeline_to_log.named_steps.get("classifier")
                if isinstance(classifier, LogisticRegression):
                    plot_feature_coefficients(
                        classifier,
                        list(X.columns),
                        save_path=coeffs_path,  # Ensure X.columns is a list
                    )
                    if os.path.exists(coeffs_path):
                        mlflow.log_artifact(coeffs_path, artifact_path="plots")
                        logger.info(f"Logged feature coefficients plot: {coeffs_path}")
                    else:
                        logger.warning(
                            f"Feature coefficients plot not found at {coeffs_path}, skipping MLflow logging."
                        )

            # Feature Distribution Plot
            if feature_for_distribution and feature_for_distribution in X.columns:
                dist_path = os.path.join(
                    assets_dir,
                    f"feature_distribution_{feature_for_distribution}_{selected_technique.lower().replace(' ', '_')}.png",
                )
                plot_feature_distribution(
                    X,
                    y,
                    feature_for_distribution,
                    selected_technique,
                    save_path=dist_path,
                )
                if os.path.exists(dist_path):
                    mlflow.log_artifact(dist_path, artifact_path="plots")
                    logger.info(f"Logged feature distribution plot: {dist_path}")
                else:
                    logger.warning(
                        f"Feature distribution plot not found at {dist_path}, skipping MLflow logging."
                    )

            # Save and log the fitted pipeline for the selected technique
            if pipeline_to_log:
                pipeline_dir = os.path.join(
                    get_project_root(), "models", "imbalance_analysis"
                )
                os.makedirs(pipeline_dir, exist_ok=True)
                pipeline_filename = (
                    f"pipeline_{selected_technique.lower().replace(' ', '_')}.pkl"
                )
                pipeline_path = os.path.join(pipeline_dir, pipeline_filename)

                try:
                    with open(pipeline_path, "wb") as f:
                        pickle.dump(pipeline_to_log, f)
                    logger.info(
                        f"Saved fitted pipeline for '{selected_technique}' to {pipeline_path}"
                    )
                    mlflow.log_artifact(pipeline_path, artifact_path="pipelines")
                    logger.info(f"Logged pipeline artifact: {pipeline_path}")
                except Exception as e:
                    logger.error(
                        f"Error saving or logging pipeline {pipeline_path}: {e}"
                    )

        else:
            logger.warning(
                f"Selected technique '{selected_technique}' not found in results or failed evaluation. "
                "Skipping detailed plots and pipeline saving/logging."
            )
            mlflow.log_param(
                "status", f"failed_selected_technique_{selected_technique}"
            )

        logger.info("Imbalance analysis complete.")
        mlflow.log_param("status", "completed")
        logger.info(f"MLflow Run {run.info.run_id} finished.")


def main() -> None:
    """
    Main function to parse arguments and run the imbalance analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze imbalance handling techniques for readmission prediction."
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="Class Weights",
        help="Technique for detailed plots",
    )
    parser.add_argument(
        "--feature", type=str, default="age", help="Feature name for distribution plot"
    )
    parser.add_argument(  # Add config argument
        "--config", type=str, default=None, help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration if specified
    config = load_config(args.config) if args.config else load_config()

    analyze_imbalance_techniques(
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        selected_technique=args.technique,
        feature_for_distribution=args.feature,
        config=config,  # Pass config
    )


if __name__ == "__main__":
    main()
