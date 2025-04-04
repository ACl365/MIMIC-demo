"""
Script to analyse and compare different imbalance handling techniques for the readmission model.

This script loads processed data, preprocesses it for the readmission task,
and then trains and evaluates a Logistic Regression model using several
imbalance handling strategies implemented via `imblearn` pipelines:

1.  **Baseline:** No explicit imbalance handling.
2.  **Class Weights:** Uses `class_weight='balanced'` in Logistic Regression.
3.  **Random Oversampling:** Uses `RandomOverSampler`.
4.  **SMOTE:** Uses `SMOTE` (Synthetic Minority Over-sampling Technique).
5.  **Random Undersampling:** Uses `RandomUnderSampler`.

For each technique, it performs cross-validation to calculate and log metrics:
- Precision, Recall, F1-score, PR AUC

It generates and saves comparison plots:
- Precision-Recall Curves (`imbalance_pr_curves.png`)
- Bar chart comparing key metrics (`imbalance_metrics_comparison.png`)

Additionally, for a selected 'best' technique (defaulting to 'Class Weights' or the
first successful one), it generates and saves detailed plots:
- Confusion Matrix (`confusion_matrix_<technique>.png`)
- Calibration Curve (`calibration_curve_<technique>.png`)
- Feature Coefficients (if applicable) (`feature_coefficients_<technique>.png`)

The script also saves the evaluation results to a CSV file (`imbalance_analysis_results.csv`)
and logs parameters, metrics, and plot artifacts to MLflow.
"""

import argparse
import logging  # Import logging for type hint
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
from imblearn.pipeline import Pipeline as ImbPipeline  # Alias to avoid confusion
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
from sklearn.preprocessing import StandardScaler  # Needed for pipelines

from src.models.model import ReadmissionModel
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


# Helper function to get git hash (same as in train_model.py)
def get_git_revision_hash() -> str:
    """Gets the current git commit hash of the repository."""
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
    Load the combined features data used for model training.

    Returns:
        pd.DataFrame: The combined features DataFrame.

    Raises:
        FileNotFoundError: If the combined features file is not found.
    """
    config = load_config()
    data_path = get_data_path("processed", "combined_features", config)
    logger.info(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    return data


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the loaded data using the ReadmissionModel's preprocess method.

    Handles feature selection, imputation, and target extraction based on the
    model's configuration. Logs the class distribution of the target variable.

    Args:
        data (pd.DataFrame): The input DataFrame (combined features).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): The preprocessed feature matrix.
            - y (pd.Series): The target variable Series.
            Returns empty DataFrame/Series if preprocessing fails or target is missing.
    """
    # Initialize a readmission model to use its preprocessing logic
    # Assumes default config is sufficient for preprocessing steps needed here
    model = ReadmissionModel()
    X, y = model.preprocess(data)

    # Log class distribution
    if y is not None and not y.empty:
        class_counts = y.value_counts()
        total = len(y)
        logger.info(f"Class distribution (Target: {model.target}):")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count} ({count/total:.2%})")
    else:
        logger.warning("Target variable 'y' is None or empty after preprocessing.")
        # Return empty DataFrame/Series if preprocessing failed to produce target
        return pd.DataFrame(), pd.Series(dtype="float64")

    # Scale features (important before applying sampling techniques)
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    logger.info("Features scaled.")

    return X, y


def create_imbalance_pipelines(random_state: int = 42) -> Dict[str, ImbPipeline]:
    """
    Create scikit-learn/imblearn pipelines for different imbalance handling techniques.

    Each pipeline includes a StandardScaler and a Logistic Regression classifier.
    Sampling techniques (Oversampling, SMOTE, Undersampling) are added before the classifier.

    Args:
        random_state (int, optional): Random state for reproducibility of sampling and classifier.
                                      Defaults to 42.

    Returns:
        Dict[str, ImbPipeline]: A dictionary where keys are technique names (e.g., "Baseline", "SMOTE")
                                and values are the corresponding imblearn Pipeline objects.
    """
    # Base classifier configuration
    lr_config = {
        "max_iter": 2000,  # Increased for convergence
        "random_state": random_state,
        "solver": "liblinear",  # Often good with L1/L2
    }

    # Create pipelines
    pipelines = {
        "Baseline": ImbPipeline(
            [
                ("scaler", StandardScaler()),  # Add scaler to each pipeline
                ("classifier", LogisticRegression(**lr_config)),
            ]
        ),
        "Class Weights": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(**lr_config, class_weight="balanced"),
                ),
            ]
        ),
        "Random Oversampling": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("sampler", RandomOverSampler(random_state=random_state)),
                ("classifier", LogisticRegression(**lr_config)),
            ]
        ),
        "SMOTE": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("sampler", SMOTE(random_state=random_state)),
                ("classifier", LogisticRegression(**lr_config)),
            ]
        ),
        "Random Undersampling": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("sampler", RandomUnderSampler(random_state=random_state)),
                ("classifier", LogisticRegression(**lr_config)),
            ]
        ),
    }

    return pipelines


def evaluate_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Dict[str, ImbPipeline],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, ImbPipeline]]:
    """
    Evaluate multiple imblearn pipelines using stratified cross-validation.

    Calculates predictions, probabilities, and various performance metrics
    (Precision, Recall, F1, PR AUC) for each pipeline. Also fits each pipeline
    on the full dataset for later analysis (e.g., coefficients).

    Args:
        X (pd.DataFrame): Preprocessed and scaled features.
        y (pd.Series): Target variable.
        pipelines (Dict[str, ImbPipeline]): Dictionary of pipelines to evaluate.
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], Dict[str, ImbPipeline]]:
            - results (Dict[str, Dict[str, Any]]): A dictionary where keys are pipeline names.
              Each value is another dictionary containing metrics ('precision', 'recall', 'f1', 'pr_auc'),
              cross-validated predictions ('y_pred', 'y_prob'), raw curve data
              ('precision_curve', 'recall_curve', 'thresholds'), a 'success' flag,
              and the fitted pipeline object ('pipeline'). Includes a '_meta' key indicating
              if any pipeline succeeded overall.
            - fitted_pipelines (Dict[str, ImbPipeline]): A dictionary containing pipelines
              fitted on the full training data.
    """
    results: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, ImbPipeline] = (
        {}
    )  # Store pipelines fitted on the full data
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Flag to track if any pipeline succeeded
    any_success = False

    for name, pipeline in pipelines.items():
        logger.info(f"Evaluating {name}...")

        # Initialize results dictionary for this pipeline
        results[name] = {
            "y_true": y.to_numpy(),  # Store numpy array for consistency
            "y_pred": None,
            "y_prob": None,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "pr_auc": np.nan,
            "precision_curve": None,
            "recall_curve": None,
            "thresholds": None,
            "success": False,  # Track success for each pipeline
            "pipeline": None,  # Store the fitted pipeline later
        }

        # Get cross-validated predictions and probabilities
        try:
            # Ensure X and y are numpy arrays for cross_val_predict if needed, though it often handles DataFrames
            X_np = X.values if isinstance(X, pd.DataFrame) else X
            y_np = y.values if isinstance(y, pd.Series) else y

            y_pred = cross_val_predict(pipeline, X_np, y_np, cv=cv, n_jobs=-1)
            y_prob = cross_val_predict(
                pipeline, X_np, y_np, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]

            # Calculate metrics
            results[name]["y_pred"] = y_pred
            results[name]["y_prob"] = y_prob
            results[name]["precision"] = precision_score(y_np, y_pred, zero_division=0)
            results[name]["recall"] = recall_score(y_np, y_pred, zero_division=0)
            results[name]["f1"] = f1_score(y_np, y_pred, zero_division=0)
            results[name]["pr_auc"] = average_precision_score(y_np, y_prob)

            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_np, y_prob)
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
                pipeline.fit(X_np, y_np)
                fitted_pipelines[name] = pipeline
                results[name][
                    "pipeline"
                ] = pipeline  # Store fitted pipeline in results too
                logger.info(f"  Fitting {name} on full data completed.")
            except Exception as fit_e:
                logger.error(f"  Error fitting {name} on full data: {str(fit_e)}")
                # Mark as failed if fitting on full data fails, even if CV worked
                results[name]["success"] = False
                # Recalculate any_success based on current results status
                any_success = any(
                    res.get("success", False)
                    for res_name, res in results.items()
                    if res_name != "_meta"
                )

        except Exception as e:
            logger.error(f"Error evaluating {name} during cross-validation: {str(e)}")
            # Ensure success is False if CV fails
            results[name]["success"] = False
            # Recalculate any_success based on current results status
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
    Plot precision-recall curves for all successfully evaluated pipelines.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results from `evaluate_pipelines`.
        save_path (Optional[str], optional): Path to save the plot image. If None, displays the plot.
                                             Defaults to None.
    """
    # Check if any pipeline succeeded
    if "_meta" in results and not results["_meta"]["any_success"]:
        logger.warning("No successful pipelines to plot PR curves for.")
        # Optionally create an empty plot with a message if saving
        if save_path:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                "No successful pipelines to compare.",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_axis_off()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty PR curve plot placeholder to {save_path}")
            plt.close(fig)
        return

    plt.figure(figsize=(10, 8))
    any_curves_plotted = False

    for name, result in results.items():
        if name == "_meta":
            continue  # Skip metadata entry

        if (
            result.get("success", False)
            and result.get("precision_curve") is not None
            and result.get("recall_curve") is not None
            and result.get("pr_auc") is not None
        ):
            plt.plot(
                result["recall_curve"],
                result["precision_curve"],
                label=f"{name} (PR AUC = {result['pr_auc']:.3f})",
                lw=2,  # Line width
            )
            any_curves_plotted = True
        elif result.get("success", False):
            logger.warning(
                f"Pipeline '{name}' succeeded but missing curve data for plotting."
            )

    if not any_curves_plotted:
        logger.warning("No valid PR curves generated to plot.")
        plt.close()  # Close the empty figure
        # Optionally save an empty plot placeholder
        if save_path:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                "No valid PR curves generated.",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_axis_off()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty PR curve plot placeholder to {save_path}")
            plt.close(fig)
        return

    # Finalize plot
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curves for Imbalance Handling Techniques")
    plt.legend(loc="lower left")  # Often better for PR curves
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])  # Allow space for labels

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PR curve plot to {save_path}")
        mlflow.log_artifact(save_path)  # Log plot to MLflow
    else:
        plt.show()
    plt.close()  # Close the figure


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None
) -> None:
    """
    Plot a bar chart comparing key metrics (Precision, Recall, F1, PR AUC)
    across all successfully evaluated pipelines.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results.
        save_path (Optional[str], optional): Path to save the plot image. If None, displays the plot.
                                             Defaults to None.
    """
    # Check if any pipeline succeeded
    if "_meta" in results and not results["_meta"]["any_success"]:
        logger.warning("No successful pipelines to plot metrics for.")
        if save_path:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No successful pipelines to compare.",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_axis_off()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(
                f"Saved empty metrics comparison plot placeholder to {save_path}"
            )
            plt.close(fig)
        return

    # Filter out the _meta entry and any failed pipelines
    pipeline_names = [
        name
        for name in results.keys()
        if name != "_meta" and results[name].get("success", False)
    ]

    if not pipeline_names:
        logger.warning("No successful pipelines found to plot metrics for.")
        if save_path:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No successful pipelines found.",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_axis_off()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(
                f"Saved empty metrics comparison plot placeholder to {save_path}"
            )
            plt.close(fig)
        return

    metrics_to_plot = ["precision", "recall", "f1", "pr_auc"]

    # Extract metrics for each successful pipeline, handling potential missing values
    metric_values = {
        metric: [results[name].get(metric, np.nan) for name in pipeline_names]
        for metric in metrics_to_plot
    }

    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame(metric_values, index=pipeline_names)

    # Create the plot
    ax = metrics_df.plot(kind="bar", figsize=(12, 8), rot=45, width=0.8)

    # Add labels and title
    ax.set_ylabel("Score")
    ax.set_title("Comparison of Metrics Across Imbalance Handling Techniques")
    ax.set_xlabel("Technique")
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, max(1.0, metrics_df.max().max() * 1.1))  # Adjust ylim dynamically
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}", label_type="edge", padding=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics comparison plot to {save_path}")
        mlflow.log_artifact(save_path)  # Log plot to MLflow
    else:
        plt.show()
    plt.close()  # Close the figure


def save_results_to_csv(results: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """
    Save the calculated metrics to a CSV file.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of evaluation results.
        save_path (str): Path to save the CSV file.
    """
    metrics_to_save = ["precision", "recall", "f1", "pr_auc"]
    summary_data = []

    for name, result in results.items():
        if name == "_meta" or not result.get("success", False):
            continue
        # Explicitly type row to allow for metric values (Any) or None
        row: Dict[str, Any | None] = {"Technique": name}
        for metric in metrics_to_save:
            row[metric.upper()] = result.get(metric)
        summary_data.append(row)

    if not summary_data:
        logger.warning("No successful results to save to CSV.")
        return

    summary_df = pd.DataFrame(summary_data)
    try:
        summary_df.to_csv(save_path, index=False, float_format="%.4f")
        logger.info(f"Saved evaluation summary to {save_path}")
        mlflow.log_artifact(save_path)  # Log CSV to MLflow
    except Exception as e:
        logger.error(f"Failed to save results CSV to {save_path}: {e}", exc_info=True)


def plot_confusion_matrix(
    # Specify dtype for ndarray (labels are integers)
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    y_pred: np.ndarray[Any, np.dtype[np.int_]],
    technique_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save the confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        technique_name (str): Name of the technique for the title and filename.
        save_path (Optional[str], optional): Path to save the plot. If None, displays the plot.
                                             Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Non-Readmit", "Predicted Readmit"],
        yticklabels=["Actual Non-Readmit", "Actual Readmit"],
    )
    plt.title(f"Confusion Matrix - {technique_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {save_path}")
        mlflow.log_artifact(save_path)
    else:
        plt.show()
    plt.close()


def plot_calibration_curve(
    # Specify dtype for ndarray (labels are int, probabilities are float)
    y_true: np.ndarray[Any, np.dtype[np.int_]],
    y_prob: np.ndarray[Any, np.dtype[np.float64]],
    technique_name: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save the calibration curve.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        technique_name (str): Name of the technique for the title and filename.
        save_path (Optional[str], optional): Path to save the plot. If None, displays the plot.
                                             Defaults to None.
    """
    plt.figure(figsize=(10, 8))
    disp = CalibrationDisplay.from_predictions(
        y_true, y_prob, n_bins=10, name=technique_name
    )
    plt.title(f"Calibration Curve - {technique_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved calibration curve plot to {save_path}")
        mlflow.log_artifact(save_path)
    else:
        plt.show()
    plt.close()


def plot_feature_coefficients(
    pipeline: ImbPipeline,
    feature_names: List[str],
    technique_name: str,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the feature coefficients/importances of the classifier in the pipeline.

    Args:
        pipeline (ImbPipeline): The fitted pipeline containing the classifier.
        feature_names (List[str]): List of feature names corresponding to the input data.
        technique_name (str): Name of the technique for the title and filename.
        top_n (int, optional): Number of top features to display. Defaults to 20.
        save_path (Optional[str], optional): Path to save the plot. If None, displays the plot.
                                             Defaults to None.
    """
    try:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            coefficients = classifier.coef_[0]  # Assuming binary classification
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Coefficient": coefficients}
            )
            importance_df["Absolute Coefficient"] = importance_df["Coefficient"].abs()
            importance_df = importance_df.sort_values(
                "Absolute Coefficient", ascending=False
            ).head(top_n)

            plt.figure(figsize=(10, max(6, top_n // 2)))  # Adjust height based on N
            sns.barplot(
                x="Coefficient", y="Feature", data=importance_df, palette="viridis"
            )
            plt.title(f"Top {top_n} Feature Coefficients - {technique_name}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved feature coefficients plot to {save_path}")
                mlflow.log_artifact(save_path)
            else:
                plt.show()
            plt.close()

        elif hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )
            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            ).head(top_n)

            plt.figure(figsize=(10, max(6, top_n // 2)))
            sns.barplot(
                x="Importance", y="Feature", data=importance_df, palette="viridis"
            )
            plt.title(f"Top {top_n} Feature Importances - {technique_name}")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved feature importances plot to {save_path}")
                mlflow.log_artifact(save_path)
            else:
                plt.show()
            plt.close()
        else:
            logger.warning(
                f"Classifier for {technique_name} has no 'coef_' or 'feature_importances_' attribute."
            )

    except Exception as e:
        logger.error(
            f"Error plotting feature coefficients/importances for {technique_name}: {e}",
            exc_info=True,
        )


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
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        feature_name (str): The name of the feature column in X to plot.
        technique_name (str): Name of the technique for the title and filename.
        save_path (Optional[str], optional): Path to save the plot. If None, displays the plot.
                                             Defaults to None.
    """
    if feature_name not in X.columns:
        logger.warning(
            f"Feature '{feature_name}' not found in data. Cannot plot distribution."
        )
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=X, x=feature_name, hue=y, kde=False, stat="density", common_norm=False
    )
    plt.title(f"Distribution of {feature_name} by Class - {technique_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(
            f"Saved feature distribution plot for '{feature_name}' to {save_path}"
        )
        mlflow.log_artifact(save_path)
    else:
        plt.show()
    plt.close()


def analyze_imbalance_techniques(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    selected_technique: str = "Class Weights",
    feature_to_plot: Optional[str] = "age",  # Example feature
    random_state: int = 42,
) -> None:
    """
    Run the full imbalance analysis pipeline.

    Creates pipelines, evaluates them, generates comparison plots,
    generates detailed plots for the selected technique, saves results,
    and logs everything to MLflow.

    Args:
        X (pd.DataFrame): Preprocessed and scaled features.
        y (pd.Series): Target variable.
        config (Dict[str, Any]): Configuration dictionary.
        selected_technique (str, optional): The technique for which to generate detailed plots
                                            (confusion matrix, calibration, coefficients).
                                            Defaults to "Class Weights".
        feature_to_plot (Optional[str], optional): A specific feature name to plot its distribution
                                                   by class for the selected technique. Defaults to "age".
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
    """
    logger.info("Starting imbalance analysis...")
    git_hash = get_git_revision_hash()
    experiment_name = config.get("mlflow", {}).get(
        "experiment_name", "MIMIC_Imbalance_Analysis"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"Imbalance_Comparison_{git_hash[:7]}"):
        logger.info(f"MLflow Run ID for comparison: {mlflow.active_run().info.run_id}")
        mlflow.log_param("analysis_type", "imbalance_comparison")
        mlflow.log_param("git_hash", git_hash)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param(
            "cv_folds",
            config.get("models", {}).get("readmission", {}).get("cv_folds", 5),
        )

        # Create pipelines
        pipelines = create_imbalance_pipelines(random_state=random_state)

        # Evaluate pipelines
        cv_folds = config.get("models", {}).get("readmission", {}).get("cv_folds", 5)
        results, fitted_pipelines = evaluate_pipelines(
            X, y, pipelines, cv_folds=cv_folds, random_state=random_state
        )

        # --- Generate and Save Plots ---
        assets_dir = os.path.join(get_project_root(), "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Plot comparison curves and metrics
        plot_pr_curves(
            results, save_path=os.path.join(assets_dir, "imbalance_pr_curves.png")
        )
        plot_metrics_comparison(
            results,
            save_path=os.path.join(assets_dir, "imbalance_metrics_comparison.png"),
        )

        # Save summary results
        results_csv_path = os.path.join(
            get_project_root(), "results", "imbalance_analysis", "metrics_summary.csv"
        )
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        save_results_to_csv(results, results_csv_path)

        # --- Detailed Plots for Selected Technique ---
        # Choose the best technique based on config or default, ensure it exists and succeeded
        valid_techniques = [
            name
            for name, res in results.items()
            if name != "_meta" and res.get("success")
        ]
        if not valid_techniques:
            logger.error("No pipelines succeeded. Cannot generate detailed plots.")
            return  # Exit if no pipeline worked

        if selected_technique not in valid_techniques:
            logger.warning(
                f"Selected technique '{selected_technique}' not found among successful pipelines or failed. Defaulting to '{valid_techniques[0]}'."
            )
            selected_technique = valid_techniques[0]

        logger.info(
            f"Generating detailed plots for selected technique: {selected_technique}"
        )

        selected_results = results[selected_technique]
        selected_pipeline = fitted_pipelines.get(
            selected_technique
        )  # Get from fitted pipelines dict

        if selected_pipeline is None:
            logger.error(
                f"Fitted pipeline for '{selected_technique}' not found. Cannot generate detailed plots."
            )
            return

        # Confusion Matrix
        if selected_results["y_pred"] is not None:
            plot_confusion_matrix(
                selected_results["y_true"],
                selected_results["y_pred"],
                selected_technique,
                save_path=os.path.join(
                    assets_dir,
                    f"confusion_matrix_{selected_technique.lower().replace(' ', '_')}.png",
                ),
            )

        # Calibration Curve
        if selected_results["y_prob"] is not None:
            plot_calibration_curve(
                selected_results["y_true"],
                selected_results["y_prob"],
                selected_technique,
                save_path=os.path.join(
                    assets_dir,
                    f"calibration_curve_{selected_technique.lower().replace(' ', '_')}.png",
                ),
            )

        # Feature Coefficients/Importances
        plot_feature_coefficients(
            selected_pipeline,
            list(X.columns),  # Feature names from the preprocessed data
            selected_technique,
            top_n=20,
            save_path=os.path.join(
                assets_dir,
                f"feature_coefficients_{selected_technique.lower().replace(' ', '_')}.png",
            ),
        )

        # Feature Distribution (Optional)
        if feature_to_plot:
            plot_feature_distribution(
                X,  # Use original scaled X before potential sampling in pipeline
                y,
                feature_to_plot,
                selected_technique,
                save_path=os.path.join(
                    assets_dir,
                    f"feature_dist_{feature_to_plot}_{selected_technique.lower().replace(' ', '_')}.png",
                ),
            )

        # --- Log selected pipeline ---
        # Save the fitted pipeline for the selected technique
        selected_pipeline_path = os.path.join(
            get_project_root(),
            "models",
            f"imbalance_pipeline_{selected_technique.lower().replace(' ', '_')}.pkl",
        )
        try:
            with open(selected_pipeline_path, "wb") as f:
                # Save pipeline and feature names together
                pipeline_data = {
                    "pipeline": selected_pipeline,
                    "features": X.columns.tolist(),  # Assuming X is the scaled feature DataFrame
                }
                pickle.dump(pipeline_data, f)
            logger.info(
                f"Saved selected pipeline '{selected_technique}' to {selected_pipeline_path}"
            )
            mlflow.log_artifact(
                selected_pipeline_path, artifact_path="selected_pipeline"
            )
        except Exception as e:
            logger.error(f"Failed to save selected pipeline: {e}", exc_info=True)

        # Log overall success status
        mlflow.log_param("overall_success", results["_meta"]["any_success"])

        logger.info("Imbalance analysis complete.")


def main() -> None:
    """
    Main function to run the imbalance analysis pipeline.

    Parses command-line arguments, loads and preprocesses data,
    and runs the analysis.
    """
    parser = argparse.ArgumentParser(description="Run imbalance handling analysis")
    # Add arguments if needed, e.g., to select specific techniques or config file
    # parser.add_argument(...)
    args = parser.parse_args()

    try:
        # Load data
        data = load_data()

        # Preprocess data
        X, y = preprocess_data(data)

        # Check if preprocessing was successful
        if X.empty or y.empty:
            logger.error("Preprocessing resulted in empty data. Aborting analysis.")
            return

        # Run analysis
        config = load_config()  # Load config again for analysis function
        analyze_imbalance_techniques(X, y, config)

    except FileNotFoundError:
        logger.error(
            "Required data file not found. Please ensure 'combined_features.csv' exists in the processed data directory."
        )
    except Exception as e:
        logger.error(
            f"An error occurred during the analysis pipeline: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
