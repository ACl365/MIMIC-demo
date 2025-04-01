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

All metrics are computed using cross-validation to ensure robust evaluation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from src.utils import get_logger, load_config, get_data_path, get_project_root
from src.models.model import ReadmissionModel


logger = get_logger(__name__)


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
    class_counts = y.value_counts()
    total = len(y)
    logger.info(f"Class distribution:")
    for cls, count in class_counts.items():
        logger.info(f"  Class {cls}: {count} ({count/total:.2%})")
    
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
    base_clf = LogisticRegression(max_iter=1000, random_state=random_state)
    
    # Create pipelines
    pipelines = {
        "Baseline": Pipeline([
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        
        "Class Weights": Pipeline([
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
        ]),
        
        "Random Oversampling": Pipeline([
            ("sampler", RandomOverSampler(random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        
        "SMOTE": Pipeline([
            ("sampler", SMOTE(random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        
        "Random Undersampling": Pipeline([
            ("sampler", RandomUnderSampler(random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ])
    }
    
    return pipelines


def evaluate_pipelines(
    X: pd.DataFrame, 
    y: pd.Series, 
    pipelines: Dict[str, Any],
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate the pipelines using cross-validation.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        pipelines (Dict[str, Any]): Dictionary of pipelines
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of evaluation results
    """
    results = {}
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
            "success": False  # Track success for each pipeline
        }
        
        # Get cross-validated predictions and probabilities
        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=cv)
            y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
            
            # Calculate metrics
            results[name]["y_pred"] = y_pred
            results[name]["y_prob"] = y_prob
            results[name]["precision"] = precision_score(y, y_pred)
            results[name]["recall"] = recall_score(y, y_pred)
            results[name]["f1"] = f1_score(y, y_pred)
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
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
            continue
    
    # Add a flag to indicate if any pipeline succeeded
    results["_meta"] = {"any_success": any_success}
    
    if not any_success:
        logger.warning("All pipelines failed evaluation. Check for data issues or model compatibility.")
    
    return results


def plot_pr_curves(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
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
        ax.text(0.5, 0.5, "No successful pipelines to compare.\nCheck logs for error details.",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty PR curve plot to {save_path}")
        else:
            plt.show()
        
        return
    
    plt.figure(figsize=(10, 8))
    
    # Flag to track if any curves were plotted
    any_curves_plotted = False
    
    for name, result in results.items():
        # Skip the _meta entry
        if name == "_meta":
            continue
            
        if result.get("success", False) and result["precision_curve"] is not None and result["recall_curve"] is not None:
            plt.plot(
                result["recall_curve"],
                result["precision_curve"],
                label=f"{name} (PR AUC = {result['pr_auc']:.3f})"
            )
            any_curves_plotted = True
    
    if not any_curves_plotted:
        logger.warning("No PR curves to plot. All pipelines failed or had missing data.")
        plt.clf()  # Clear the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No PR curves to plot.\nAll pipelines failed or had missing data.",
                ha='center', va='center', fontsize=14)
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


def plot_metrics_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
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
        ax.text(0.5, 0.5, "No successful pipelines to compare.\nCheck logs for error details.",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty metrics comparison plot to {save_path}")
        else:
            plt.show()
        
        return
    
    # Filter out the _meta entry and any failed pipelines
    pipeline_names = [name for name in results.keys()
                     if name != "_meta" and results[name].get("success", False)]
    
    if not pipeline_names:
        logger.warning("No successful pipelines to plot metrics for.")
        
        # Create a simple plot with a message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No successful pipelines to compare.\nCheck logs for error details.",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved empty metrics comparison plot to {save_path}")
        else:
            plt.show()
        
        return
    
    metrics = ["precision", "recall", "f1", "pr_auc"]
    
    # Extract metrics for each pipeline
    metric_values = {metric: [results[name][metric] for name in pipeline_names] for metric in metrics}
    
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
    ax.set_xticks(x + width, pipeline_names)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics comparison plot to {save_path}")
    else:
        plt.show()


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
        df = pd.DataFrame([{
            "Technique": "No successful pipelines",
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "PR AUC": None,
            "Note": "All pipelines failed. Check logs for error details."
        }])
        df.to_csv(save_path, index=False)
        logger.info(f"Saved empty results to {save_path}")
        return
    
    # Extract metrics for each pipeline
    data = []
    for name, result in results.items():
        # Skip the _meta entry
        if name == "_meta":
            continue
            
        # Only include successful pipelines
        if result.get("success", False):
            data.append({
                "Technique": name,
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1"],
                "PR AUC": result["pr_auc"]
            })
    
    if not data:
        logger.warning("No successful pipelines to save results for.")
        
        # Create a simple CSV with a message
        df = pd.DataFrame([{
            "Technique": "No successful pipelines",
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "PR AUC": None,
            "Note": "All pipelines failed. Check logs for error details."
        }])
    else:
        # Create dataframe from successful pipeline data
        df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Saved results to {save_path}")


def analyze_imbalance_techniques(
    output_dir: Optional[str] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Analyse different imbalance handling techniques.
    
    Args:
        output_dir (Optional[str], optional): Directory to save outputs.
            If None, uses the default directory. Defaults to None.
        cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of evaluation results
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "results", "imbalance_analysis")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    logger.info("Note: The data will be restructured to ensure each row represents a unique hospital admission")
    data = load_data()
    X, y = preprocess_data(data)
    
    # Create pipelines
    pipelines = create_imbalance_pipelines(random_state)
    
    # Evaluate pipelines
    results = evaluate_pipelines(X, y, pipelines, cv_folds, random_state)
    
    # Plot results
    pr_curve_path = os.path.join(output_dir, "pr_curves.png")
    plot_pr_curves(results, pr_curve_path)
    
    metrics_path = os.path.join(output_dir, "metrics_comparison.png")
    plot_metrics_comparison(results, metrics_path)
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, "imbalance_results.csv")
    save_results_to_csv(results, csv_path)
    
    # Print summary
    logger.info("\nSummary of Results:")
    for name, result in results.items():
        # Skip the _meta entry
        if name == "_meta":
            continue
            
        logger.info(f"{name}:")
        # Only print metrics if they exist
        if result.get("precision") is not None:
            logger.info(f"  Precision: {result['precision']:.4f}")
        if result.get("recall") is not None:
            logger.info(f"  Recall: {result['recall']:.4f}")
        if result.get("f1") is not None:
            logger.info(f"  F1: {result['f1']:.4f}")
        if result.get("pr_auc") is not None:
            logger.info(f"  PR AUC: {result['pr_auc']:.4f}")
    
    # Print discussion
    logger.info("\nDiscussion of Trade-offs:")
    logger.info("1. Baseline vs. Class Weights:")
    logger.info("   - Class weights adjust the importance of each class during training, which can help")
    logger.info("     the model pay more attention to the minority class without changing the data.")
    logger.info("   - This typically improves recall at the expense of precision.")
    
    logger.info("\n2. Random Oversampling vs. SMOTE:")
    logger.info("   - Random oversampling duplicates existing minority samples, which can lead to overfitting")
    logger.info("     as the model sees the exact same minority samples multiple times.")
    logger.info("   - SMOTE creates synthetic examples by interpolating between existing minority samples,")
    logger.info("     which can help the model generalize better by learning from a more diverse set of")
    logger.info("     minority class examples.")
    logger.info("   - SMOTE may perform differently than random oversampling because it creates new,")
    logger.info("     synthetic samples rather than just duplicating existing ones, potentially leading")
    logger.info("     to better generalization but possibly introducing noise if the synthetic samples")
    logger.info("     are not representative of the true data distribution.")
    
    logger.info("\n3. Oversampling vs. Undersampling:")
    logger.info("   - Oversampling techniques (Random Oversampling, SMOTE) increase the number of minority")
    logger.info("     class samples to balance the classes, preserving all available information but")
    logger.info("     potentially leading to longer training times and overfitting.")
    logger.info("   - Undersampling reduces the number of majority class samples, which can lead to")
    logger.info("     information loss but may help prevent the model from being biased towards the")
    logger.info("     majority class and can reduce training time.")
    
    logger.info("\nNote on Small Dataset Size:")
    logger.info("With only ~200 demo patients, the absolute performance metrics may be unstable and")
    logger.info("not generalizable. The relative differences between techniques are more informative")
    logger.info("than the absolute values. The goal is to demonstrate methodological understanding")
    logger.info("and implementation skills rather than achieving high predictive accuracy.")
    
    return results


def main() -> None:
    """
    Main function to run the imbalance analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyse different imbalance handling techniques for the readmission model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    args = parser.parse_args()
    
    # Run analysis
    analyze_imbalance_techniques(
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()