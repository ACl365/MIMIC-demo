"""
Script to train models for the MIMIC project.
"""

import argparse
import os
import subprocess
from typing import Dict, Optional

import mlflow
import mlflow.sklearn  # Assuming models are scikit-learn compatible
import pandas as pd

from src.models.model import LengthOfStayModel, MortalityModel, ReadmissionModel
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


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


def train_readmission_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a readmission prediction model.

    Args:
        data (pd.DataFrame): Input data
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
        algorithm (Optional[str], optional): Algorithm to use. If None, uses the first
            algorithm in the configuration. Defaults to None.
        save_path (Optional[str], optional): Path to save the model to.
            If None, uses the default path. Defaults to None.

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info("Training readmission prediction model")
    git_hash = get_git_revision_hash()
    model_type = "readmission"
    run_name = f"{model_type}_{algorithm or 'default'}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algorithm or "default")
        mlflow.log_param("git_hash", git_hash)
        # Log relevant config parameters if needed
        # Example: mlflow.log_params(config['models']['readmission']['hyperparameters'])

        # Initialize model
        model = ReadmissionModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        # Assuming model.model is the trained sklearn estimator
        # and model.features is the list of features used.
        # Adjust artifact_path and registered_model_name as needed.
        if hasattr(model, "model") and hasattr(model, "features"):
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path=model_type,
                input_example=data[model.features].head(),  # Log input example
                # registered_model_name=f"{model_type}-model" # Optional: Register model
            )
            logger.info(f"Logged model artifact to MLflow path: {model_type}")
        else:
            logger.warning(
                "Model object does not have '.model' or '.features' attribute. Saving raw pickle."
            )
            # Fallback: Log the saved pickle file if sklearn logging fails
            if save_path is None:
                save_path = os.path.join(
                    get_project_root(), "models", "readmission_model.pkl"
                )
            model.save(save_path)  # Save locally first
            mlflow.log_artifact(save_path, artifact_path=model_type)
            logger.info(f"Logged model pickle artifact: {save_path}")

    return metrics


def train_mortality_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a mortality prediction model.

    Args:
        data (pd.DataFrame): Input data
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
        algorithm (Optional[str], optional): Algorithm to use. If None, uses the first
            algorithm in the configuration. Defaults to None.
        save_path (Optional[str], optional): Path to save the model to.
            If None, uses the default path. Defaults to None.

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info("Training mortality prediction model")
    git_hash = get_git_revision_hash()
    model_type = "mortality"
    run_name = f"{model_type}_{algorithm or 'default'}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algorithm or "default")
        mlflow.log_param("git_hash", git_hash)

        # Initialize model
        model = MortalityModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        if hasattr(model, "model") and hasattr(model, "features"):
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path=model_type,
                input_example=data[model.features].head(),
                # registered_model_name=f"{model_type}-model"
            )
            logger.info(f"Logged model artifact to MLflow path: {model_type}")
        else:
            logger.warning(
                "Model object does not have '.model' or '.features' attribute. Saving raw pickle."
            )
            if save_path is None:
                save_path = os.path.join(
                    get_project_root(), "models", "mortality_model.pkl"
                )
            model.save(save_path)
            mlflow.log_artifact(save_path, artifact_path=model_type)
            logger.info(f"Logged model pickle artifact: {save_path}")

    return metrics


def train_los_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a length of stay prediction model.

    Args:
        data (pd.DataFrame): Input data
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
        algorithm (Optional[str], optional): Algorithm to use. If None, uses the first
            algorithm in the configuration. Defaults to None.
        save_path (Optional[str], optional): Path to save the model to.
            If None, uses the default path. Defaults to None.

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info("Training length of stay prediction model")
    git_hash = get_git_revision_hash()
    model_type = "los"
    run_name = f"{model_type}_{algorithm or 'default'}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algorithm or "default")
        mlflow.log_param("git_hash", git_hash)

        # Initialize model
        model = LengthOfStayModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        if hasattr(model, "model") and hasattr(model, "features"):
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path=model_type,
                input_example=data[model.features].head(),
                # registered_model_name=f"{model_type}-model"
            )
            logger.info(f"Logged model artifact to MLflow path: {model_type}")
        else:
            logger.warning(
                "Model object does not have '.model' or '.features' attribute. Saving raw pickle."
            )
            if save_path is None:
                save_path = os.path.join(get_project_root(), "models", "los_model.pkl")
            model.save(save_path)
            mlflow.log_artifact(save_path, artifact_path=model_type)
            logger.info(f"Logged model pickle artifact: {save_path}")

    return metrics


def train_models(
    config: Optional[Dict] = None,
    model_type: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Train models for the MIMIC project.

    Args:
        config (Optional[Dict], optional): Configuration dictionary.
            If None, loads the default configuration. Defaults to None.
        model_type (Optional[str], optional): Type of model to train.
            If None, trains all models. Defaults to None.
        algorithm (Optional[str], optional): Algorithm to use.
            If None, uses the first algorithm in the configuration. Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: Evaluation metrics for each model
    """
    if config is None:
        config = load_config()

    # Load data
    logger.info("Loading data")
    data_path = get_data_path("processed", "combined_features", config)
    data = pd.read_csv(data_path)

    # MLflow will handle model artifact storage, so explicit directory creation might be redundant
    # models_dir = os.path.join(get_project_root(), "models")
    # os.makedirs(models_dir, exist_ok=True)
    # Set MLflow experiment
    experiment_name = config.get("mlflow", {}).get("experiment_name", "MIMIC_Training")
    mlflow.set_experiment(experiment_name)
    logger.info(f"Using MLflow experiment: {experiment_name}")

    # Train models
    metrics = {}

    if model_type is None or model_type == "readmission":
        metrics["readmission"] = train_readmission_model(data, config, algorithm)

    if model_type is None or model_type == "mortality":
        metrics["mortality"] = train_mortality_model(data, config, algorithm)

    if model_type is None or model_type == "los":
        metrics["los"] = train_los_model(data, config, algorithm)

    return metrics


def main() -> None:
    """
    Main function to train models.
    """
    parser = argparse.ArgumentParser(description="Train models for the MIMIC project")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["readmission", "mortality", "los"],
        default=None,
        help="Type of model to train",
    )
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm to use")
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Train models
    metrics = train_models(config, args.model, args.algorithm)

    # Print metrics
    for model_type, model_metrics in metrics.items():
        logger.info(f"{model_type.capitalize()} model metrics:")
        for metric_name, metric_value in model_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
