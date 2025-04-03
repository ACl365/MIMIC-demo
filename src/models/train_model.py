"""
Script to train models for the MIMIC project.

This script handles the training process for different prediction tasks
(readmission, mortality, length of stay) based on a configuration file.
It loads the combined features, initializes the appropriate model class
from `src.models.model`, trains the model using the specified algorithm(s),
evaluates the model, and logs parameters, metrics, and model artifacts
using MLflow.
"""

import argparse
import os
import subprocess
from typing import Any, Dict, Optional  # Added Any

import mlflow
import mlflow.sklearn  # Assuming models are scikit-learn compatible
import pandas as pd

from src.models.model import TemporalReadmissionModel  # <-- Add import
from src.models.model import LengthOfStayModel, MortalityModel, ReadmissionModel
from src.utils import get_data_path, get_logger, get_project_root, load_config

logger = get_logger(__name__)


def get_git_revision_hash() -> str:
    """
    Gets the current git commit hash of the repository.

    Returns:
        str: The git commit hash as a string, or 'unknown' if git command fails.
    """
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
    config: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,  # Keep save_path for potential fallback saving
) -> Dict[str, float]:
    """
    Train, evaluate, and log a readmission prediction model using MLflow.

    Initializes a ReadmissionModel, fits it to the data using the specified
    algorithm (or the default from config), logs parameters and metrics to MLflow,
    and logs the trained model as an MLflow artifact.

    Args:
        data (pd.DataFrame): Input DataFrame containing features and target.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        algorithm (Optional[str], optional): Specific algorithm to use (e.g., 'logistic_regression').
                                             If None, uses the default specified in the config. Defaults to None.
        save_path (Optional[str], optional): Local path to save the model pickle as a fallback
                                             if MLflow sklearn logging fails. Defaults to None,
                                             using a standard path in the 'models/' directory.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics from the trained model.
    """
    logger.info("Training readmission prediction model")
    if config is None:
        config = load_config()  # Load default if not provided

    git_hash = get_git_revision_hash()
    model_type = "readmission"
    algo_name = (
        algorithm or config["models"][model_type].get("algorithms", ["default"])[0]
    )
    run_name = f"{model_type}_{algo_name}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("git_hash", git_hash)
        # Log relevant config parameters
        if model_type in config.get("models", {}):
            mlflow.log_params(config["models"][model_type])

        # Initialize model
        model = ReadmissionModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)  # Pass algorithm explicitly
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        # Use context manager for safer file handling if saving locally
        if model.model is not None and model.feature_names is not None:
            try:
                # Prepare input example safely
                input_example_df = (
                    data[model.feature_names] if model.feature_names else data
                )
                input_example = (
                    input_example_df.head() if not input_example_df.empty else None
                )

                mlflow.sklearn.log_model(
                    sk_model=model.model,
                    artifact_path=model_type,
                    input_example=input_example,
                    # registered_model_name=f"{model_type}-model" # Optional: Register model
                )
                logger.info(f"Logged model artifact to MLflow path: {model_type}")
            except Exception as e:
                logger.error(
                    f"Failed to log model with mlflow.sklearn: {e}", exc_info=True
                )
                logger.warning("Attempting to save raw pickle as fallback.")
                # Fallback: Log the saved pickle file
                if save_path is None:
                    save_path = os.path.join(
                        get_project_root(),
                        "models",
                        f"{model_type}_model_{run_id[:8]}.pkl",  # Include run_id part
                    )
                try:
                    model.save(save_path)  # Save locally first
                    mlflow.log_artifact(save_path, artifact_path=model_type)
                    logger.info(f"Logged model pickle artifact: {save_path}")
                except Exception as save_e:
                    logger.error(
                        f"Failed to save model pickle locally: {save_e}", exc_info=True
                    )

        else:
            logger.warning(
                "Trained model object or feature names not found. Cannot log model artifact."
            )

    return metrics


def train_mortality_model(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train, evaluate, and log a mortality prediction model using MLflow.

    Initializes a MortalityModel, fits it to the data using the specified
    algorithm (or the default from config), logs parameters and metrics to MLflow,
    and logs the trained model as an MLflow artifact.

    Args:
        data (pd.DataFrame): Input DataFrame containing features and target.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        algorithm (Optional[str], optional): Specific algorithm to use. Defaults to None.
        save_path (Optional[str], optional): Local path to save the model pickle as a fallback.
                                             Defaults to None.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics from the trained model.
    """
    logger.info("Training mortality prediction model")
    if config is None:
        config = load_config()

    git_hash = get_git_revision_hash()
    model_type = "mortality"
    algo_name = (
        algorithm or config["models"][model_type].get("algorithms", ["default"])[0]
    )
    run_name = f"{model_type}_{algo_name}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("git_hash", git_hash)
        if model_type in config.get("models", {}):
            mlflow.log_params(config["models"][model_type])

        # Initialize model
        model = MortalityModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        if model.model is not None and model.feature_names is not None:
            try:
                input_example_df = (
                    data[model.feature_names] if model.feature_names else data
                )
                input_example = (
                    input_example_df.head() if not input_example_df.empty else None
                )
                mlflow.sklearn.log_model(
                    sk_model=model.model,
                    artifact_path=model_type,
                    input_example=input_example,
                    # registered_model_name=f"{model_type}-model"
                )
                logger.info(f"Logged model artifact to MLflow path: {model_type}")
            except Exception as e:
                logger.error(
                    f"Failed to log model with mlflow.sklearn: {e}", exc_info=True
                )
                logger.warning("Attempting to save raw pickle as fallback.")
                if save_path is None:
                    save_path = os.path.join(
                        get_project_root(),
                        "models",
                        f"{model_type}_model_{run_id[:8]}.pkl",
                    )
                try:
                    model.save(save_path)
                    mlflow.log_artifact(save_path, artifact_path=model_type)
                    logger.info(f"Logged model pickle artifact: {save_path}")
                except Exception as save_e:
                    logger.error(
                        f"Failed to save model pickle locally: {save_e}", exc_info=True
                    )
        else:
            logger.warning(
                "Trained model object or feature names not found. Cannot log model artifact."
            )

    return metrics


def train_los_model(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train, evaluate, and log a length of stay (LOS) prediction model using MLflow.

    Initializes a LengthOfStayModel, fits it to the data using the specified
    regression algorithm (or the default from config), logs parameters and metrics
    to MLflow, and logs the trained model as an MLflow artifact.

    Args:
        data (pd.DataFrame): Input DataFrame containing features and target.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        algorithm (Optional[str], optional): Specific regression algorithm to use. Defaults to None.
        save_path (Optional[str], optional): Local path to save the model pickle as a fallback.
                                             Defaults to None.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics (regression metrics) from the trained model.
    """
    logger.info("Training length of stay prediction model")
    if config is None:
        config = load_config()

    git_hash = get_git_revision_hash()
    model_type = "los"
    algo_name = (
        algorithm or config["models"][model_type].get("algorithms", ["default"])[0]
    )
    run_name = f"{model_type}_{algo_name}_{git_hash[:7]}"

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("git_hash", git_hash)
        if model_type in config.get("models", {}):
            mlflow.log_params(config["models"][model_type])

        # Initialize model
        model = LengthOfStayModel(config=config)

        # Train model
        metrics = model.fit(data, algorithm=algorithm)
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact
        if model.model is not None and model.feature_names is not None:
            try:
                input_example_df = (
                    data[model.feature_names] if model.feature_names else data
                )
                input_example = (
                    input_example_df.head() if not input_example_df.empty else None
                )
                mlflow.sklearn.log_model(
                    sk_model=model.model,
                    artifact_path=model_type,
                    input_example=input_example,
                    # registered_model_name=f"{model_type}-model"
                )
                logger.info(f"Logged model artifact to MLflow path: {model_type}")
            except Exception as e:
                logger.error(
                    f"Failed to log model with mlflow.sklearn: {e}", exc_info=True
                )
                logger.warning("Attempting to save raw pickle as fallback.")
                if save_path is None:
                    save_path = os.path.join(
                        get_project_root(),
                        "models",
                        f"{model_type}_model_{run_id[:8]}.pkl",
                    )
                try:
                    model.save(save_path)
                    mlflow.log_artifact(save_path, artifact_path=model_type)
                    logger.info(f"Logged model pickle artifact: {save_path}")
                except Exception as save_e:
                    logger.error(
                        f"Failed to save model pickle locally: {save_e}", exc_info=True
                    )
        else:
            logger.warning(
                "Trained model object or feature names not found. Cannot log model artifact."
            )

    return metrics


# --- Add function for Temporal Model ---


def train_temporal_readmission_model(
    data: pd.DataFrame,  # Note: Temporal model preprocesses differently inside fit
    config: Optional[Dict[str, Any]] = None,
    # Algorithm selection might be handled internally by TemporalReadmissionModel based on config
    # algorithm: Optional[str] = None,
    save_path: Optional[str] = None,  # Keep for potential fallback saving if needed
) -> Dict[str, float]:
    """
    Train, evaluate, and log a temporal readmission prediction model using MLflow.

    Initializes a TemporalReadmissionModel, fits it to the data (which includes
    its own preprocessing, training loop, and evaluation), logs parameters and metrics
    to MLflow, and handles model artifact logging (potentially within the model's save method).

    Args:
        data (pd.DataFrame): Input DataFrame containing features and target (will be preprocessed internally).
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        save_path (Optional[str], optional): Base path for saving model artifacts if needed.
                                             Defaults to None.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics from the trained model.
    """
    logger.info("Training temporal readmission prediction model")
    if config is None:
        config = load_config()  # Load default if not provided

    git_hash = get_git_revision_hash()
    model_type = "temporal_readmission"
    # Algorithm/hyperparameters are likely defined within the model's config section
    # algo_name = algorithm or config['models'][model_type].get('algorithms', ['default'])[0]
    run_name = f"{model_type}_{git_hash[:7]}"  # Simplified name as specific algo isn't passed here

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", model_type)
        # mlflow.log_param("algorithm", algo_name) # Algorithm managed internally
        mlflow.log_param("git_hash", git_hash)
        # Log relevant config parameters for the temporal model
        if model_type in config.get("models", {}):
            mlflow.log_params(config["models"][model_type])

        # Initialize model
        # The TemporalReadmissionModel handles its own device placement, optimizer, etc.
        model = TemporalReadmissionModel(config=config)

        # Train model - TemporalReadmissionModel.fit handles the full loop
        # It needs the raw data to perform its specific preprocessing
        # The 'algorithm' parameter might not be applicable here if the model uses a fixed architecture (LSTM)
        # and hyperparameters are set via config.
        metrics = model.fit(data)  # Call fit without algorithm, assumes it uses config
        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        # Log model artifact - TemporalReadmissionModel should handle saving via its save method
        # We might log the path or let the model log directly if it takes mlflow_client
        # For now, assume the model's save method is sufficient or logs separately.
        # If direct MLflow logging is needed here, model.save() would need to return the path
        # or we'd need access to the trained PyTorch model object.
        # Example: Log the config used.
        # config_path = os.path.join(get_project_root(), "configs", "config.yaml") # Or the specific one used
        # mlflow.log_artifact(config_path, artifact_path="config")

        # Fallback saving (similar to other models, if needed)
        if model.model is not None:  # Check if the internal PyTorch model exists
            try:
                # Standard PyTorch saving is usually handled by model.save()
                # If we want MLflow to track it explicitly:
                if save_path is None:
                    save_path = os.path.join(
                        get_project_root(),
                        "models",
                        f"{model_type}_model_{run_id[:8]}",  # Dir for PyTorch model files
                    )
                model.save(save_path)  # Save locally first (might save multiple files)
                mlflow.log_artifacts(
                    save_path, artifact_path=model_type
                )  # Log the directory
                logger.info(
                    f"Logged model artifacts to MLflow path: {model_type} from dir: {save_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save or log temporal model artifact: {e}", exc_info=True
                )
        else:
            logger.warning(
                "Trained temporal model object not found. Cannot log model artifact via train script."
            )

    return metrics


def train_models(
    config: Optional[Dict[str, Any]] = None,
    model_type: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Train one or all models based on the configuration and arguments.

    Loads the combined feature data and calls the appropriate training function
    (readmission, mortality, los) based on the `model_type` argument. If `model_type`
    is None, all models specified in the config are trained. Logs results using MLflow.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration dictionary. Defaults to None.
        model_type (Optional[str], optional): Specific model type to train ('readmission',
                                             'mortality', 'los'). If None, trains all.
                                             Defaults to None.
        algorithm (Optional[str], optional): Specific algorithm to use for the selected model type.
                                             If None, uses defaults from config. Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are model types and values
                                     are dictionaries of their evaluation metrics.
    """
    if config is None:
        config = load_config()

    # Load data
    logger.info("Loading combined features data")
    data_path = get_data_path("processed", "combined_features", config)
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}. Shape: {data.shape}")
    except FileNotFoundError:
        logger.error(
            f"Combined features file not found at {data_path}. Cannot train models."
        )
        return {}
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}", exc_info=True)
        return {}

    # Set MLflow experiment
    try:
        experiment_name = config.get("mlflow", {}).get(
            "experiment_name", "MIMIC_Training"
        )
        mlflow.set_experiment(experiment_name)
        logger.info(f"Using MLflow experiment: {experiment_name}")
    except Exception as e:
        logger.error(
            f"Failed to set MLflow experiment '{experiment_name}': {e}", exc_info=True
        )
        # Decide if training should proceed without MLflow or stop
        logger.warning(
            "Proceeding with training without MLflow tracking due to setup error."
        )

    # Train models
    metrics: Dict[str, Dict[str, float]] = {}
    models_to_train = config.get("models", {})

    if model_type:  # Train only specified model type
        if model_type in models_to_train:
            logger.info(f"Training only the '{model_type}' model.")
            if model_type == "readmission":
                metrics["readmission"] = train_readmission_model(
                    data, config, algorithm
                )
            elif model_type == "mortality":
                metrics["mortality"] = train_mortality_model(data, config, algorithm)
            elif model_type == "los":
                metrics["los"] = train_los_model(data, config, algorithm)
            elif model_type == "temporal_readmission":  # <-- Add temporal model case
                # Note: algorithm param might not be needed if handled by config in TemporalReadmissionModel
                metrics["temporal_readmission"] = train_temporal_readmission_model(
                    data, config
                )
            # Add elif for other model types like 'temporal_readmission' if needed
            else:
                logger.warning(
                    f"Training function for model type '{model_type}' not implemented."
                )
        else:
            logger.error(
                f"Specified model type '{model_type}' not found in configuration."
            )
    else:  # Train all models defined in config
        logger.info("Training all models specified in configuration.")
        if "readmission" in models_to_train:
            metrics["readmission"] = train_readmission_model(data, config, algorithm)
        if "mortality" in models_to_train:
            metrics["mortality"] = train_mortality_model(data, config, algorithm)
        if "los" in models_to_train:
            metrics["los"] = train_los_model(data, config, algorithm)
        if "temporal_readmission" in models_to_train:  # <-- Add temporal model case
            metrics["temporal_readmission"] = train_temporal_readmission_model(
                data, config
            )
        # Add other models here

    return metrics


def main() -> None:
    """
    Main execution function for the script.

    Parses command-line arguments for optional config path, specific model type,
    and algorithm, then loads data and initiates the training process via `train_models`.
    Finally, prints the evaluation metrics for the trained models.
    """
    parser = argparse.ArgumentParser(description="Train models for the MIMIC project")
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
            "temporal_readmission",
        ],  # <-- Add temporal choice
        default=None,
        help="Type of model to train (default: all)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Specific algorithm to use (optional)",
    )
    args = parser.parse_args()

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()

    # Train models
    metrics = train_models(config, args.model, args.algorithm)

    # Print metrics
    if metrics:
        logger.info("--- Training Summary ---")
        for model_type, model_metrics in metrics.items():
            logger.info(f"{model_type.capitalize()} model metrics:")
            for metric_name, metric_value in model_metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
        logger.info("--- End Summary ---")
    else:
        logger.warning("No models were trained or metrics generated.")


if __name__ == "__main__":
    main()
