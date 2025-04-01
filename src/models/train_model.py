"""
Script to train models for the MIMIC project.
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from src.utils import get_logger, load_config, get_data_path, get_project_root
from src.models.model import ReadmissionModel, MortalityModel, LengthOfStayModel


logger = get_logger(__name__)


def train_readmission_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None
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
    
    # Initialize model
    model = ReadmissionModel(config=config)
    
    # Train model
    metrics = model.fit(data, algorithm=algorithm)
    
    # Save model
    if save_path is None:
        save_path = os.path.join(
            get_project_root(), 
            "models", 
            "readmission_model.pkl"
        )
    
    model.save(save_path)
    
    return metrics


def train_mortality_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None
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
    
    # Initialize model
    model = MortalityModel(config=config)
    
    # Train model
    metrics = model.fit(data, algorithm=algorithm)
    
    # Save model
    if save_path is None:
        save_path = os.path.join(
            get_project_root(), 
            "models", 
            "mortality_model.pkl"
        )
    
    model.save(save_path)
    
    return metrics


def train_los_model(
    data: pd.DataFrame,
    config: Optional[Dict] = None,
    algorithm: Optional[str] = None,
    save_path: Optional[str] = None
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
    
    # Initialize model
    model = LengthOfStayModel(config=config)
    
    # Train model
    metrics = model.fit(data, algorithm=algorithm)
    
    # Save model
    if save_path is None:
        save_path = os.path.join(
            get_project_root(), 
            "models", 
            "los_model.pkl"
        )
    
    model.save(save_path)
    
    return metrics


def train_models(
    config: Optional[Dict] = None,
    model_type: Optional[str] = None,
    algorithm: Optional[str] = None
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
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(get_project_root(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Train models
    metrics = {}
    
    if model_type is None or model_type == "readmission":
        metrics["readmission"] = train_readmission_model(
            data, config, algorithm
        )
    
    if model_type is None or model_type == "mortality":
        metrics["mortality"] = train_mortality_model(
            data, config, algorithm
        )
    
    if model_type is None or model_type == "los":
        metrics["los"] = train_los_model(
            data, config, algorithm
        )
    
    return metrics


def main() -> None:
    """
    Main function to train models.
    """
    parser = argparse.ArgumentParser(description="Train models for the MIMIC project")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["readmission", "mortality", "los"], 
        default=None, 
        help="Type of model to train"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default=None, 
        help="Algorithm to use"
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
    for model_type, model_metrics in metrics.items():
        logger.info(f"{model_type.capitalize()} model metrics:")
        for metric_name, metric_value in model_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()