"""
Configuration utilities for the MIMIC project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from .logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Define expected configuration structure
CONFIG_SCHEMA = {
    "logging": {"level", "file_output", "console_output"},
    "data": {"raw", "processed", "external"},
    "features": {"demographic", "vitals", "labs", "medications", "procedures", "diagnoses", "temporal"},
    "models": {"readmission", "mortality", "los"},
    "evaluation": {"classification", "regression"},
    "api": {"host", "port", "debug"},
    "dashboard": {"host", "port", "debug"}
}

# Define expected mappings structure
MAPPINGS_SCHEMA = {
    "lab_tests": {"common_labs", "mappings"},
    "vital_signs": {"categories", "itemids"},
    "icd9_categories": {"ranges", "specific_codes"}
}


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root directory
    """
    # This assumes the script is in src/utils/config.py
    return Path(__file__).parent.parent.parent


def validate_config_structure(config: Dict[str, Any], schema: Dict[str, set], path: str = "") -> List[str]:
    """
    Validate the structure of a configuration dictionary against a schema.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        schema (Dict[str, set]): Schema dictionary with expected keys
        path (str, optional): Current path in the configuration for error messages
    
    Returns:
        List[str]: List of validation errors, empty if valid
    """
    errors = []
    
    # Check for missing required top-level sections
    for section in schema:
        if section not in config:
            errors.append(f"Missing required section '{section}' in configuration{' at ' + path if path else ''}")
    
    # Check each section that exists
    for section, value in config.items():
        if section in schema:
            # If schema defines subsections and value is a dictionary, validate recursively
            if isinstance(schema[section], set) and isinstance(value, dict):
                # Check for missing required subsections
                for subsection in schema[section]:
                    if subsection not in value:
                        errors.append(f"Missing required subsection '{subsection}' in '{path + '.' if path else ''}{section}'")
            
            # If schema defines a dictionary structure but value is not a dictionary
            elif isinstance(schema[section], set) and not isinstance(value, dict):
                errors.append(f"Section '{path + '.' if path else ''}{section}' should be a dictionary")
    
    return errors

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, uses the default config.yaml in the configs directory.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file is malformed or missing required sections
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
        
        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be a dictionary, got {type(config).__name__}")
        
        # Validate configuration structure
        errors = validate_config_structure(config, CONFIG_SCHEMA)
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {e}" for e in errors)
            logger.warning(error_msg)
            # Don't raise an exception, just log warnings
        
        return config
    
    except (ParserError, ScannerError) as e:
        error_msg = f"YAML syntax error in configuration file {config_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Error loading configuration from {config_path}: {str(e)}"
        logger.error(error_msg)
        raise


def load_mappings() -> Dict[str, Any]:
    """
    Load clinical feature mappings from the mappings.yaml file.
    
    Returns:
        Dict[str, Any]: Mappings dictionary containing lab tests, vital signs, and ICD-9 categories
    
    Raises:
        FileNotFoundError: If the mappings file does not exist
        ValueError: If the mappings file is malformed or missing required sections
    """
    mappings_path = os.path.join(get_project_root(), "configs", "mappings.yaml")
    
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
    
    try:
        with open(mappings_path, "r") as f:
            mappings = yaml.safe_load(f)
        
        if mappings is None:
            raise ValueError(f"Mappings file is empty: {mappings_path}")
        
        if not isinstance(mappings, dict):
            raise ValueError(f"Mappings must be a dictionary, got {type(mappings).__name__}")
        
        # Validate mappings structure
        errors = validate_config_structure(mappings, MAPPINGS_SCHEMA)
        if errors:
            error_msg = "Mappings validation errors:\n" + "\n".join(f"- {e}" for e in errors)
            logger.warning(error_msg)
            # Don't raise an exception, just log warnings
        
        return mappings
    
    except (ParserError, ScannerError) as e:
        error_msg = f"YAML syntax error in mappings file {mappings_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Error loading mappings from {mappings_path}: {str(e)}"
        logger.error(error_msg)
        raise


def get_data_path(data_type: str, dataset: str = None, config: Dict[str, Any] = None) -> str:
    """
    Get the path to a data file or directory.
    
    Args:
        data_type (str): Type of data ('raw', 'processed', or 'external')
        dataset (str, optional): Specific dataset (e.g., 'mimic_iii', 'patient_data')
        config (Dict[str, Any], optional): Configuration dictionary.
            If None, loads the default configuration.
    
    Returns:
        str: Path to the data file or directory
    
    Raises:
        ValueError: If data_type is not valid
        KeyError: If the requested dataset is not found in the configuration
    """
    if config is None:
        config = load_config()
    
    valid_data_types = ["raw", "processed", "external"]
    if data_type not in valid_data_types:
        raise ValueError(f"data_type must be one of {valid_data_types}, got '{data_type}'")
    
    # Ensure data section exists
    if "data" not in config:
        raise KeyError("'data' section not found in configuration")
    
    # Ensure data type section exists
    if data_type not in config["data"]:
        raise KeyError(f"'{data_type}' section not found in data configuration")
    
    # Get the base path for the data type
    if dataset is None:
        try:
            return config["data"][data_type]["base_path"]
        except KeyError:
            raise KeyError(f"'base_path' not found in '{data_type}' data configuration")
    
    # Get the path for the specific dataset
    try:
        path = config["data"][data_type][dataset]
    except KeyError:
        raise KeyError(f"Dataset '{dataset}' not found in configuration for {data_type} data")
    
    # Convert relative paths to absolute paths
    if isinstance(path, str) and path.startswith(".."):
        path = os.path.join(get_project_root(), path)
    
    return path


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        config_path (Optional[str], optional): Path to save the configuration file.
            If None, uses the default config.yaml in the configs directory.
    
    Raises:
        ValueError: If the configuration is not valid
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs", "config.yaml")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Validate configuration
    errors = validate_config_structure(config, CONFIG_SCHEMA)
    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(f"- {e}" for e in errors)
        logger.warning(error_msg)
    
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        error_msg = f"Error saving configuration to {config_path}: {str(e)}"
        logger.error(error_msg)
        raise