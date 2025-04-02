"""
Configuration utilities for the MIMIC project.
"""

import os
from functools import lru_cache  # Import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional  # Removed json, Union

import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

# We will import get_logger inside functions to avoid circular import
# from .logger import get_logger
# logger = get_logger(__name__)

# Define expected configuration structure
CONFIG_SCHEMA = {
    "logging": {"level", "file_output", "console_output"},
    "data": {"raw", "processed", "external"},
    "features": {
        "demographic",
        "vitals",
        "labs",
        "medications",
        "procedures",
        "diagnoses",
        "temporal",
    },
    "models": {"readmission", "mortality", "los"},
    "evaluation": {"classification", "regression"},
    "api": {"host", "port", "debug"},
    "dashboard": {"host", "port", "debug"},
}

# Define expected mappings structure
MAPPINGS_SCHEMA = {
    "lab_tests": {"common_labs", "mappings"},
    "vital_signs": {"categories", "itemids"},
    "icd9_categories": {"ranges", "specific_codes"},
}


# Define project root at the module level, assuming standard structure
# This avoids calling Path(__file__) inside functions during import cycles
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)  # Resolve ensures absolute path


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: Path to the project root directory
    """
    return PROJECT_ROOT  # Return the pre-calculated path


def validate_config_structure(
    config: Dict[str, Any], schema: Dict[str, set], path: str = ""
) -> List[str]:
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
            errors.append(
                f"Missing required section '{section}' in configuration{' at ' + path if path else ''}"
            )

    # Check each section that exists
    for section, value in config.items():
        if section in schema:
            # If schema defines subsections and value is a dictionary, validate recursively
            if isinstance(schema[section], set) and isinstance(value, dict):
                # Check for missing required subsections
                for subsection in schema[section]:
                    if subsection not in value:
                        errors.append(
                            f"Missing required subsection '{subsection}' in '{path + '.' if path else ''}{section}'"
                        )

            # If schema defines a dictionary structure but value is not a dictionary
            elif isinstance(schema[section], set) and not isinstance(value, dict):
                errors.append(
                    f"Section '{path + '.' if path else ''}{section}' should be a dictionary"
                )

    return errors


@lru_cache(maxsize=None)  # Cache the result of load_config
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
    # Determine the correct path object
    if config_path is None:
        config_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)

    # Use the object for exists check and opening, convert to string for messages
    config_path_str = str(config_path_obj)

    if not config_path_obj.exists():
        # Cannot use logger here reliably due to potential recursion
        print(f"ERROR: Configuration file not found: {config_path_str}")
        raise FileNotFoundError(f"Configuration file not found: {config_path_str}")

    try:  # Ensure we open the Path object
        with config_path_obj.open("r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path_str}")
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration must be a dictionary, got {type(config).__name__}"
            )

        # Validate configuration structure
        errors = validate_config_structure(config, CONFIG_SCHEMA)
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            # Cannot use logger here reliably due to potential recursion
            print(
                f"WARNING: Config validation: {error_msg}"
            )  # Print warning instead of logging
            # Don't raise an exception, just log warnings

        return config

    except (ParserError, ScannerError) as e:
        error_msg = (
            f"YAML syntax error in configuration file {config_path_str}: {str(e)}"
        )
        # Cannot use logger here reliably
        print(f"ERROR: Config loading: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        error_msg = f"Error loading configuration from {config_path_str}: {str(e)}"
        # Cannot use logger here reliably
        print(f"ERROR: Config loading: {error_msg}")
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
    mappings_path_obj = get_project_root() / "configs" / "mappings.yaml"
    mappings_path_str = str(mappings_path_obj)

    if not mappings_path_obj.exists():
        # Use print for safety during potential import issues
        print(f"ERROR: Mappings file not found: {mappings_path_str}")
        raise FileNotFoundError(f"Mappings file not found: {mappings_path_str}")

    try:
        with mappings_path_obj.open("r") as f:
            mappings = yaml.safe_load(f)

        if mappings is None:
            raise ValueError(f"Mappings file is empty: {mappings_path_str}")

        if not isinstance(mappings, dict):
            raise ValueError(
                f"Mappings must be a dictionary, got {type(mappings).__name__}"
            )

        # Validate mappings structure
        errors = validate_config_structure(mappings, MAPPINGS_SCHEMA)
        if errors:
            error_msg = "Mappings validation errors:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            # Use print for safety during potential import issues
            print(f"WARNING: Mappings validation: {error_msg}")
            # Don't raise an exception, just log warnings

        return mappings

    except (ParserError, ScannerError) as e:
        error_msg = f"YAML syntax error in mappings file {mappings_path_str}: {str(e)}"
        # Use print for safety during potential import issues
        print(f"ERROR: Mappings loading: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        error_msg = f"Error loading mappings from {mappings_path_str}: {str(e)}"
        # Use print for safety during potential import issues
        print(f"ERROR: Mappings loading: {error_msg}")
        raise


def get_data_path(
    data_type: str,
    dataset: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
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
        config = load_config()  # This will use the cached version after first call

    valid_data_types = ["raw", "processed", "external"]
    if data_type not in valid_data_types:
        raise ValueError(
            f"data_type must be one of {valid_data_types}, got '{data_type}'"
        )

    # Ensure data section exists
    if "data" not in config:
        raise KeyError("'data' section not found in configuration")

    # Ensure data type section exists
    if data_type not in config["data"]:
        raise KeyError(f"'{data_type}' section not found in data configuration")

    # Get the base path for the data type
    if dataset is None:
        try:
            # Assume base_path is relative to project root if not absolute
            base_path_str = config["data"][data_type]["base_path"]
            base_path = Path(base_path_str)
            if not base_path.is_absolute():
                base_path = get_project_root() / base_path
            return str(base_path.resolve())
        except KeyError:
            raise KeyError(f"'base_path' not found in '{data_type}' data configuration")

    # Get the path for the specific dataset
    try:
        path_str = config["data"][data_type][dataset]
        path = Path(path_str)
    except KeyError:
        raise KeyError(
            f"Dataset '{dataset}' not found in configuration for {data_type} data"
        )

    # Convert relative paths to absolute paths based on project root
    if not path.is_absolute():
        path = get_project_root() / path

    return str(path.resolve())


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
    # Determine the correct path object
    if config_path is None:
        config_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)
    config_path_str = str(config_path_obj)  # For messages

    # Create directory if it doesn't exist
    os.makedirs(config_path_obj.parent, exist_ok=True)

    # Validate configuration
    errors = validate_config_structure(config, CONFIG_SCHEMA)
    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        # Use print instead of logger here too for safety during potential import issues
        print(f"WARNING: Config validation: {error_msg}")

    try:  # Ensure we open the Path object
        with config_path_obj.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        # Use print here too
        print(f"INFO: Configuration saved to {config_path_str}")
    except Exception as e:
        error_msg = f"Error saving configuration to {config_path_str}: {str(e)}"
        # Use print here too
        print(f"ERROR: Config saving: {error_msg}")
        raise
