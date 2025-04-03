"""
Configuration utilities for the MIMIC project.

Provides functions to:
- Get the project root directory.
- Load and validate the main configuration file (`config.yaml`).
- Load and validate the mappings file (`mappings.yaml`).
- Construct absolute paths to data files based on the configuration.
- Save configuration dictionaries back to YAML files.
"""

import logging  # Import logging for type hint
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set  # Added Set

import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

# We will import get_logger inside functions to avoid circular import if needed
# from .logger import get_logger
# logger = get_logger(__name__)

# Define expected configuration structure (top-level keys and required sub-keys)
# Using sets for efficient checking of required keys within sections.
CONFIG_SCHEMA: Dict[str, Set[str]] = {
    "logging": {"level", "file_output", "console_output"},
    "data": {
        "raw",
        "processed",
        "external",
    },  # 'base_path' is often implied or handled by get_data_path
    "features": {
        "demographic",
        "vitals",
        "lab_values",  # Renamed from 'labs' for clarity
        "medications",
        "procedures",
        "diagnoses",
        "temporal",
    },
    "models": {
        "readmission",
        "mortality",
        "los",
        "temporal_readmission",
    },  # Added temporal
    "evaluation": {"classification", "regression"},
    "api": {"host", "port", "debug", "model_path"},  # Added model_path
    "dashboard": {"host", "port", "debug"},
    "mlflow": {"experiment_name", "tracking_uri"},  # Added mlflow section
}

# Define expected mappings structure
MAPPINGS_SCHEMA: Dict[str, Set[str]] = {
    "lab_tests": {"common_labs", "mappings", "lab_name_variations"},  # Added variations
    "vital_signs": {"categories", "itemids"},
    "icd9_categories": {"ranges", "specific_codes"},
}


# Define project root at the module level for efficiency
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.

    Assumes the script is located within a standard project structure
    (e.g., src/utils/config.py).

    Returns:
        Path: Path object representing the project root directory.
    """
    return PROJECT_ROOT


def validate_config_structure(
    config: Dict[str, Any], schema: Dict[str, Set[str]], path: str = ""
) -> List[str]:
    """
    Recursively validate the structure of a configuration dictionary against a schema.

    Checks for missing required sections and subsections defined in the schema.

    Args:
        config (Dict[str, Any]): Configuration dictionary to validate.
        schema (Dict[str, Set[str]]): Schema dictionary where keys are section names
                                      and values are sets of required subsection keys.
        path (str, optional): Current path in the configuration for error messages
                              (used in recursive calls). Defaults to "".

    Returns:
        List[str]: A list of validation error messages. Empty if the structure is valid.
    """
    errors: List[str] = []
    current_path_prefix = f"{path}." if path else ""

    # Check for missing required top-level sections defined in the schema
    for section in schema:
        if section not in config:
            errors.append(f"Missing required section '{current_path_prefix}{section}'")

    # Check each section that exists in the config
    for section, value in config.items():
        if section in schema:
            # If schema expects subsections (value is a set) and config value is a dict
            if isinstance(schema[section], set) and isinstance(value, dict):
                # Check for missing required subsections within this section
                for subsection in schema[section]:
                    if subsection not in value:
                        errors.append(
                            f"Missing required subsection '{subsection}' in '{current_path_prefix}{section}'"
                        )
                # Optionally, could add recursive validation here if schema had nested dicts
                # errors.extend(validate_config_structure(value, schema[section], f"{current_path_prefix}{section}"))
            # If schema expects subsections but config value is not a dict
            elif isinstance(schema[section], set) and not isinstance(value, dict):
                errors.append(
                    f"Section '{current_path_prefix}{section}' should be a dictionary, but found {type(value).__name__}"
                )
        # Optionally, warn about sections present in config but not in schema
        # else:
        #     errors.append(f"Unexpected section '{current_path_prefix}{section}' found in configuration.")

    return errors


@lru_cache(maxsize=None)  # Cache the result to avoid repeated file reads and parsing
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, validate its structure, and cache the result.

    Args:
        config_path (Optional[str], optional): Path to the configuration file.
            If None, uses the default 'configs/config.yaml' relative to the project root.
            Defaults to None.

    Returns:
        Dict[str, Any]: The loaded and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty, malformed (not valid YAML or not a dictionary),
                    or missing required sections/subsections based on CONFIG_SCHEMA.
    """
    # Determine the correct path object
    if config_path is None:
        config_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)

    # Use the object for exists check and opening, convert to string for messages
    config_path_str = str(config_path_obj.resolve())  # Use resolved path in messages

    if not config_path_obj.exists():
        # Cannot use logger here reliably due to potential recursion during startup
        print(f"ERROR: Configuration file not found: {config_path_str}")
        raise FileNotFoundError(f"Configuration file not found: {config_path_str}")

    try:
        with config_path_obj.open("r", encoding="utf-8") as f:  # Specify encoding
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path_str}")
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration must be a dictionary, got {type(config).__name__} in {config_path_str}"
            )

        # Validate configuration structure
        errors = validate_config_structure(config, CONFIG_SCHEMA)
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            # Use print for safety during potential import cycles
            print(f"WARNING: Config validation: {error_msg}")
            # Consider raising ValueError if strict validation is required:
            # raise ValueError(error_msg)

        return config

    except (ParserError, ScannerError) as e:
        error_msg = (
            f"YAML syntax error in configuration file {config_path_str}: {str(e)}"
        )
        print(f"ERROR: Config loading: {error_msg}")  # Use print for safety
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading configuration from {config_path_str}: {str(e)}"
        print(f"ERROR: Config loading: {error_msg}")  # Use print for safety
        raise


@lru_cache(maxsize=None)  # Cache the result
def load_mappings() -> Dict[str, Any]:
    """
    Load clinical feature mappings from the 'configs/mappings.yaml' file.

    Validates the structure against MAPPINGS_SCHEMA.

    Returns:
        Dict[str, Any]: Mappings dictionary containing sections like 'lab_tests',
                        'vital_signs', and 'icd9_categories'.

    Raises:
        FileNotFoundError: If the mappings file does not exist.
        ValueError: If the mappings file is empty, malformed, or missing required sections.
    """
    mappings_path_obj = get_project_root() / "configs" / "mappings.yaml"
    mappings_path_str = str(mappings_path_obj.resolve())

    if not mappings_path_obj.exists():
        print(
            f"ERROR: Mappings file not found: {mappings_path_str}"
        )  # Use print for safety
        raise FileNotFoundError(f"Mappings file not found: {mappings_path_str}")

    try:
        with mappings_path_obj.open("r", encoding="utf-8") as f:  # Specify encoding
            mappings = yaml.safe_load(f)

        if mappings is None:
            raise ValueError(f"Mappings file is empty: {mappings_path_str}")
        if not isinstance(mappings, dict):
            raise ValueError(
                f"Mappings must be a dictionary, got {type(mappings).__name__} in {mappings_path_str}"
            )

        # Validate mappings structure
        errors = validate_config_structure(mappings, MAPPINGS_SCHEMA)
        if errors:
            error_msg = "Mappings validation errors:\n" + "\n".join(
                f"- {e}" for e in errors
            )
            print(f"WARNING: Mappings validation: {error_msg}")  # Use print for safety
            # Consider raising ValueError if strict validation is required

        return mappings

    except (ParserError, ScannerError) as e:
        error_msg = f"YAML syntax error in mappings file {mappings_path_str}: {str(e)}"
        print(f"ERROR: Mappings loading: {error_msg}")  # Use print for safety
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading mappings from {mappings_path_str}: {str(e)}"
        print(f"ERROR: Mappings loading: {error_msg}")  # Use print for safety
        raise


def get_data_path(
    data_type: str,
    dataset: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Construct the absolute path to a data file or directory based on the configuration.

    Resolves paths relative to the project root if they are not absolute in the config.

    Args:
        data_type (str): Type of data ('raw', 'processed', or 'external').
        dataset (Optional[str], optional): Specific dataset key within the data_type section
                                           (e.g., 'mimic_iii', 'patient_data', 'base_path').
                                           If None, returns the 'base_path' for the data_type.
                                           Defaults to None.
        config (Optional[Dict[str, Any]], optional): Configuration dictionary.
                                                     If None, loads the default configuration.
                                                     Defaults to None.

    Returns:
        str: Absolute path to the data file or directory.

    Raises:
        ValueError: If `data_type` is not 'raw', 'processed', or 'external'.
        KeyError: If the 'data' section, the specified `data_type` section, or the
                  requested `dataset` key (or 'base_path') is not found in the configuration.
    """
    if config is None:
        config = load_config()  # Uses cached version after first call

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

    # Determine the key to look up within the data_type section
    lookup_key = dataset if dataset is not None else "base_path"

    # Get the path string from the config
    try:
        path_str = config["data"][data_type][lookup_key]
        path = Path(path_str)
    except KeyError:
        raise KeyError(
            f"Dataset key '{lookup_key}' not found in configuration for '{data_type}' data"
        )
    except TypeError as e:
        # Handle cases where config["data"][data_type] might not be a dictionary
        raise KeyError(
            f"Configuration for '{data_type}' data is not structured correctly: {e}"
        ) from e

    # Convert relative paths to absolute paths based on project root
    if not path.is_absolute():
        path = get_project_root() / path

    # Return the resolved, absolute path as a string
    return str(path.resolve())


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Validates the configuration structure before saving. Creates the directory
    if it doesn't exist.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        config_path (Optional[str], optional): Path to save the configuration file.
            If None, uses the default 'configs/config.yaml'. Defaults to None.

    Raises:
        ValueError: If the configuration structure is invalid (logged as warning by default).
        Exception: If there is an error writing the file.
    """
    # Determine the correct path object
    if config_path is None:
        config_path_obj = get_project_root() / "configs" / "config.yaml"
    else:
        config_path_obj = Path(config_path)
    config_path_str = str(config_path_obj.resolve())  # For messages

    # Create directory if it doesn't exist
    os.makedirs(config_path_obj.parent, exist_ok=True)

    # Validate configuration (optional, but good practice before saving)
    errors = validate_config_structure(config, CONFIG_SCHEMA)
    if errors:
        error_msg = "Configuration validation errors (saving anyway):\n" + "\n".join(
            f"- {e}" for e in errors
        )
        # Use print for safety during potential import cycles
        print(f"WARNING: Config validation: {error_msg}")
        # Consider raising ValueError here if saving invalid config should be prevented

    try:
        with config_path_obj.open("w", encoding="utf-8") as f:  # Specify encoding
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        # Use print here too
        print(f"INFO: Configuration saved to {config_path_str}")
    except Exception as e:
        error_msg = f"Error saving configuration to {config_path_str}: {str(e)}"
        print(f"ERROR: Config saving: {error_msg}")  # Use print for safety
        raise


def get_log_level_from_config(config: Optional[Dict[str, Any]] = None) -> int:
    """
    Retrieves the logging level from the configuration.

    Defaults to logging.INFO if not specified or invalid.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration dictionary.
                                                     If None, loads default. Defaults to None.

    Returns:
        int: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    if config is None:
        config = load_config()
    level_str = config.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    if not isinstance(level, int):  # Fallback if getattr fails unexpectedly
        print(
            f"WARNING: Invalid log level '{level_str}' in config. Defaulting to INFO."
        )
        level = logging.INFO
    return level
